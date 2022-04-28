import os
import re
from itertools import product
from pathlib import Path
from typing import Sequence, Dict, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import serialization
from flax.training import train_state
from sklearn.metrics import classification_report
from tqdm import tqdm

DATA_DIR = os.environ.get('DATA_DIR')
MODEL_DIR = os.environ.get('MODEL_DIR')
RESUME_FROM = os.environ.get('RESUME_FROM')
WANDB_ENTITY = os.environ.get('WANDB_ENTITY', 'bentenmann')
WANDB_PROJECT_NAME = os.environ.get('WANDB_PROJECT_NAME', 'protein-classification-development')

DATA_FILES = ['source.npy', 'target.npy']
DATA_SPLITS = ['train', 'dev', 'test']
SOURCE_COLUMN = 'sequence'
TARGET_COLUMN = 'family_accession'
SEQUENCE_LENGTH = 256

NUM_TOKENS = 25
NUM_CLASSES = 17_929
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5
BATCH_SIZE = 64
NUM_EPOCHS = 80
MODEL_CONF = dict(
    num_embeddings=NUM_TOKENS,
    embedding_dim=64,
    residual_block_def={
        'input_features': 64,
        'block_features': 128,
        'kernel_size': (9,),
        'dilation': 3,
        'padding': 'same'
    },
    n_residual_blocks=4,
    num_labels=NUM_CLASSES
)
MODEL_EVAL_CONF = MODEL_CONF.copy()
MODEL_EVAL_CONF['residual_block_def']['use_running_avg'] = True


def get_datasets() -> dict:
    data_dir = Path(DATA_DIR)
    datasets = {split: {} for split in DATA_SPLITS}
    for split, filename in product(DATA_SPLITS, DATA_FILES):
        file_path = data_dir / split / filename
        data_array = jnp.load(str(file_path))
        column = {
            'source': SOURCE_COLUMN,
            'target': TARGET_COLUMN
        }[file_path.stem]
        datasets[split][column] = data_array
    return datasets


def cross_entropy_loss(*, logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=NUM_CLASSES)
    return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))


def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


def get_batch_indices(rng: jnp.ndarray, dataset_size: int, batch_size: int) -> jnp.ndarray:
    steps_per_epoch = dataset_size // batch_size

    perms = jax.random.permutation(rng, dataset_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    return perms


def load_checkpoint(path: Union[str, Path], target) -> tuple:
    if isinstance(path, str):
        path = Path(path)
    byte_str = path.read_bytes()
    variables = serialization.from_bytes(target, byte_str)
    params = variables['params']
    variables.pop('params')
    return params, variables


@jax.jit
def train_step(state: train_state.TrainState, batch_stats: dict, batch: Dict[str, jnp.ndarray]) -> tuple:
    def loss_fn(params):
        logits_, batch_stats_ = ResNet(**MODEL_CONF).apply({'params': params, **batch_stats},
                                                           batch[SOURCE_COLUMN],
                                                           mutable=['batch_stats'])

        def l2_norm(i, w):
            z = i + jnp.sum(w ** 2)
            return z

        loss = (
                cross_entropy_loss(logits=logits_, labels=batch[TARGET_COLUMN])
                + WEIGHT_DECAY * jax.tree_util.tree_reduce(l2_norm, params, initializer=0)
        )
        return loss, (logits_, batch_stats_)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (logits, batch_stats)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch[TARGET_COLUMN])
    class_labels = jnp.argmax(logits, -1)
    return state, batch_stats, metrics, class_labels


def train_epoch(state: train_state.TrainState,
                batch_stats: dict,
                train_ds: Dict[str, jnp.ndarray],
                batch_size: int,
                epoch: int,
                rng: jnp.ndarray) -> tuple:
    train_ds_size = len(train_ds[SOURCE_COLUMN])
    perms = get_batch_indices(rng, train_ds_size, batch_size)
    batch_metrics = []
    predictions, targets = [], []
    for perm in tqdm(perms):
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, batch_stats, metrics, class_labels = train_step(state, batch_stats, batch)
        wandb.log({'training': metrics})
        batch_metrics.append(metrics)
        src = jax.device_get(class_labels)
        tgt = jax.device_get(batch[TARGET_COLUMN])
        predictions.append(src)
        targets.append(tgt)
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    report = {
        key: val for key, val in classification_report(targets, predictions, output_dict=True).items()
        if key in {'macro avg', 'weighted avg', 'accuracy'}
    }
    wandb.log({'training': report})

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in ['loss', 'accuracy']
    }

    print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (
        epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

    return state, batch_stats


@jax.jit
def eval_step(params, batch_stats, batch):
    logits = ResNet(**MODEL_EVAL_CONF).apply({'params': params, **batch_stats}, batch[SOURCE_COLUMN])
    loss = cross_entropy_loss(logits=logits, labels=batch[TARGET_COLUMN])
    class_labels = jnp.argmax(logits, -1)
    return loss, class_labels


def eval_model(params, batch_stats: dict, test_ds: Dict[str, jnp.ndarray], batch_size: int, rng: jnp.ndarray):
    test_ds_size = len(test_ds[SOURCE_COLUMN])
    perms = get_batch_indices(rng, test_ds_size, batch_size)
    predictions, targets = [], []
    total_loss = 0
    for perm in tqdm(perms):
        batch = {k: v[perm, ...] for k, v in test_ds.items()}
        loss, class_labels = eval_step(params, batch_stats, batch)
        src = jax.device_get(class_labels)
        tgt = jax.device_get(batch[TARGET_COLUMN])
        predictions.append(src)
        targets.append(tgt)
        total_loss += jax.device_get(loss).item()
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    report = classification_report(targets, predictions, output_dict=True)
    return {**report, 'loss': total_loss / len(perms)}


class ResidualBlock(nn.Module):
    """
    Flax implementation of a residual block from the ResNet architecture [1]. Uses ReLU activation, batch norm and a
    dilated convolution.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition. In Proceedings of
        the IEEE conference on computer vision and pattern recognition (pp. 770-778).

    Attributes
    ----------
    input_features: int
        The number of incoming channels for the (dilated) convolution layer.
    block_features: int
        The number channels internal to the block i.e. the ones passed into the second (bottleneck) convolution layer.
        Generally, `input_features < block_features`.
    kernel_size: Sequence[int]
        A sequence of integers defining the kernel size for both convolution layers.
    dilation: int
        The size of the kernel dilation of the first convolution layer. If dilation is 1, then there is no dilation.
    use_running_avg: bool
        Define whether batch norm layers should use the running average. Set to `True` for model evaluation.
        Default=False
    """
    input_features: int
    block_features: int
    kernel_size: Sequence[int]
    dilation: int
    padding: str
    use_running_avg: bool = False

    @nn.compact
    def __call__(self, x):
        norm = nn.BatchNorm(use_running_average=self.use_running_avg)(x)
        act = nn.relu(norm)
        conv = nn.Conv(
            features=self.block_features,
            kernel_size=self.kernel_size,
            padding=self.padding.upper(),
            kernel_dilation=self.dilation
        )(act)
        norm = nn.BatchNorm(use_running_average=self.use_running_avg)(conv)
        act = nn.relu(norm)
        conv = nn.Conv(
            features=self.input_features,
            kernel_size=self.kernel_size,
            padding=self.padding.upper()
        )(act)
        return conv + x


class ResNet(nn.Module):
    """
    Flax implementation of the ResNet architecture [1]. Embeds integer sequences of shape :math:`(N, L)`, followed by a
    number of residual blocks, max pooling along the sequence dimension, linear projection and log-softmax
    normalization.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition. In Proceedings of
        the IEEE conference on computer vision and pattern recognition (pp. 770-778).

    Attributes
    ----------
    num_embeddings: int
        Number of input classes to be one-hot encoded.
    embedding_dim: int
        Dimension the one-hot class labels will be projected into. Must match with the input features on the residual
        block.
    residual_block_def: dict
        Definition of residual block. See `ResidualBlock`.
    n_residual_blocks: int
        Number of residual blocks.
    num_labels: int
        The number of output classes.
    """
    num_embeddings: int
    embedding_dim: int
    residual_block_def: dict
    n_residual_blocks: int
    num_labels: int

    @nn.compact
    def __call__(self, x):
        x = jax.nn.one_hot(x, num_classes=self.num_embeddings)
        x = nn.Dense(self.embedding_dim)(x)
        for _ in range(self.n_residual_blocks):
            x = ResidualBlock(**self.residual_block_def)(x)
        x = jnp.max(x, axis=1)
        x = nn.Dense(features=self.num_labels)(x)
        x = nn.log_softmax(x)
        return x


def main():
    wandb.init(
        job_type='training',
        config={'model': MODEL_CONF,
                'epochs': NUM_EPOCHS,
                'batch_size': BATCH_SIZE,
                'lr': LEARNING_RATE,
                'sequence_length': SEQUENCE_LENGTH},
        project=WANDB_PROJECT_NAME,
        group='jax.ResNet',
        entity=WANDB_ENTITY
    )
    model = ResNet(**MODEL_CONF)
    batch = jnp.ones((BATCH_SIZE, SEQUENCE_LENGTH))
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    variables = model.init(init_rng, batch)
    params = variables['params']
    batch_stats = {}
    if RESUME_FROM:
        params, batch_stats = load_checkpoint(RESUME_FROM, variables)
        eps = re.findall(r"epoch-([0-9]+)\.flx", RESUME_FROM)[0]
        prev_epoch = int(eps)
    else:
        prev_epoch = 0

    tx = optax.adam(learning_rate=LEARNING_RATE)
    train_state_ = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    datasets = get_datasets()
    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        raise FileNotFoundError(f'{model_dir} does not exist')
    for epoch in range(1 + prev_epoch, NUM_EPOCHS + 1):
        print(f'epoch: {epoch}/{NUM_EPOCHS}')
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        train_state_, batch_stats = train_epoch(
            state=train_state_,
            train_ds=datasets['train'],
            batch_stats=batch_stats,
            batch_size=BATCH_SIZE,
            epoch=epoch,
            rng=input_rng
        )
        # Evaluate on the test set after each training epoch
        rng, input_rng = jax.random.split(rng)
        report = eval_model(params=train_state_.params,
                            batch_stats=batch_stats,
                            test_ds=datasets['dev'],
                            batch_size=BATCH_SIZE,
                            rng=input_rng)
        wandb.log({'dev': {key: report[key] for key in ('macro avg', 'weighted avg', 'accuracy', 'loss')}})
        print('dev epoch: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, report['loss'], report['accuracy'] * 100))
        with (model_dir / f'epoch-{epoch}.flx').open(mode='wb') as file:
            byte_str = serialization.to_bytes({'params': train_state_.params, **batch_stats})
            file.write(byte_str)

    _, input_rng = jax.random.split(rng)
    report = eval_model(params=train_state_.params,
                        batch_stats=batch_stats,
                        test_ds=datasets['test'],
                        batch_size=BATCH_SIZE,
                        rng=input_rng)
    wandb.log({'test': {key: report[key] for key in ('macro avg', 'weighted avg', 'accuracy', 'loss')}})
    print('test results: loss: %.2f, accuracy: %.2f' % (
        report['loss'], report['accuracy'] * 100))


if __name__ == '__main__':
    main()
