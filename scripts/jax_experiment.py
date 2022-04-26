import os
from itertools import product
from pathlib import Path
from typing import Sequence, Dict

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
WANDB_PROJECT_NAME = os.environ.get('WANDB_PROJECT_NAME', 'protein-classification-development')

DATA_FILES = ['source.npy', 'target.npy']
DATA_SPLITS = ['train', 'dev', 'test']
SOURCE_COLUMN = 'sequence'
TARGET_COLUMN = 'family_accession'
SEQUENCE_LENGTH = 256

NUM_TOKENS = 25
NUM_CLASSES = 10_000
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
NUM_EPOCHS = 80
MODEL_CONF = dict(
    num_embeddings=NUM_TOKENS,
    embedding_dim=32,
    residual_block_def={
        'input_features': 32,
        'block_features': 64,
        'kernel_size': (3,),
        'dilation': 2,
        'padding': 'same'
    },
    n_residual_blocks=4,
    num_labels=NUM_CLASSES
)
MODEL_EVAL_CONF = MODEL_CONF.copy()
MODEL_EVAL_CONF['residual_block_def']['use_running_avg'] = True

wandb.init(
    job_type='training',
    config={'model': MODEL_CONF,
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LEARNING_RATE,
            'sequence_length': SEQUENCE_LENGTH},
    project=WANDB_PROJECT_NAME,
    group='jax.ResNet',
)


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


@jax.jit
def train_step(state: train_state.TrainState, batch_stats: dict, batch: Dict[str, jnp.ndarray]) -> tuple:
    def loss_fn(params):
        logits_, batch_stats_ = ResNet(**MODEL_CONF).apply({'params': params, **batch_stats},
                                                           batch[SOURCE_COLUMN],
                                                           mutable=['batch_stats'])
        loss = cross_entropy_loss(logits=logits_, labels=batch[TARGET_COLUMN])
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
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    predictions, targets = [], []
    for perm in tqdm(perms):
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, batch_stats, metrics, class_labels = train_step(state, batch_stats, batch)
        wandb.log({'training': metrics})
        batch_metrics.append(metrics)
        src = jax.device_get(class_labels)
        tgt = jax.device_get(batch[TARGET_COLUMN][perm, ...])
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


def eval_model(params, batch_stats: dict, test_ds: Dict[str, jnp.ndarray]):
    loss, class_labels = eval_step(params, batch_stats, test_ds)
    src = jax.device_get(class_labels)
    tgt = jax.device_get(test_ds[TARGET_COLUMN])
    report = classification_report(tgt, src, output_dict=True)
    loss = jax.device_get(loss)
    return {**report, 'loss': loss.item()}


class ResidualBlock(nn.Module):
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
    model = ResNet(**MODEL_CONF)
    batch = jnp.ones((BATCH_SIZE, SEQUENCE_LENGTH))
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    params = model.init(init_rng, batch)['params']

    tx = optax.adam(learning_rate=LEARNING_RATE)
    train_state_ = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    datasets = get_datasets()
    batch_stats = {}
    model_dir = Path(MODEL_DIR)
    if not model_dir.exists():
        raise FileNotFoundError(f'{model_dir} does not exist')
    for epoch in range(1, NUM_EPOCHS + 1):
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
        report = eval_model(params=train_state_.params,
                            batch_stats=batch_stats,
                            test_ds=datasets['dev'])
        wandb.log({'dev': {key: report[key] for key in ('macro avg', 'weighted avg', 'accuracy')}})
        print('dev epoch: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, report['loss'], report['accuracy'] * 100))
        with (model_dir / f'epoch-{epoch}.flx').open(mode='wb') as file:
            byte_str = serialization.to_bytes({'params': train_state_.params, **batch_stats})
            file.write(byte_str)

    report = eval_model(params=train_state_.params,
                        batch_stats=batch_stats,
                        test_ds=datasets['test'])
    wandb.log({'test': {key: report[key] for key in ('macro avg', 'weighted avg', 'accuracy')}})
    print('test results: loss: %.2f, accuracy: %.2f' % (
        report['loss'], report['accuracy'] * 100))


if __name__ == '__main__':
    main()