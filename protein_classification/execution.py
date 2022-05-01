import inspect
from functools import wraps
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax.training import train_state
from sklearn.metrics import classification_report
from tqdm import tqdm

from .constants import (
    NUM_CLASSES,
    MODEL_CONF,
    MODEL_EVAL_CONF,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    WEIGHT_DECAY,
)
from .model import ResNet
from .utils import get_batch_indices

__all__ = [
    'train_epoch',
    'eval_model'
]


def categorical_loss(loss_fn):
    # wraps loss function to cast class indices to one hot encodings
    _loss_fn_signature = {'logits', 'labels'}
    _fn_signature = inspect.signature(loss_fn)

    if _loss_fn_signature.difference(_fn_signature.parameters):
        raise ValueError(f'{loss_fn.__name__} does not have required `logits` and `labels` parameters')

    @wraps(loss_fn)
    def _loss_fn(*, logits: jnp.ndarray, labels: jnp.ndarray, **kwargs):
        one_hot = jax.nn.one_hot(labels, num_classes=NUM_CLASSES)
        loss = loss_fn(logits=logits, labels=one_hot, **kwargs)
        return loss
    return _loss_fn


@categorical_loss
def cross_entropy_loss(*, logits: jnp.ndarray, labels: jnp.ndarray):
    # logits are not normalized
    normalized_logits = nn.log_softmax(logits)
    loss = -jnp.mean(jnp.sum(labels * normalized_logits, axis=-1))
    return loss


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
    wandb.log({'train': report})

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
