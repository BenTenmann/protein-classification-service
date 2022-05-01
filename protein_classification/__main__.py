import re
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import serialization
from flax.training import train_state

from .constants import (
    DATA_DIR,
    MODEL_CONF,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    SEQUENCE_LENGTH,
    WANDB_PROJECT_NAME,
    WANDB_ENTITY,
    RESUME_FROM,
    MODEL_DIR
)
from .execution import train_epoch, eval_model
from .model import ResNet
from .utils import load_checkpoint, get_datasets


def main():
    """Entrypoint for ResNet training. See `__init__.py` for variable definitions."""
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

    datasets = get_datasets(DATA_DIR)
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
