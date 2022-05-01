from itertools import product
from pathlib import Path
from typing import Union

import jax
import jax.numpy as jnp
from flax import serialization

from .constants import (
    DATA_FILES,
    DATA_SPLITS,
    SOURCE_COLUMN,
    TARGET_COLUMN
)

__all__ = [
    'get_datasets',
    'get_batch_indices',
    'load_checkpoint'
]


def get_datasets(data_dir: Union[str, Path]) -> dict:
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
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
