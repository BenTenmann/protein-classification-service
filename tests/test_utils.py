import jax
import jax.numpy as jnp
import pytest

from protein_classification import utils
from protein_classification.constants import DATA_SPLITS, SOURCE_COLUMN, TARGET_COLUMN


class Cases:
    BATCH_INDEX = [
        ({'dataset_size': 1024, 'batch_size': 32}, (32, 32)),
        ({'dataset_size': 2577, 'batch_size': 64}, (40, 64)),
        ({'dataset_size': 1311, 'batch_size': 8}, (163, 8))
    ]


def test_get_datasets(mock_jnp_load, array_shape):
    datasets = utils.get_datasets('')

    assert isinstance(datasets, dict)
    assert not set(datasets).difference(DATA_SPLITS)
    assert all(not set(split).difference([SOURCE_COLUMN, TARGET_COLUMN]) for split in datasets.values())

    def is_of_expected_shape(init: bool, arr: jnp.ndarray):
        cond = init and (arr.shape == array_shape)
        return cond
    assert jax.tree_util.tree_reduce(is_of_expected_shape, datasets, initializer=True)


@pytest.mark.parametrize(['test_case', 'target'], Cases.BATCH_INDEX)
def test_get_batch_indices(test_case, target, rng):
    perms = utils.get_batch_indices(rng, **test_case)
    assert isinstance(perms, jnp.ndarray)
    assert perms.shape == target


def test_load_checkpoint(mock_read_bytes, mock_from_bytes):
    params, variables = utils.load_checkpoint('', {})
    assert all(isinstance(elem, dict) for elem in [params, variables])
