import pytest

from protein_classification import execution
from protein_classification.constants import BATCH_SIZE
from protein_classification.utils import get_datasets


@pytest.fixture
def train_state(base_model):
    import optax
    from flax.training import train_state

    from protein_classification.constants import LEARNING_RATE

    model, params = base_model
    tx = optax.adam(learning_rate=LEARNING_RATE)
    st = train_state.TrainState.create(apply_fn=model.apply, params=params['params'], tx=tx)
    return st


def test_train_epoch(train_state, base_model, rng, mock_wandb, mock_jnp_load_2):
    model, params = base_model
    datasets = get_datasets('')

    _ = execution.train_epoch(
        state=train_state,
        batch_stats=params['batch_stats'],
        train_ds=datasets['train'],
        batch_size=BATCH_SIZE,
        epoch=0,
        rng=rng
    )


def test_eval_model(base_model, train_state, mock_jnp_load_2, rng, mock_wandb):
    model, params = base_model
    datasets = get_datasets('')

    state, batch_stats = execution.train_epoch(
        state=train_state,
        batch_stats=params['batch_stats'],
        train_ds=datasets['train'],
        batch_size=BATCH_SIZE,
        epoch=0,
        rng=rng
    )
    _ = execution.eval_model(state.params, batch_stats, datasets['test'], BATCH_SIZE, rng)
