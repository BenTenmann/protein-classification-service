import pytest

import jax


@pytest.fixture
def rng():
    init_rng = jax.random.PRNGKey(0)
    return init_rng


@pytest.fixture
def array_shape():
    shape = (1024, 256)
    return shape


@pytest.fixture
def mock_jnp_load(monkeypatch, rng, array_shape):
    monkeypatch.setattr(
        'protein_classification.utils.jnp.load',
        lambda *args, **kwargs: jax.random.randint(rng, array_shape, 0, 25)
    )


@pytest.fixture
def mock_jnp_load_2(monkeypatch, rng):
    def jnp_load(file_path: str) -> jax.numpy.ndarray:
        shape, n_classes = ((128, 256), 25) if 'source' in file_path else ((128, 1), 17_929)
        mock_array = jax.random.randint(rng, shape, 0, n_classes)
        return mock_array

    monkeypatch.setattr(
        'protein_classification.utils.jnp.load',
        jnp_load
    )


@pytest.fixture
def mock_read_bytes(monkeypatch):
    monkeypatch.setattr(
        'protein_classification.utils.Path.read_bytes',
        lambda *args, **kwargs: b''
    )


@pytest.fixture
def mock_from_bytes(monkeypatch):
    monkeypatch.setattr(
        'protein_classification.utils.serialization.from_bytes',
        lambda *args, **kwargs: {'params': {}, 'batch_stats': {}}
    )


@pytest.fixture
def mock_wandb(monkeypatch):
    monkeypatch.setattr(
        'protein_classification.execution.wandb.log',
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        'protein_classification.execution.wandb.init',
        lambda *args, **kwargs: None
    )


@pytest.fixture
def base_model(array_shape, rng):
    from protein_classification import model, constants
    resnet = model.ResNet(**constants.MODEL_CONF)
    batch = jax.numpy.ones(array_shape)
    params = resnet.init(rng, batch)
    return resnet, params
