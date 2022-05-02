import jax.numpy as jnp
import pytest

from protein_classification import utils
from protein_classification.tokenize import Tokenizer


@pytest.fixture
def mock_read_json(monkeypatch):
    monkeypatch.setattr(
        'protein_classification.utils.srsly.read_json',
        lambda *args, **kwargs: {'A': 1, 'S': 2, 'Q': 3}
    )


@pytest.fixture
def model_conf():
    def _conf():
        conf = dict(
            num_embeddings=25,
            embedding_dim=64,
            residual_block_def={
                'input_features': 64,
                'block_features': 128,
                'kernel_size': (9,),
                'dilation': 3,
                'padding': 'same',
                'use_running_avg': True
            },
            n_residual_blocks=4,
            num_labels=17_929
        )
        return conf
    return _conf


@pytest.fixture
def mock_read_bytes(monkeypatch):
    monkeypatch.setattr(
        'protein_classification.utils.Path.read_bytes',
        lambda *args, **kwargs: b''
    )


@pytest.fixture
def batch():
    batch = jnp.ones((1, 256))
    return batch


@pytest.fixture
def mock_from_bytes(monkeypatch, model_conf, batch, rng):
    from protein_classification.model import ResNet
    conf = model_conf()
    conf['residual_block_def']['use_running_avg'] = False
    model = ResNet(**conf)
    params = model.init(rng, batch)
    _, variables = model.apply(params, batch, mutable=['batch_stats'])

    monkeypatch.setattr(
        'protein_classification.utils.serialization.from_bytes',
        lambda *args, **kwargs: {**params, **variables}
    )


@pytest.fixture
def mock_read_yaml(monkeypatch, model_conf):
    monkeypatch.setattr(
        'protein_classification.utils.srsly.read_yaml',
        lambda *args, **kwargs: model_conf()
    )


def test_load_tokenizer(mock_read_json):
    tok = utils.load_tokenizer('some/path')
    assert type(tok) == Tokenizer
    assert hasattr(tok, 'token_map')


class TestLoadModel:
    def test_load_model(self, mock_read_bytes, mock_from_bytes, mock_read_yaml):
        model = utils.load_model('', '', (1, 256))
        assert callable(model)

    def test_model(self, mock_read_bytes, mock_from_bytes, mock_read_yaml, batch, model_conf):
        model = utils.load_model('', '', (1, 256))
        y_hat = model(batch)

        (_, C) = None, model_conf()['num_labels']
        (N, L) = batch.shape
        assert isinstance(y_hat, jnp.ndarray)
        assert y_hat.shape == (N, C)
