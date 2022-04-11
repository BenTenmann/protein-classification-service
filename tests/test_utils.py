import pytest

from protein_classification import utils
from protein_classification.dataset import Tokenizer


@pytest.fixture
def tokenizer():
    def _method():
        return Tokenizer({'A': 1, 'S': 2, 'Q': 3})
    return _method


def test_set_seed():
    utils.set_seed()


def test_load_tokenizer(mock_read_json):
    tok = utils.load_tokenizer('some/path')
    assert type(tok) == Tokenizer
    assert hasattr(tok, 'token_map')
