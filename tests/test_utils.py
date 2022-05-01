import pytest

from protein_classification import utils
from protein_classification.tokenize import Tokenizer


@pytest.fixture
def mock_read_json(monkeypatch):
    monkeypatch.setattr(
        'protein_classification.utils.srsly.read_json',
        lambda *args, **kwargs: {'A': 1, 'S': 2, 'Q': 3}
    )


def test_load_tokenizer(mock_read_json):
    tok = utils.load_tokenizer('some/path')
    assert type(tok) == Tokenizer
    assert hasattr(tok, 'token_map')
