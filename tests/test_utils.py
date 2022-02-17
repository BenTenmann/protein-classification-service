import pytest
import torch

from protein_classification import utils
from protein_classification.dataset import Tokenizer


@pytest.fixture
def tokenizer():
    def _method():
        return Tokenizer({'A': 1, 'S': 2, 'Q': 3})
    return _method


def test_set_seed():
    utils.set_seed()


def test_load_dataloader(mock_read_jsonl, tokenizer, label_map, mock_source_type):
    tk = tokenizer()
    lb = label_map()
    loader = utils.load_dataloader('some/path.jsonl', tk, lb, batch_size=1, max_length=7)

    X, y = next(iter(loader))
    assert (X == torch.tensor([[1, 1, 2, 3, 0, 0, 0]], dtype=torch.long)).all()
    assert (y == torch.tensor([0], dtype=torch.long)).all()


def test_load_tokenizer(mock_read_json):
    tok = utils.load_tokenizer('some/path')
    assert type(tok) == Tokenizer
    assert hasattr(tok, 'token_map')


def test_now():
    tp = utils.now()

    assert type(tp) == str
