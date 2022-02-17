import pytest
import torch
import warnings
from sklearn.metrics._classification import UndefinedMetricWarning
from torch import nn

from protein_classification import execution
from protein_classification.model import MLPOneHot
from protein_classification.utils import load_tokenizer, load_dataloader


class Cases:
    OPTIM = [
        ((64, 10, 3), nn.MSELoss()),
        ((10, 3, 1), nn.MSELoss()),
        ((128, 10, 3), lambda *_: None)
    ]
    LOOP = [
        (nn.CrossEntropyLoss(), torch.optim.Adam, 'training'),
        (nn.CrossEntropyLoss(), lambda *args, **kwargs: None, 'validation'),
        (lambda *_: None, lambda *args, **kwargs: None, 'testing'),
        (lambda *_: None, torch.optim.Adam, 'breaking?')
    ]


@pytest.fixture
def mock_wandb_log(monkeypatch):
    monkeypatch.setattr(
        'protein_classification.execution.wandb.log',
        lambda *args, **kwargs: None
    )


@pytest.fixture
def model_init():
    def _method(shape):
        model = nn.Linear(*shape)
        return model
    return _method


@pytest.mark.parametrize(['shape', 'loss_fn'], Cases.OPTIM)
def test_optimizer(model_init, shape, loss_fn):
    (N, H, C) = shape
    model = model_init((H, C))
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    X = torch.randn(N, H)
    y = torch.randn(N, C)
    optimizer = execution.OptimizerStep(optim)

    y_hat = model(X)
    loss = loss_fn(y_hat, y)

    optimizer(loss)


@pytest.mark.parametrize(['loss_fn', 'optim', 'status'], Cases.LOOP)
def test_loop(mock_wandb_log, mock_read_jsonl, mock_read_json, mock_source_type, label_map, loss_fn, optim, status):
    tokenizer = load_tokenizer('')
    loader = load_dataloader('', tokenizer, label_map(), batch_size=1, max_length=7)

    model = MLPOneHot(
        n_tokens=4,
        embed_dim=128,
        num_labels=2,
        seq_len=7,
        negative_slope=0.3,
        dropout=0.1
    )
    opt = optim(model.parameters(), lr=0.01)
    optimizer = execution.OptimizerStep(opt)
    if opt is None:
        optimizer = optim

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UndefinedMetricWarning)
        model = execution._loop(model, loader, loss_fn, optimizer, status)
