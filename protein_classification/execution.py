from functools import partial
from typing import Callable, Optional

import torch
import wandb
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

__all__ = [
    'OptimizerStep',
    'training',
    'testing',
    'validation'
]


class OptimizerStep:
    """
    Thin wrapper around a `torch.optim.Optimizer` object. Handles optimizer logic and is useful for writing a generic
    model execution loop (see below).

    Attributes
    ----------
    optim: torch.optim.Optimizer
        The optimizer object.

    Examples
    --------
    The model training use case:
    >>> model = nn.Linear(10, 1)
    >>> optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> optimizer = OptimizerStep(optim)
    >>> loss_fn = nn.BCELoss()
    >>> X = torch.randn(64, 10)
    >>> y = (torch.randn(64, 1) >= 0.).float()
    >>> y_hat = model(X)
    >>> loss = loss_fn(y_hat, y)
    >>> optimizer(loss)  # does update step
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optim = optimizer

    def __call__(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


def _loop(model: nn.Module,
          dataset: DataLoader,
          loss_fn: Optional[Callable] = lambda *_: None,
          optimizer: Optional[Callable] = lambda _: None,
          status: str = 'training'):
    """
    Generic model execution loop. Iterates over a dataloader and creates predictions from a model. Under certain
    parameters it computes a loss and performs update steps.

    Parameters
    ----------
    model: nn.Module
        The model used for generating predictions.
    dataset: DataLoader
        The dataloader object, providing the batched inputs and targets in a (X, y) tuple of tensors.
    loss_fn: Optional[Callable]
        A callable taking a number of positional arguments and returning either a `torch.Tensor` or `None`. The inputs
        to the callable are, in order, the model predictions and the model targets. If unset, the callable returns
        `None`, i.e. no loss is computed.
    optimizer: Optional[Callable]
        A callable taking a single positional argument and returning `None`. The argument to `optimizer` is the output
        of `loss_fn`. If unset, the callable returns `None` and no optimizer step is performed.
    status: str
        The execution status identifier. Usually one of: 'training', 'testing', 'validation'. Used by logger to group
        metrics.

    Returns
    -------
    model: nn.Module
        The input model.

    Examples
    --------
    >>> import protein_classification.utils as utils
    >>> dataloader = utils.load_dataloader('path/to/file.jsonl', ...)
    >>> model = _loop(model, dataloader, loss_fn, optimizer, status='training')
    """
    predictions = []
    targets = []
    for (X, y) in tqdm(dataset, desc=status, unit='batch'):
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        optimizer(loss)
        if loss:
            wandb.log({status: {'loss': loss.item()}})
        prd = torch.softmax(y_hat, dim=-1).argmax(dim=-1)

        predictions.extend(prd.tolist())
        targets.extend(y.tolist())
    report = classification_report(targets, predictions, output_dict=True)
    wandb.log({status: {k: report[k] for k in ('macro avg', 'weighted avg', 'accuracy')}})
    return model


# partial functions wrapping the generic loop for convenience
training = partial(_loop, status='training')
validation = torch.no_grad()(partial(_loop, status='validation'))
testing = torch.no_grad()(partial(_loop, status='testing'))
