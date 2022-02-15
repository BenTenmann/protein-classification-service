from functools import partial
from typing import Callable

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
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optim = optimizer

    def __call__(self, loss: torch.Tensor):
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


def _loop(model: nn.Module,
          dataset: DataLoader,
          loss_fn: Callable = lambda *_: None,
          optimizer: Callable = lambda _: None,
          status: str = 'training'):
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


training = partial(_loop, status='training')
validation = torch.no_grad()(partial(_loop, status='validation'))
testing = torch.no_grad()(partial(_loop, status='testing'))
