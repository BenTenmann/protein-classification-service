import random
from pathlib import Path

import srsly
import torch

from .dataset import (
    Tokenizer
)

__all__ = [
    'load_tokenizer',
    'set_seed',
]


def set_seed() -> None:
    """
    Set the seed of the system.

    Returns
    -------
    None
    """
    random.seed(42)
    torch.manual_seed(42)


def load_tokenizer(path: Path or str) -> Tokenizer:
    """
    Load a Tokenizer object by specifying a target token map.

    Parameters
    ----------
    path: Path or str
        Path specifying the target token map in `.json` format.

    Returns
    -------
    out: Tokenizer
        The initialised Tokenizer object.
    """
    token_map = srsly.read_json(path)
    out = Tokenizer(token_map)
    return out
