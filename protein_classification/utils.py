import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import srsly
import torch
from torch.utils.data import DataLoader

from .dataset import (
    ProteinFamilyDataset,
    Tokenizer
)

__all__ = [
    'load_dataloader',
    'load_tokenizer',
    'now',
    'set_seed',
]


def set_seed(seed: int = 42) -> None:
    """
    Set the seed of the system.

    Returns
    -------
    None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_dataloader(path: Path or str,
                    tokenizer: Tokenizer,
                    label_map: dict,
                    batch_size: int,
                    **tokenizer_args) -> DataLoader:
    """
    Load a dataloader object using data from a specified file.

    Parameters
    ----------
    path: Path or str
        Path specifying input data in `.jsonl` format.
    tokenizer: Tokenizer
        The tokenizer used by the ProteinFamilyDataset object.
    label_map: Dict[str, int]
        The label map used  by the ProteinFamilyDataset object.
    batch_size: int
        The loader batch size/
    tokenizer_args: Any
        Keyword arguments passed to the tokenizer on call.

    Returns
    -------
    loader: DataLoader
        The initialised DataLoader object.

    Examples
    --------
    >>> tokenizer = load_tokenizer('path/to/token-map.json')
    >>> label_map = srsly.read_json('path/to/label-map.json')
    >>> dataloader = load_dataloader('path/to/train.jsonl', tokenizer, label_map, batch_size=10, max_length=512)
    """
    lines = srsly.read_jsonl(path)
    data_frame = pd.DataFrame(lines)
    generator = torch.Generator()
    generator.manual_seed(42)

    dataset = ProteinFamilyDataset(data_frame, tokenizer, label_map, **tokenizer_args)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        worker_init_fn=seed_worker,
                        generator=generator)
    return loader


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


def now() -> str:
    """
    Convenience wrapper for getting the current datetime in the `%Y-%m-%d-%X` format.

    Returns
    -------
    out: str
        The current datetime in the `%Y-%m-%d-%X` format.
    """
    out = datetime.now().strftime('%Y-%m-%d-%X')
    return out
