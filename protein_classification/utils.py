import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import srsly
import torch
from torch.utils.data import DataLoader

from .dataset import (
    ProteinFamilyDataset,
    Tokenizer
)

__all__ = [
    'flatten_batch',
    'load_dataloader',
    'load_tokenizer',
    'now',
    'set_seed',
]


def set_seed() -> None:
    random.seed(42)
    torch.manual_seed(42)


def load_dataloader(path: Path or str,
                    tokenizer: Tokenizer,
                    label_map: dict,
                    batch_size: int,
                    **tokenizer_args) -> DataLoader:
    lines = srsly.read_jsonl(path)
    data_frame = pd.DataFrame(lines)

    dataset = ProteinFamilyDataset(data_frame, tokenizer, label_map, **tokenizer_args)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def load_tokenizer(path: Path or str) -> Tokenizer:
    token_map = srsly.read_json(path)
    out = Tokenizer(token_map)
    return out


def now():
    out = datetime.now().strftime('%Y-%m-%d-%X')
    return out


def flatten_batch(fn):
    def _fn(x: torch.Tensor):
        batch_size = x.size(0)
        out = fn(x.view(batch_size, -1))
        return out
    return _fn
