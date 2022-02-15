import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import DEVICE


class Tokenizer:
    def __init__(self, token_map: dict):
        self.token_map = token_map

        elem, *_ = token_map.values()
        self._pad_token = [0] * len(elem)

    def __call__(self, sequence: str, padding: str, truncation: bool, max_length: int) -> list:
        sequence_len = len(sequence)

        toks = []
        toks.extend([self.token_map.get(token, self._pad_token) for token in sequence])
        if padding == 'max_length':
            toks.extend([self._pad_token for _ in range(max_length - sequence_len)])

        if truncation:
            toks = toks[:max_length]
        return toks

    @property
    def feature_dim(self) -> int:
        out = len(self._pad_token)
        return out


class ProteinFamilyDataset(Dataset):
    """
    Dataset class for protein sequences.
    """
    _source_column: str = os.environ.get('SOURCE_COLUMN', 'sequence')
    _target_column: str = os.environ.get('TARGET_COLUMN', 'family_accession')

    def __init__(self, data_source: pd.DataFrame, tokenizer: Tokenizer, label_map: dict, **tokenizer_args):
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.tokenizer_args = {'padding': 'max_length', 'truncation': True, **tokenizer_args}

    def __len__(self):
        return len(self.data_source)

    def _tokenize(self, sequence: str) -> list:
        out = self.tokenizer(sequence, **self.tokenizer_args)
        return out

    def _cast_sequence_to_tensor(self, sequence: str) -> torch.Tensor:
        toks = self._tokenize(sequence)
        out = torch.tensor(toks, dtype=torch.float32, device=DEVICE)
        return out

    def _cast_label_to_tensor(self, label: str) -> torch.Tensor:
        idx = self.label_map.get(label)
        out = torch.tensor(idx, dtype=torch.long, device=DEVICE)
        return out

    def __getitem__(self, item: int):
        record = self.data_source.iloc[item]
        src = record[self._source_column]
        tgt = record[self._target_column]

        source = self._cast_sequence_to_tensor(src)
        target = self._cast_label_to_tensor(tgt)

        return source, target
