import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import DEVICE, TYPES

SOURCE_TYPE = os.environ.get('SOURCE_TYPE', 'FLOAT')
TARGET_TYPE = os.environ.get('TARGET_TYPE', 'LONG')


class Tokenizer:
    """
    The sequence tokenizer object. Maps a string of letters into a list of (of lists) of numerics, based on a provided
    token map.
    Sequences can be padded and truncated to give uniform length. If an unknown token is encountered, the pad token is
    inserted. The pad token is hard coded and depends on the token map value type: if the values are integers, then the
    pad token is 0, otherwise it is a list of 0s of length `len(value)`. Thus, 0 cannot be used in the token map.

    Attributes
    ----------
    token_map: Dict[str, int or List[float]]
        The mapping from alphabetical to numeric representation.
    feature_dim: int
        The dimensionality of the numeric token representation. Is 0 when index tokens are used.

    Examples
    --------
    A simple example; notice we avoid 0 in the token map:
    >>> token_map = {'A': 1, 'G': 2}
    >>> tokenizer = Tokenizer(token_map)
    >>> tokenizer('AGGAX')
    ... [1, 2, 2, 1, 0]

    The 'X' token is mapped to 0, which is the pad token.

    We can also specify truncation and padding etc.:
    >>> tokenizer('AGGAX', padding='max_length', truncation=True, max_length=7)
    ... [1, 2, 2, 1, 0, 0, 0]
    >>> tokenizer('AGGAX', padding='max_length', truncation=True, max_length=4)
    ... [1, 2, 2, 1]

    Notice that we are exclusively right padding / truncating.
    """
    def __init__(self, token_map: Dict[str, int or List[float]]):
        """
        Initialise a Tokenizer object.

        Parameters
        ----------
        token_map: Dict[str, int ot List[float]]
            The mapping from alphabetical to numeric representation.
        """
        self.token_map = token_map

        elem, *_ = token_map.values()
        self._pad_token = 0 if type(elem) == int else [0] * len(elem)

    def __call__(self, sequence: str, padding: str, truncation: bool, max_length: int) -> list:
        sequence_len = len(sequence)

        toks = []
        toks.extend([self.token_map.get(token, self._pad_token) for token in sequence])
        if padding == 'max_length':
            toks.extend([self._pad_token] * (max_length - sequence_len))

        if truncation:
            toks = toks[:max_length]
        return toks

    @property
    def feature_dim(self) -> int:
        if type(self._pad_token) == int:
            return 0
        out = len(self._pad_token)
        return out


class ProteinFamilyDataset(Dataset):
    """
    Dataset class for protein family classification training.

    Attributes
    ----------
    data_source: pd.DataFrame
        The dataframe holding both the examples and the labels, in a row-wise association.
    tokenizer: Tokenizer
        The tokenizer object used to map the input sequences from alphabetical to numeric space.
    label_map: Dict[str, int]
        The string to index mapping of the example labels.
    tokenizer_args: Dict[str, Any]
        The arguments passed to the tokenizer upon call.
    source_column: str
        The column name of the model input. Set through environment variable `SOURCE_COLUMN`.
    target_column: str
        The column name of the model targets. Set through environment variable `TARGET_COLUMN`
    source_type: torch.dtype
        The tensor type for the model inputs. Set through environment variable `SOURCE_TYPE`
    target_type: torch.dtype
        The tensor type for the model targets. Set through environment variable `TARGET_TYPE`

    Examples
    --------
    The model training use case:
    >>> df = pd.DataFrame({'sequence': ['AASQ'], 'family_accession': ['PFxxxxx.y']})
    >>> token_map = {'A': 1, 'S': 2, 'Q': 3}
    >>> tokenizer = Tokenizer(token_map)
    >>> label_map = {'PFxxxxx.y': 0}
    >>> dataset = ProteinFamilyDataset(df, tokenizer, label_map, max_length=8)
    >>> dataset[0]  # X, y
    ... (tensor([1, 1, 2, 3, 0, 0, 0, 0]), tensor(0))
    """
    source_column: str = os.environ.get('SOURCE_COLUMN', 'sequence')
    target_column: str = os.environ.get('TARGET_COLUMN', 'family_accession')
    source_type: torch.dtype = TYPES.get(SOURCE_TYPE)
    target_type: torch.dtype = TYPES.get(TARGET_TYPE)

    def __init__(self, data_source: pd.DataFrame, tokenizer: Tokenizer, label_map: Dict[str, int], **tokenizer_args):
        """
        Initialise a ProteinFamilyDataset object. Be aware of environment variable dependency of column name and tensor
        type used by the object.

        Parameters
        ----------
        data_source: pd.DataFrame
            The dataframe holding both the examples and the labels, in a row-wise association.
        tokenizer: Tokenizer
            The tokenizer object used to map the input sequences from alphabetical to numeric space.
        label_map: Dict[str, int]
            The string to index mapping of the example labels.
        tokenizer_args: Dict[str, Any]
            The arguments passed to the tokenizer upon call.
        """
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.tokenizer_args = {'padding': 'max_length', 'truncation': True, **tokenizer_args}

    def __len__(self):
        return len(self.data_source)

    def _tokenize(self, sequence: str) -> List[int or List[float]]:
        out = self.tokenizer(sequence, **self.tokenizer_args)
        return out

    def _cast_sequence_to_tensor(self, sequence: str) -> torch.Tensor:
        toks = self._tokenize(sequence)
        out = torch.tensor(toks, dtype=self.source_type, device=DEVICE)
        return out

    def _cast_label_to_tensor(self, label: str) -> torch.Tensor:
        idx = self.label_map.get(label)
        out = torch.tensor(idx, dtype=self.target_type, device=DEVICE)
        return out

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.data_source.iloc[item]
        src = record[self.source_column]
        tgt = record[self.target_column]

        source = self._cast_sequence_to_tensor(src)
        target = self._cast_label_to_tensor(tgt)

        return source, target
