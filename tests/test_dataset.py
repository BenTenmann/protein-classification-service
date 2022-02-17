import pandas as pd
import pytest
import torch

from protein_classification import dataset


class Cases:
    TOKENIZER = [
        ('AASQ', {'A': 1, 'S': 2, 'Q': 3}, 7, [1, 1, 2, 3, 0, 0, 0]),
        ('GTA', {'G': [0, 1], 'A': [1, 0]}, 5, [[0, 1], [0, 0], [1, 0], [0, 0], [0, 0]]),
        ('XDDCTY', {'D': 1}, 5, [0, 1, 1, 0, 0])
    ]


@pytest.fixture
def data_source():
    def _method():
        df = pd.DataFrame({'sequence': ['AASQ'], 'family_accession': ['XX']})
        return df
    return _method


@pytest.fixture
def tokenizer():
    def _method():
        tok = dataset.Tokenizer({'A': 1, 'S': 2, 'Q': 3})
        return tok
    return _method


@pytest.fixture
def label_map():
    def _method():
        mp = {'XX': 0}
        return mp
    return _method


@pytest.fixture
def dataset_attr():
    def _method():
        return (
            ('data_source', pd.DataFrame),
            ('tokenizer', dataset.Tokenizer),
            ('label_map', dict),
            ('tokenizer_args', dict),
            ('source_column', str),
            ('target_column', str),
            ('source_type', torch.dtype),
            ('target_type', torch.dtype)
        )
    return _method


@pytest.mark.parametrize(['sequence', 'token_map', 'max_length', 'target'], Cases.TOKENIZER)
def test_tokenizer(sequence, token_map, max_length, target):
    tokenizer = dataset.Tokenizer(token_map)

    result = tokenizer(sequence, padding='max_length', truncation=True, max_length=max_length)
    assert result == target

    val, *_ = tokenizer.token_map.values()
    assert tokenizer.feature_dim == (0 if type(val) == int else len(val))


class TestDataset:
    def test_base(self, data_source, tokenizer, label_map, dataset_attr):
        df = data_source()
        tk = tokenizer()
        lb = label_map()

        dset = dataset.ProteinFamilyDataset(df, tk, lb, max_length=10)
        assert all(
            hasattr(dset, attr) and type(getattr(dset, attr)) == type_ for attr, type_ in dataset_attr()
        )

        X, y = dset[0]
        assert type(X) == torch.Tensor
        assert type(y) == torch.Tensor

    def test_index_error(self, data_source, tokenizer, label_map):
        df = data_source()
        tk = tokenizer()
        lb = label_map()

        dset = dataset.ProteinFamilyDataset(df, tk, lb, max_length=10)
        with pytest.raises(IndexError):
            dset[1]
