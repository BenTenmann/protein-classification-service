import pytest
import torch
from torch import nn

from protein_classification import model


class Cases:
    FLATTEN = [
        [(10, 8, 8), (10, 1)],
        [(64, 15, 1), (64, 5)],
        [(5, 128, 32), (5, 3)]
    ]
    TRANSFORMER = [
        (
            {'n_tokens': 25,
             'd_model': 512,
             'seq_len': 512,
             'n_head': 8,
             'dim_ff': 2048,
             'n_layers': 2,
             'hidden_dim': 4096,
             'negative_slope': 0.3,
             'num_labels': 1000,
             'dropout': 0.01},
            {'input': (10, 512), 'output': (10, 1000)}
        ),
        (
            {'n_tokens': 12,
             'd_model': 128,
             'seq_len': 256,
             'n_head': 8,
             'dim_ff': 1024,
             'n_layers': 2,
             'hidden_dim': 2048,
             'negative_slope': 0.3,
             'num_labels': 3,
             'dropout': 0.01},
            {'input': (64, 256), 'output': (64, 3)}
        )
    ]
    MLP = [
        (
            {'d_model': 10,
             'seq_len': 512,
             'hidden_dim': 256,
             'num_labels': 1000,
             'negative_slope': 0.3,
             'dropout': 0.1,
             'flatten_first': True},
            {'input': (10, 512, 10), 'output': (10, 1000)}
        ),
        (
            {'d_model': 5,
             'seq_len': 256,
             'hidden_dim': 64,
             'num_labels': 10,
             'negative_slope': 0.3,
             'dropout': 0.1,
             'flatten_first': False},
            {'input': (10, 256, 5), 'output': (10, 10)}
        )
    ]
    MLP_ONE_HOT = [
        (
            {'n_tokens': 25,
             'embed_dim': 256,
             'seq_len': 512,
             'num_labels': 1000,
             'negative_slope': 0.3,
             'dropout': 0.1},
            {'input': (10, 512), 'output': (10, 1000)}
        ),
        (
            {'n_tokens': 5,
             'embed_dim': 64,
             'seq_len': 256,
             'num_labels': 10,
             'negative_slope': 0.3,
             'dropout': 0.1},
            {'input': (10, 256), 'output': (10, 10)}
        )
    ]


@pytest.mark.parametrize(['test_case', 'target'], Cases.FLATTEN)
class TestFlattenBatch:
    def test_base(self, test_case, target):
        (N, L, H) = test_case
        (_, C) = target
        module = nn.Linear(L * H, C)
        module = model.FlattenBatch(module)

        X = torch.randn(N, L, H)
        y_hat = module(X)
        assert y_hat.shape == target

    def test_runtime_error(self, test_case, target):
        (N, L, H) = test_case
        (_, C) = target
        module = nn.Linear(H, C)
        module = model.FlattenBatch(module)

        X = torch.randn(N, L, H)
        with pytest.raises(RuntimeError):
            module(X)


@pytest.mark.parametrize(['test_case', 'target'], Cases.TRANSFORMER)
class TestTransformerClassifier:
    def test_base(self, test_case, target):
        transformer = model.TransformerClassifier(**test_case)
        (N, L) = target.get('input')
        (_, C) = target.get('output')

        X = torch.randint(test_case.get('n_tokens'), size=(N, L))
        y_hat = transformer(X)
        assert y_hat.shape == (N, C)


@pytest.mark.parametrize(['test_case', 'target'], Cases.MLP)
class TestMLP:
    def test_base(self, test_case, target):
        mlp = model.MLP(**test_case)
        (N, L, H) = target.get('input')
        (_, C) = target.get('output')

        X = torch.randn(N, L, H)
        y_hat = mlp(X)
        assert y_hat.shape == (N, C)


@pytest.mark.parametrize(['test_case', 'target'], Cases.MLP_ONE_HOT)
class TestMLPOneHot:
    def test_base(self, test_case, target):
        mlp_oh = model.MLPOneHot(**test_case)
        (N, L) = target.get('input')
        (_, C) = target.get('output')

        X = torch.randint(test_case.get('n_tokens'), size=(N, L))
        y_hat = mlp_oh(X)
        assert y_hat.shape == (N, C)
