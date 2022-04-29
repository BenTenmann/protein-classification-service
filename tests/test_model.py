import jax.numpy as jnp
import pytest

from protein_classification import model


class Cases:
    RESNET = [
        (
            {
                'num_embeddings': 25,
                'embedding_dim': 64,
                'residual_block_def': {
                    'input_features': 64,
                    'block_features': 128,
                    'kernel_size': (9,),
                    'dilation': 3,
                    'padding': 'same'
                },
                'n_residual_blocks': 4,
                'num_labels': 17_929
            },
            {'input': (64, 256), 'output': (64, 17_929)}
        ),
        (
            {
                'num_embeddings': 25,
                'embedding_dim': 32,
                'residual_block_def': {
                    'input_features': 32,
                    'block_features': 512,
                    'kernel_size': (3,),
                    'dilation': 1,
                    'padding': 'same'
                },
                'n_residual_blocks': 2,
                'num_labels': 100
            },
            {'input': (32, 128), 'output': (32, 100)}
        )
    ]


@pytest.mark.parametrize(['test_case', 'target'], Cases.RESNET)
class TestResNet:
    def test_base(self, test_case, target, rng):
        resnet = model.ResNet(**test_case)
        (N, L) = target.get('input')
        (_, C) = target.get('output')

        X = jnp.ones((N, L))
        params = resnet.init(rng, X)
        y_hat, _ = resnet.apply(params, X, mutable=['batch_stats'])
        assert y_hat.shape == (N, C)
