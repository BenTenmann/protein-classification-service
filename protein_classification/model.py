from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

__all__ = [
    'ResNet'
]


class ResidualBlock(nn.Module):
    """
    Flax implementation of a residual block from the ResNet architecture [1]. Uses ReLU activation, batch norm and a
    dilated convolution.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition. In Proceedings of
        the IEEE conference on computer vision and pattern recognition (pp. 770-778).

    Attributes
    ----------
    input_features: int
        The number of incoming channels for the (dilated) convolution layer.
    block_features: int
        The number channels internal to the block i.e. the ones passed into the second (bottleneck) convolution layer.
        Generally, `input_features < block_features`.
    kernel_size: Sequence[int]
        A sequence of integers defining the kernel size for both convolution layers.
    dilation: int
        The size of the kernel dilation of the first convolution layer. If dilation is 1, then there is no dilation.
    use_running_avg: bool
        Define whether batch norm layers should use the running average. Set to `True` for model evaluation.
        Default=False
    """
    input_features: int
    block_features: int
    kernel_size: Sequence[int]
    dilation: int
    padding: str
    use_running_avg: bool = False

    @nn.compact
    def __call__(self, x):
        norm = nn.BatchNorm(use_running_average=self.use_running_avg)(x)
        act = nn.relu(norm)
        conv = nn.Conv(
            features=self.block_features,
            kernel_size=self.kernel_size,
            padding=self.padding.upper(),
            kernel_dilation=self.dilation
        )(act)
        norm = nn.BatchNorm(use_running_average=self.use_running_avg)(conv)
        act = nn.relu(norm)
        conv = nn.Conv(
            features=self.input_features,
            kernel_size=self.kernel_size,
            padding=self.padding.upper()
        )(act)
        return conv + x


class ResNet(nn.Module):
    """
    Flax implementation of the ResNet architecture [1]. Embeds integer sequences of shape :math:`(N, L)`, followed by a
    number of residual blocks, max pooling along the sequence dimension, linear projection and log-softmax
    normalization.

    References
    ----------
    [1] He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition. In Proceedings of
        the IEEE conference on computer vision and pattern recognition (pp. 770-778).

    Attributes
    ----------
    num_embeddings: int
        Number of input classes to be one-hot encoded.
    embedding_dim: int
        Dimension the one-hot class labels will be projected into. Must match with the input features on the residual
        block.
    residual_block_def: dict
        Definition of residual block. See `ResidualBlock`.
    n_residual_blocks: int
        Number of residual blocks.
    num_labels: int
        The number of output classes.
    """
    num_embeddings: int
    embedding_dim: int
    residual_block_def: dict
    n_residual_blocks: int
    num_labels: int

    @nn.compact
    def __call__(self, x):
        x = jax.nn.one_hot(x, num_classes=self.num_embeddings)
        x = nn.Dense(self.embedding_dim)(x)
        for _ in range(self.n_residual_blocks):
            x = ResidualBlock(**self.residual_block_def)(x)
        x = jnp.max(x, axis=1)
        x = nn.Dense(features=self.num_labels)(x)
        x = nn.log_softmax(x)
        return x
