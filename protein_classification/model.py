import math

import torch
from torch import nn

__all__ = [
    'TransformerClassifier',
    'MLP',
    'MLPOneHot'
]


class FlattenBatch(nn.Module):
    """
    Thin nn.Module wrapper for flattening a tensor of shape :math:`(N, L, H)` to :math:`(N, L \times H)` and then
    executing the module.

    Attributes
    ----------
    module: nn.Module
        The module which will be executed on the flattened tensors.

    Examples
    --------
    >>> model = nn.Linear(64, 1)
    >>> model = FlattenBatch(model)
    >>> X = torch.randn(10, 8, 8)
    >>> y_hat = model(X)
    """
    def __init__(self, module: nn.Module):
        super(FlattenBatch, self).__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        out = self.module(x.view(batch_size, -1))
        return out


class PositionalEncoding(nn.Module):
    """
    Positional encoding as defined in [1]. Adapted from `https://pytorch.org/tutorials/beginner/transformer_tutorial.html`

    References
    ----------
    [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I.,
        2017. Attention is all you need. Advances in neural information processing systems, 30.
    """
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(0).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Transformer Encoder as defined in [1] with a classification head.

    References
    ----------
    [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I.,
        2017. Attention is all you need. Advances in neural information processing systems, 30.
    """
    def __init__(self,
                 n_tokens: int,
                 d_model: int,
                 seq_len: int,
                 n_head: int,
                 dim_ff: int,
                 n_layers: int,
                 hidden_dim: int,
                 negative_slope: float,
                 num_labels: int,
                 dropout: float):
        """
        Initialise a TranformerClassifier object.

        Parameters
        ----------
        n_tokens: int
            The number of tokens, i.e. the vocabulary size.
        d_model: int
            The dimension of the token embeddings.
        seq_len: int
            The sequence length.
        n_head: int
            The number of heads in each Transformer encoder layer.
        dim_ff: int
            Feedforward dimension in each Transformer encoder layer.
        n_layers: int
            The number of Transformer encoder layers.
        hidden_dim: int
            The dimension of the hidden layer of the classification head.
        negative_slope: float
            The negative slope of the leaky ReL unit in the classification head.
        num_labels: int
            The number of output classes.
        dropout: float
            The unit dropout rate.
        """
        super(TransformerClassifier, self).__init__()
        self.embed_layer = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=seq_len)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_ff,
                batch_first=True
            ),
            num_layers=n_layers
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.proj_1 = nn.Linear(in_features=d_model, out_features=hidden_dim)
        self.l_relu = nn.LeakyReLU(negative_slope)
        self.proj_2 = nn.Linear(in_features=hidden_dim, out_features=num_labels)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classification head MLP with leaky ReLU non-linearity.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape :math:`(N, L, H)`, where :math:`L` is the sequence length and :math:`H` is the feature
            dimension.

        Returns
        -------
        out: torch.Tensor
            The unnormalised class scores of shape :math:`(N, C)`, where :math:`C` is the number of classes.
        """
        proj = self.dropout(self.proj_1(x))
        act = self.l_relu(proj)
        out = self.proj_2(act)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        The sequence encoding layer, i.e. the Transformer.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor of shape :math:`(N, L)`, where :math:`L` is the sequence length.

        Returns
        -------
        out: torch.Tensor
            The encoded sequence of shape :math:`(N, L, H)`, where :math:`H` is the feature dimension.
        """
        x = self.embed_layer(x)
        pos_enc = self.pos_enc(x)
        encoding = self.encoder(pos_enc)
        out = self.layer_norm(encoding + x)
        return out

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder and classification head. The encoder output is mean-pooled along the sequence dimension.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor of shape :math:`(N, L)`, where :math:`L` is the sequence length.

        Returns
        -------
        out: torch.Tensor
            The unnormalised class scores of shape :math:`(N, C)`, where :math:`C` is the number of classes.
        """
        encoding = self.encode(x)
        pooled = encoding.mean(dim=1)
        out = self.project(pooled)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embed(x)
        return out


class MLP(nn.Module):
    """
    Multi-layer Perceptron [1] for classification task.

    References
    ----------
    [1] Haykin, S., 1994. Neural networks: a comprehensive foundation, Prentice Hall PTR.
    """
    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 hidden_dim: int,
                 num_labels: int,
                 negative_slope: float,
                 dropout: float,
                 flatten_first: bool = False):
        """
        Initialise a basic MLP.

        Parameters
        ----------
        d_model: int
            The dimension of the token features.
        seq_len: int
            The sequence length.
        hidden_dim: int
            The dimension of the hidden layer.
        num_labels: int
            The number of output classes.
        negative_slope: float
            The negative slope of the leaky ReL unit.
        dropout: float
            The unit dropout rate.
        flatten_first: Optional[bool]
            Whether to flatten the sequence dimension (= concatenate tokens) before the first layer (`True`), or the
            before the second (`False`; default).
        """
        super(MLP, self).__init__()
        seq_len_1 = seq_len ** flatten_first  # if True, multiply the number of input dimensions
        self.proj_1 = nn.Linear(in_features=int(d_model * seq_len_1),
                                out_features=hidden_dim)
        self.l_relu = nn.LeakyReLU(negative_slope)
        seq_len_2 = seq_len ** (not flatten_first)  # otherwise, multiply the number of hidden dimensions
        self.proj_2 = nn.Linear(in_features=int(hidden_dim * seq_len_2),
                                out_features=num_labels)
        self.dropout = nn.Dropout(dropout)

        if flatten_first:
            self.proj_1 = FlattenBatch(self.proj_1)
        else:
            self.proj_2 = FlattenBatch(self.proj_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MLP forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape :math:`(N, L, H)`, where :math:`L` is the sequence length and :math:`H` is the token
            feature dimension.

        Returns
        -------
        out: torch.Tensor
            The unnormalised class scores of shape :math:`(N, C)`, where the :math:`C` is the number of classes.
        """
        x = self.dropout(x)
        proj = self.proj_1(x)
        act = self.l_relu(proj)
        out = self.proj_2(act)
        return out


class MLPOneHot(nn.Module):
    """
    Multi-layer Perceptron [1] for classification. Unlike `MLP`, `MLPOneHot` takes a sequence of class indices rather
    than a sequence of token embeddings.

    References
    ----------
    [1] Haykin, S., 1994. Neural networks: a comprehensive foundation, Prentice Hall PTR.
    """
    def __init__(self,
                 n_tokens: int,
                 embed_dim: int,
                 seq_len: int,
                 num_labels: int,
                 negative_slope: float,
                 dropout: float):
        """
        Initialise an MLPOneHot object.

        Parameters
        ----------
        n_tokens: int
            The number of tokens, i.e. the size of the vocabulary.
        embed_dim: int
            The token embedding dimension.
        seq_len: int
            The sequence length.
        num_labels:
            The number of output classes.
        negative_slope:
            The negative slope of the leaky ReL unit.
        dropout: float
            The unit dropout probability.
        """
        super(MLPOneHot, self).__init__()
        self.embed = nn.Embedding(num_embeddings=n_tokens, embedding_dim=embed_dim)
        self.l_relu = nn.LeakyReLU(negative_slope)
        self.proj = nn.Linear(in_features=embed_dim * seq_len, out_features=num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        embedding = self.dropout(self.embed(x))
        act = self.l_relu(embedding)
        out = self.proj(act.view(batch_size, -1))
        return out
