import math

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

__all__ = [
    'TransformerClassifier',
    'LanguageModelClassifier',
    'ConvolutionClassifier',
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


class LanguageModelClassifier(nn.Module):
    """
    Language Model classifier (BERT) [1]. Uses pretrained language model from HuggingFace model hub [2] for fine-tuning
    on classification task.

    References
    ----------
    [1] Devlin, J., Chang, M.W., Lee, K. and Toutanova, K., 2018. Bert: Pre-training of deep bidirectional transformers
        for language understanding. arXiv preprint arXiv:1810.04805.
    [2] Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz,
        M. and Davison, J., 2019. Huggingface's transformers: State-of-the-art natural language processing.
        arXiv preprint arXiv:1910.03771. https://huggingface.co/models

    Examples
    --------
    >>> model = LanguageModelClassifier('Rostlab/prot_bert')  # downloads pretrained ProtBert model
    """
    def __init__(self, identifier: str, freeze_bert: bool = False, **kwargs):
        """
        Initialize a LanguageModelClassifier object.

        Parameters
        ----------
        identifier: str
            The model hub [2] identifier string for the pre-trained language model.
        freeze_bert: bool
            Boolean defining whether to deactivate gradients (i.e. "freeze") the BERT layers. If set to `True`, then
            only the classification head parameters will be updated. (default=False)
        kwargs
            Keyword arguments passed onto `AutoModelForSequenceClassification.from_pretrained`.
        """
        super(LanguageModelClassifier, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(identifier, **kwargs)
        for param in self.model.bert.parameters():
            param.requires_grad = not freeze_bert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x).logits
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


class ResidualBlock(nn.Module):
    def __init__(self,
                 input_channels: int,
                 block_channels: int,
                 kernel_size: tuple,
                 dilation: tuple,
                 padding: tuple or str):
        super(ResidualBlock, self).__init__()
        self.batch_norm_1 = nn.BatchNorm1d(input_channels)
        self.relu = nn.ReLU()
        self.dilated_convolution = nn.Conv1d(
            in_channels=input_channels,
            out_channels=block_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.batch_norm_2 = nn.BatchNorm1d(block_channels)
        self.bottleneck_convolution = nn.Conv1d(
            in_channels=block_channels,
            out_channels=input_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_1 = self.batch_norm_1(x)
        act_1 = self.relu(norm_1)
        conv_1 = self.dilated_convolution(act_1)
        norm_2 = self.batch_norm_2(conv_1)
        act_2 = self.relu(norm_2)
        conv_2 = self.bottleneck_convolution(act_2)
        out = conv_2 + x
        return out


class ConvolutionClassifier(nn.Module):
    _pooling_fn: set = {'amax', 'mean'}

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 n_residual_blocks: int,
                 residual_block_def: dict,
                 num_labels: int,
                 pooling: str = 'max'
                 ):
        super(ConvolutionClassifier, self).__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        residual_block_def.update({'input_channels': embedding_dim})
        residual_blocks = [ResidualBlock(**residual_block_def) for _ in range(n_residual_blocks)]
        self.residual_blocks = nn.Sequential(*residual_blocks)
        if pooling not in self._pooling_fn:
            raise ValueError(f'{repr(pooling)} is not a valid pooling function')

        self.pooling = getattr(torch, pooling)
        self.projection = nn.Linear(
            in_features=embedding_dim,
            out_features=num_labels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.embed(x)
        (N, L, C) = embedding.size()
        # the 1d convolution is done over the sequence dimension, so we need to have the sequence dimension as the last
        # dimension
        embedding = embedding.view(N, C, L)
        embedding = self.residual_blocks(embedding)
        embedding = self.pooling(embedding, dim=-1)
        out = self.projection(embedding)
        return out
