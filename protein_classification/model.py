import math

import torch
from torch import nn

from .utils import flatten_batch

__all__ = [
    'ProteinFamilyClassifier',
    'MLP',
    'MLPOneHot'
]


class PositionalEncoding(nn.Module):
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


class ProteinFamilyClassifier(nn.Module):
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
        super(ProteinFamilyClassifier, self).__init__()
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
        proj = self.dropout(self.proj_1(x))
        act = self.l_relu(proj)
        out = self.proj_2(act)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_layer(x)
        pos_enc = self.pos_enc(x)
        encoding = self.encoder(pos_enc)
        out = self.layer_norm(encoding + x)
        return out

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        encoding = self.encode(x)
        pooled = encoding.mean(dim=1)
        out = self.project(pooled)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embed(x)
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, seq_len: int, hidden_dim: int, num_labels: int, negative_slope: float, dropout: float, flatten_first: bool = False):
        super(MLP, self).__init__()
        seq_len_1 = seq_len ** flatten_first
        self.proj_1 = nn.Linear(in_features=int(d_model * seq_len_1),
                                out_features=hidden_dim)
        self.l_relu = nn.LeakyReLU(negative_slope)
        seq_len_2 = seq_len ** (not flatten_first)
        self.proj_2 = nn.Linear(in_features=int(hidden_dim * seq_len_2),
                                out_features=num_labels)
        self.dropout = nn.Dropout(dropout)

        if flatten_first:
            self.proj_1 = flatten_batch(self.proj_1)
        else:
            self.proj_2 = flatten_batch(self.proj_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        proj = self.proj_1(x)
        act = self.l_relu(proj)
        out = self.proj_2(act)
        return out


class MLPOneHot(nn.Module):
    def __init__(self, n_tokens: int, embed_dim: int, seq_len: int, num_labels: int, negative_slope: float, dropout: float):
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
