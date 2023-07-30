import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class PositionwiseFeedforward(nn.Module):
    def __init__(self, model_dim, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(model_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, model_dim)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_of_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_of_heads)
        self.feed_forward = PositionwiseFeedforward(model_dim, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x, mask):
        x_norm = self.norm1(x)
        context = self.self_attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout1(context)

        x_norm = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x_norm))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, model_dim, num_of_heads, d_ff, n_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pe = PositionalEncoding(model_dim)
        self.layers = nn.ModuleList([EncoderLayer(model_dim, num_of_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.dropout(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_of_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_of_heads)
        self.src_attn = MultiHeadAttention(model_dim, num_of_heads)
        self.feed_forward = PositionwiseFeedforward(model_dim, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, x, memory, src_mask, tgt_mask):
        x_norm = self.norm1(x)
        context = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout1(context)

        x_norm = self.norm2(x)
        context = self.src_attn(x_norm, memory, memory, src_mask)
        x = x + self.dropout2(context)

        x_norm = self.norm3(x)
        x = x + self.dropout3(self.feed_forward(x_norm))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, model_dim, num_of_heads, d_ff, n_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pe = PositionalEncoding(model_dim)
        self.layers = nn.ModuleList([DecoderLayer(model_dim, num_of_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.embedding(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.dropout(x)
        return x