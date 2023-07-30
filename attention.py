import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num__of_heads):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num__of_heads == 0

        self.model_dim = model_dim
        self.num_of_heads = num__of_heads
        self.head_dim = model_dim // num__of_heads
        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.linear_o = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.linear_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.linear_o(context)
        return output


