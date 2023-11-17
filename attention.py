import torch
import torch.nn as nn



class SelfAttention(nn.Module):
    def __init__(self, model_dim, head_dim, seq_length, mask, dropout_probability=0.1):
        super(SelfAttention, self).__init__()
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.W_k = nn.Linear(model_dim, head_dim, bias=False)
        self.W_q = nn.Linear(model_dim, head_dim, bias=False)
        self.W_v = nn.Linear(model_dim, head_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer('tril', torch.tril(torch.ones(seq_length, seq_length)))
        self.attention_dropout = nn.Dropout(p=dropout_probability)
        self.mask = mask

    def forward(self, queries, keys, values):
        # batch_size, seq_length, embed_dim = input_embeddings
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        scores = torch.matmul(queries, keys.transpose(-2,-1)) / self.head_dim**0.5

        if self.mask:
            #tril = torch.tril(torch.ones((scores.shape[-1], scores.shape[-1])))
            scores = scores.masked_fill(self.tril==0, float('-inf'))

        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        attention_vectors = torch.matmul(attention_weights, values)

        return attention_vectors
    

class  MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_of_heads, seq_length, mask, dropout_probability=0.1):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_of_heads == 0

        self.model_dim = model_dim
        self.num_of_heads = num_of_heads
        self.head_dim = int(model_dim / num_of_heads)
        self.attention_heads = nn.ModuleList(SelfAttention(self.model_dim, self.head_dim, seq_length, mask, dropout_probability)
                                             for _ in range(num_of_heads))
        self.W_o = nn.Linear(self.num_of_heads*self.head_dim, self.model_dim, bias = False)
        self.mask = mask
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, queries, keys, values):
        heads = [attn_head(queries, keys, values) for attn_head in self.attention_heads]
        heads_concat = torch.cat(heads, dim = -1)
        attention_vectors = self.W_o(heads_concat)
        attention_vectors = self.dropout(attention_vectors)

        return attention_vectors

