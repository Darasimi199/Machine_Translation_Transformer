import torch
import torch.nn as nn
from attention import MultiHeadAttention


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, dropout_probability = 0.1):
        super(EmbeddingLayer, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding = nn.Embedding(sequence_length, embedding_dim)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, input_ids):
        _, seq_len = input_ids.shape
        word_embeddings = self.word_embedding(input_ids)

        positional_embeddings = self.positional_embedding(torch.arange(seq_len))
        embeddings = word_embeddings + positional_embeddings
        return self.dropout(embeddings)


class PositionWiseFeedforward(nn.Module):
    def __init__(self, model_dim, width_factor=4):
        super(PositionWiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(model_dim, width_factor*model_dim)
        self.linear2 = nn.Linear(width_factor*model_dim, model_dim)
        self.relu = nn.ReLU()

    def forward(self, input_tensors):
        output = self.linear1(input_tensors)
        output = self.relu(output)
        output = self.linear2(output)
        
        return output


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_of_heads, seq_length, dropout=0.1):
        super(EncoderLayer, self).__init__()
        assert model_dim % num_of_heads == 0, f'model_dim {model_dim} is not divisible by num_of_heads {num_of_heads}'
        self.head_dim = int(model_dim / num_of_heads)
        self.self_attn = MultiHeadAttention(model_dim, num_of_heads, seq_length, mask=False)
        self.feed_forward = PositionWiseFeedforward(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def _linear_projection(self, X):
        W = nn.Linear(X.size(-1), self.head_dim, bias=False)
        return W(X)


    def forward(self, input_tensors):
        attention_vectors = self.self_attn(self._linear_projection(input_tensors),
                                            self._linear_projection(input_tensors),
                                            self._linear_projection(input_tensors))
        
        attention_vectors = self.dropout(attention_vectors)
        attention_vectors = self.norm1(attention_vectors + input_tensors)

        feed_forward_output = self.feed_forward(attention_vectors)
        feed_forward_output = self.dropout(feed_forward_output)
        output = self.norm2(feed_forward_output + attention_vectors)

        return output

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_of_layers, seq_length, num_of_heads, dropout=0.1):
        super(Encoder, self).__init__()
        self.num_of_layers = num_of_layers
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_of_heads, seq_length, dropout) for _ in range(num_of_layers)])
        

    def forward(self, source_embeddings):
        output = source_embeddings
        for layer in self.layers:
            output = layer(output)
       
        return output 

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_of_heads, seq_length, dropout=0.1):
        super(DecoderLayer, self).__init__()
        assert model_dim % num_of_heads == 0, f'model_dim {model_dim} is not divisible by num_of_heads {num_of_heads}'
        self.head_dim = int(model_dim / num_of_heads)
        self.self_attn = MultiHeadAttention(model_dim, num_of_heads, seq_length, mask=True)
        self.src_attn = MultiHeadAttention(model_dim, num_of_heads, seq_length, mask=False)
        self.feed_forward = PositionWiseFeedforward(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def _linear_projection(self, X):
        W = nn.Linear(X.size(-1), self.head_dim, bias=False)
        return W(X)

    def forward(self, encoder_output, decoder_input):
        keys, queries, values = (self._linear_projection(decoder_input),
                                self._linear_projection(decoder_input))
        
        masked_attention_vectors = self.self_attn(keys, queries, values)
        masked_attention_vectors = self.dropout(masked_attention_vectors)
        masked_attention_vectors = self.norm1(masked_attention_vectors + decoder_input)
        
        attention_vectors = self.src_attn(self._linear_projection(encoder_output), queries, self._linear_projection(encoder_output))
        attention_vectors = self.dropout(attention_vectors)
        attention_vectors = self.norm2(attention_vectors + masked_attention_vectors)

        feed_forward_output = self.feed_forward(attention_vectors)
        feed_forward_output = self.dropout(feed_forward_output)
        output = self.norm3(feed_forward_output + attention_vectors)

        return output


class Decoder(nn.Module):
    def __init__(self, vocab_size, model_dim, num_of_layers, seq_length, num_of_heads, dropout=0.1):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(model_dim, vocab_size)
        self.num_of_layers = num_of_layers
        self.layers = nn.ModuleList([DecoderLayer(model_dim, num_of_heads, seq_length, dropout) for _ in range(num_of_layers)])

    def forward(self, encoder_output, target_embeddings):
        output = target_embeddings
        for layer in self.layers:
            output = layer(encoder_output, target_embeddings)
        
        return nn.functional.log_softmax(self.linear(output), dim=-1)