import torch
import torch.nn as nn
from encoder_decoder import Encoder, Decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, model_dim, num_of_heads, d_ff, n_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, model_dim, num_of_heads, d_ff, n_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, model_dim, num_of_heads, d_ff, n_layers, dropout)
        self.output_layer = nn.Linear(model_dim, tgt_vocab_size)

    def forward(self, src_input, tgt_input, src_mask, tgt_mask):
        memory = self.encoder(src_input, src_mask)
        output = self.decoder(tgt_input, memory, src_mask, tgt_mask)
        output = self.output_layer(output)
        return output
    