import torch
from torch import nn


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=256,
        nhead=4,
        num_layers=4,
        ffn_hidden_size=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.decoder_layer = nn.TransformerDecoderLayer(
            hidden_size, nhead, ffn_hidden_size, dropout
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
fewfwef