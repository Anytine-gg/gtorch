import torch
from torch import nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


class DecoderOnlyBlock(nn.Module):
    def __init__(self, hidden_size, nhead, ffn_hidden_size, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, nhead, dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size, ffn_hidden_size)
        self.linear2 = nn.Linear(ffn_hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x.shape = seq_len,batch_size,hidden_size

        # Pre-LN结构
        norm_x = self.ln1(x)
        attention_output, _ = self.attention(norm_x, norm_x, norm_x, attn_mask=mask)
        x = x + self.dropout1(attention_output)

        norm_x = self.ln2(x)
        ffn_output = self.linear2(self.dropout(F.relu(self.linear1(norm_x))))
        x = x + self.dropout2(ffn_output)
        return x


class TransformerDecoderOnly(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=256,
        nhead=4,
        num_layers=4,
        ffn_hidden_size=512,
        dropout=0.1,
        max_seqlen=1024,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout, max_seqlen)
        self.layers = nn.Sequential()
        self.layers = nn.Sequential(
            *[
                DecoderOnlyBlock(hidden_size, nhead, ffn_hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.ln = nn.LayerNorm(hidden_size)
        self.classify = nn.Linear(hidden_size, vocab_size)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # shape: seq_len batch_size
        seq_len = x.size(0)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        mask = mask.to(x.device)
        x = self.embedding(x) * math.sqrt(self.hidden_size)

        for layer in self.layers:
            x = layer(x, mask=mask)
            # 单decoder做自回归一般再加一层layer norm (gpt)
        x = self.ln(x)
        x = self.classify(x)
        return x


class Classify(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        return self.linear(x)
