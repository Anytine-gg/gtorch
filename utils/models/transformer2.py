import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码层（支持动态长度）"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model//2,)
        pe = torch.zeros(1, max_len, d_model)  # (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DecoderOnlyTransformer(nn.Module):
    """纯 Decoder-Only Transformer（类似 GPT 结构）"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_len: int = 100,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    activation="gelu",
                    batch_first=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch)
        x_emb = self.embedding(x) * math.sqrt(self.d_model)  # (seq_len, batch, d_model)
        x_emb = self.pos_encoder(x_emb)
        causal_mask = self._generate_square_subsequent_mask(x.size(0)).to(x.device)

        # 逐层处理
        for layer in self.layers:
            # 手动跳过 Cross-Attention
            x_emb = layer(x_emb, memory=None, tgt_mask=causal_mask)

        logits = self.fc_out(x_emb)  # (seq_len, batch, vocab_size)
        return logits

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """生成因果掩码（防止看到未来信息）"""
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
