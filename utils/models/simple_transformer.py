import torch
from torch import nn
import torch.nn.functional as F
class DecoderOnlyBlock(nn.Module):
    def __init__(self, hidden_size,nhead,ffn_hidden_size,dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size,nhead,dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size,ffn_hidden_size)
        self.linear2 = nn.Linear(ffn_hidden_size,hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self,x,mask=None):
        # x.shape = seq_len,batch_size,hidden_size

        # mask self-attention

        attention_output,_ = self.attention(x,x,x,attn_mask=mask)
        x = x + self.dropout1(attention_output)
        x = self.ln1(x)
        #FFN
        ffn_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(ffn_output)
        x = self.ln2(x)
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
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,hidden_size)
        self.layers = nn.Sequential()
        self.layers = nn.Sequential(
            *[
                DecoderOnlyBlock(hidden_size, nhead, ffn_hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.ln = nn.LayerNorm(hidden_size)
    def forward(self,x):
        # shape: batch_size seq_len hidden_size
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        x = self.embedding(x)
        x = x.permute(1,0,2)
        for layer in self.layers:
            x =  layer(x,mask = mask)
        return self.ln(x) #单decoder做自回归一般再加一层layer norm (gpt)
