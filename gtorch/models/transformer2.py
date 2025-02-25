import torch
import torch.nn as nn
import math
from tqdm import tqdm
from gtorch.nlp.Vocab import Vocab


class GPTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        # 自注意力
        x_norm = self.ln_1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_output

        # 前馈网络
        x_norm = self.ln_2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        return x


class GPTEncoder(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_heads, num_layers, max_len=512, dropout=0.1
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.layers = nn.ModuleList(
            [GPTBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0)
        )
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        token_embeddings = self.token_emb(input_ids)
        position_embeddings = self.pos_emb(positions)
        x = token_embeddings + position_embeddings
        x = self.dropout(x)

        # GPT 风格的下三角掩码
        attn_mask = self.mask[0, :seq_len, :seq_len]
        attn_mask = attn_mask == 0

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def train_model(
    model,
    data_loader,
    optimizer,
    criterion,
    device,
    epochs=1,
    vocab=None,
    prefix=None,
    max_len=None,
):
    model.train()  # 切换到训练模式
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}", unit="batch")
        for batch in progress_bar:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            
            # 假设 labels 与 input_ids 相同，用于下一词预测
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(data_loader)
        perplexity = math.exp(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}  Perplexity: {perplexity:.4f}")
        if vocab is not None:
            predicted = predict_str(
                model=model,
                prefix=prefix,
                vocab=vocab,
                device=device,
                max_length=max_len,
            )
            print(predicted)
def tokenize(text: str):
    import re
    # \w+ 匹配单词， [^\w\s] 匹配标点， [\s] 匹配空白（包括空格、换行等）
    # tokens = re.findall(r'\w+|[^\w\s]|[\s]', text)
    tokens = list(text)
    return tokens

def predict_str(model, prefix: str, vocab: Vocab, device, max_length=50):
    tokens = tokenize(prefix)
    input_ids = [[vocab[token] for token in tokens]]
    input_ids = torch.tensor(input_ids)
    output_ids = predict(model, input_ids, device, max_length)
    output_ids = output_ids.squeeze(dim=0).tolist()
    s = [vocab.to_tokens(idx) for idx in output_ids]
    s = "".join(s)
    return s

def predict(model, input_ids, device, max_length=50):
    model.eval()  # 切换到推理模式
    input_ids = input_ids.to(device)
    with torch.no_grad():
        
        for _ in range(max_length):
            outputs = model(input_ids)
            # 取最后一个时间步的预测结果
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
   
    return input_ids
