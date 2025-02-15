import torch
import torch.nn.functional as F
from numpy import exp
from torch import GradScaler, autocast, mode, nn
import torch.optim.adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.datasets.LangDataset import LangDataset

from utils.mytorch import try_gpu
from utils.nlp.Vocab import Vocab, load_books,en_tokenize
from utils.models.simple_transformer import TransformerDecoderOnly, Classify
import utils.models.transformer2 as myTF

seq_len = 64
batch_size = 128

train_dataset = LangDataset(
    books_path="/root/projs/python/mytorch/enbooks/1",
    seq_len=seq_len,
    min_freq=100,
    lang="en",
)

vocab = train_dataset.vocab
print(f"vocab size: {len(vocab)}")
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)


@torch.no_grad()
def generate_text(model, prefix, max_len=50, temperature=1.0, device='cpu'):
    model.eval()
    generated = prefix.clone().to(device)  # (prefix_len, batch)
    
    for _ in range(max_len - prefix.size(0)):
        logits = model(generated)  # (current_len, batch, vocab_size)
        next_logits = logits[-1, :, :] / temperature  # (batch, vocab_size)
        next_tokens = torch.argmax(next_logits, dim=-1)  # (batch,)
        generated = torch.cat([generated, next_tokens.unsqueeze(0)], dim=0)  # 追加新 token
    
    return generated  # (max_len, batch)


def train_pure_decoder(model, dataloader, optimizer, device, epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 假设用 -100 表示填充符
    
    for epoch in range(epochs):
        total_loss = 0.0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for inputs, labels in dataloader:  # inputs和labels形状均为 (seq_len, batch)
                inputs = inputs.T
                labels = labels.T
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                logits = model(inputs)  # (seq_len, batch, vocab_size)
                
                # 计算损失：预测下一个 token（忽略最后一个位置）
                loss = criterion(
                    logits[:-1].reshape(-1, logits.size(-1)),  # (seq_len-1)*batch, vocab_size
                    labels[1:].reshape(-1)                      # (seq_len-1)*batch
                )
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}, Perplexity: {exp(total_loss/len(dataloader)):.4f}")
        
if __name__ == '__main__':
    model = myTF.DecoderOnlyTransformer(
        vocab_size=len(vocab),
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        max_len=256
        )
    model.to(try_gpu())
    optim = torch.optim.Adam(model.parameters(),lr = 0.001)
    train_pure_decoder(model,train_loader,optimizer=optim,device=try_gpu(),epochs=10)