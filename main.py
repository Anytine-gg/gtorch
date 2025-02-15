import torch
import torch.nn.functional as F
from numpy import exp
from torch import GradScaler, autocast, mode, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.datasets.LangDataset import LangDataset

from utils.mytorch import try_gpu
from utils.nlp.Vocab import Vocab, load_books,en_tokenize
from utils.models.simple_transformer import TransformerDecoderOnly, Classify

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


def predict_seq(model, input_seq, vocab: Vocab, seq_len=32):
    model.eval()
    with torch.no_grad():
        # 形状: [seq_len, batch_size=1]
        seq = [vocab[c] for c in en_tokenize(input_seq)]
        
        seq = torch.tensor(seq).unsqueeze(1).to(try_gpu())

        seq_out = input_seq
        for _ in range(seq_len):
            # 模型输出: [seq_len, batch_size, vocab_size]
            predict_out = model(seq)
           
            # 在 vocab_size 这个维度 argmax => [seq_len, batch_size]
            predicted_indices = torch.argmax(predict_out, dim=-1).cpu().numpy()

            # 取最后一个时间步的 token（序列维度在 0，所以 predicted_indices[-1, 0]）
            last_token_idx = predicted_indices[-1][0]
            last_predict = torch.tensor(
                [[last_token_idx]], device=seq.device
            )  # shape [1,1]

            # 拼接时要在 seq_len 这个维度上追加（dim=0）
            seq = torch.cat([seq, last_predict], dim=0)

            # 将索引转换为空间的 token
            seq_out += vocab.to_tokens(last_predict)

        return seq_out


def train(model, begin=0, num_epoch=2000):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        lr=0.0001, params=model.parameters(), weight_decay=0.01
    )
    scaler = GradScaler()
    for epoch in range(begin, num_epoch):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epoch}") as pbar:
            for feature, label in train_loader:
                feature = feature.T
                label = label.T
                feature = feature.to(try_gpu())
                label = label.to(try_gpu())
                model.zero_grad()

                with autocast("cuda"):
                    predict = model(feature)
                    predict = predict.permute(0, 2, 1)

                    loss = ce_loss(predict, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)
        with torch.no_grad():
            print(
                f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}, Perplexity :{exp(epoch_loss/len(train_loader))}"
            )
            print(
                predict_seq(
                    model,
                    "she was such a little girl that",
                    vocab,
                    seq_len=50,
                )
            )
            print()
            torch.save(
                model.state_dict(),
                f"/root/projs/python/mytorch/saved_models/trans/enbooks/trans_model{epoch}.pth",
            )


def get_predict(model):
    print(
        predict_seq(
            model=model,
            input_seq="she was such a little girl that",
            vocab=vocab,
            seq_len=100,
        )
    )


if __name__ == "__main__":
    input_sz = output_sz = vocab_sz = len(vocab)
    embedding_size = 128
    model = TransformerDecoderOnly(
        vocab_size=vocab_sz,
        hidden_size=embedding_size,
        nhead=4,
        num_layers=3,
        ffn_hidden_size=512,
        dropout=0.1,
        max_seqlen=1024,
    )
    # model.load_state_dict(
    #     torch.load(
    #         "/root/projs/python/mytorch/saved_models/trans/enbooks/trans_model19.pth",
    #         weights_only=True,
    #     )
    # )
    model.to(try_gpu())
    #get_predict(model)
    train(model,0)
