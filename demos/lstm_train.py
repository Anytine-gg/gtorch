import torch
import torch.nn.functional as F
from numpy import exp
from torch import GradScaler, autocast, mode, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.datasets.LangDataset import LangDataset
from utils.models.LTSM import LSTM_demo
from utils.mytorch import try_gpu
from utils.nlp.Vocab import Vocab, load_books

seq_len = 200
batch_size = 64
num_layers = 1
hidden_size = 256
train_dataset = LangDataset(
    books_path="/root/projs/python/mytorch/books/2",
    seq_len=seq_len,
    min_freq=20,
    lang="zh",
)

vocab = train_dataset.vocab
print(f"vocab size: {len(vocab)}")
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)


def predict_seq(model, input_seq, vocab: Vocab, seq_len=32):
    model.eval()
    with torch.no_grad():
        h = torch.zeros(num_layers, 1, hidden_size).to(try_gpu())
        c = torch.zeros(num_layers, 1, hidden_size).to(try_gpu())
        for char in input_seq:
            feature = torch.tensor(vocab[char]).reshape(1, 1).long().to(try_gpu())
            pred, (h, c) = model(feature, h0=h, c0=c)
        output_seq = input_seq + vocab.to_tokens(torch.argmax(pred.reshape(-1), dim=0))
        for _ in range(seq_len):
            feature = (
                torch.tensor(vocab[output_seq[-1]]).reshape(1, 1).long().to(try_gpu())
            )
            pred, (h, c) = model(feature, h0=h, c0=c)
            output_seq += vocab.to_tokens(torch.argmax(pred.reshape(-1), dim=0))
        return output_seq



def train(model, begin=0, num_epoch=2000):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
    scaler = GradScaler()
    for epoch in range(begin, num_epoch):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epoch}") as pbar:
            for feature, label in train_loader:
                feature = feature.T.long()
                label = label.T
                feature = feature.to(try_gpu())
                label = label.to(try_gpu())
                model.zero_grad()

                with autocast("cuda"):
                    predict, (h, c) = model(feature)
                    predict = predict.permute(1, 2, 0)
                    label = label.T
                    loss = ce_loss(predict, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)
        with torch.no_grad():
            train_dataset.random_slice()
            print(
                f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}, Perplexity :{exp(epoch_loss/len(train_loader))}"
            )
            print(
                predict_seq(
                    model,
                    "今天是星期四，威我五十看看实力， ",
                    vocab,
                    seq_len=100,
                )
            )
            print()
            torch.save(
                model.state_dict(),
                f"/root/projs/python/mytorch/saved_models/lstm/enbooks/lstm_model{epoch}.pth",
            )


def get_predict(model):
    print(
        predict_seq(
            model=model,
            input_seq="有了这点简单的分析，我们再说祥子的地位，就象说——我们希望——一盘机器上的某种钉子那么准确了。祥子，在与“骆驼”这个外号发生关系以前，是个较比有自由的洋车夫，这就是说，他是属于年轻力壮，而且自己有车的那一类：自己的车，自己的生活，都在自己手里，高等车夫。这可绝不是件容易的事。一年，二年，至少有三四年；一滴汗，两滴汗，不知道多少万滴汗，才挣出那辆车。从风里雨里的咬牙，从饭里茶里的自苦，才赚出那辆车。那辆车是他的一切挣扎与困苦的总结果与报酬，象身经百战的武士的一颗徽章。在他赁人家的车的时候，他从早到晚，由东到西，由南到北，象被人家抽着转的陀",
            vocab=vocab,
            seq_len=4000,
        )
    )


if __name__ == "__main__":
    input_sz = output_sz = vocab_sz = len(vocab)
    embedding_size = 256
    model = LSTM_demo(
        input_size=input_sz,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_sz,
    )
    model.load_state_dict(
        torch.load(
            "/root/projs/python/mytorch/saved_models/lstm/enbooks/lstm_model1999.pth",
            weights_only=True,
        )
    )
    model.to(try_gpu())
    get_predict(model)
    # train(model,0)
