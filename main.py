from ctypes import util
from numpy import exp
from pytz import utc
import torch
from torch import mode, nn
from zmq import REQ_RELAXED
from utils.nlp.Vocab import Vocab, load_books
from utils.datasets.LangDataset import LangDataset
from torch.utils.data import DataLoader, Dataset
from utils.models.LTSM import LSTM_demo
from utils.mytorch import try_gpu
import torch.nn.functional as F

seq_len = 32
batch_size = 128
num_layers = 2
hidden_size = 256
train_dataset = LangDataset(
    books_path="/root/projs/py/demo/enbooks", seq_len=seq_len, min_freq=0
)
vocab = train_dataset.vocab
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
print(vocab.vocab_size)


def predict_seq(model, input_seq, vocab: Vocab, seq_len=32):
    model.eval()
    with torch.no_grad():
        h = torch.zeros(num_layers, 1, hidden_size).to(try_gpu())
        c = torch.zeros(num_layers, 1, hidden_size).to(try_gpu())
        for char in input_seq:
            feature = torch.tensor(vocab[char]).reshape(1, 1)
            feature = F.one_hot(feature, num_classes=len(vocab)).float().to(try_gpu())
            pred, (h, c) = model(feature, h0=h, c0=c)
        output_seq = input_seq + vocab.to_tokens(torch.argmax(pred.reshape(-1), dim=0))
        for _ in range(seq_len):
            feature = torch.tensor(vocab[output_seq[-1]]).reshape(1, 1)
            feature = F.one_hot(feature, num_classes=len(vocab)).float().to(try_gpu())
            pred, (h, c) = model(feature, h0=h, c0=c)
            output_seq += vocab.to_tokens(torch.argmax(pred.reshape(-1), dim=0))
        return output_seq

    # 获取预测结果的索引
    predicted_indices = torch.argmax(predict, dim=1).cpu().numpy()
    # 将索引转换为词
    predicted_tokens = vocab.to_tokens(predicted_indices)

    return predicted_tokens


def train():
    num_epoch = 500
    ce_loss = nn.CrossEntropyLoss()
    input_sz = output_sz = vocab_sz = len(vocab)
    model = LSTM_demo(input_sz, hidden_size, num_layers, output_sz)
    model.to(try_gpu())
    optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())
    for epoch in range(num_epoch):
        model.train()
        for feature, label in train_loader:
            feature = feature.T
            feature = F.one_hot(feature, num_classes=vocab_sz).float()
            label = label.T
            label = label.reshape(-1)
            feature = feature.to(try_gpu())
            label = label.to(try_gpu())
            predict, (h, c) = model(feature)
            loss = ce_loss(predict, label)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch:", epoch)
        print(loss.item(), " ", exp(loss.item()))
        print(predict_seq(model, "my name is ", vocab))
        print()
        torch.save(
            model.state_dict(),
            f"/root/projs/py/demo/saved_models/lstm/enbooks/lstm_model{epoch}.pth",
        )


train()
