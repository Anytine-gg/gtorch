import os
import torch
import torch.nn.functional as F
from numpy import exp
from torch import GradScaler, autocast, mode, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.datasets.LangDataset import LangDataset
from gtorch.models.LTSM import LSTM_demo
from gtorch.torch import try_gpu
from gtorch.nlp.Vocab import Vocab, load_books

def lstm_train_epoch(model,cur_epoch,train_loader=None,train_dataset:LangDataset=None,save_path=None,vocab:Vocab=None):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
    scaler = GradScaler()
    model.train()
    epoch_loss = 0
    with tqdm(total=len(train_loader), desc=f"Epoch {cur_epoch+1}") as pbar:
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
            f"Epoch {cur_epoch+1}, Loss: {epoch_loss/len(train_loader)}, Perplexity :{exp(epoch_loss/len(train_loader))}"
        )
        print(
            lstm_predict_seq(
                model,
                predict,
                vocab,
                seq_len=32,
            )
        )
        torch.save(
            model.state_dict(),
            os.path.join(save_path,"lstm_model{epoch}.pth"),
        )

def lstm_predict_seq(model:nn.Module, input_seq, vocab: Vocab, seq_len=32):
    model.eval()
    device = next(model.parameters()).device
    num_layers = model.num_layers
    hidden_size = model.hidden_size
    with torch.no_grad():
        h = torch.zeros(num_layers, 1, hidden_size).to(device)
        c = torch.zeros(num_layers, 1, hidden_size).to(device)
        for char in input_seq:
            feature = torch.tensor(vocab[char]).reshape(1, 1).long().to(device)
            pred, (h, c) = model(feature, h0=h, c0=c)
        output_seq = input_seq + vocab.to_tokens(torch.argmax(pred.reshape(-1), dim=0))
        for _ in range(seq_len):
            feature = (
                torch.tensor(vocab[output_seq[-1]]).reshape(1, 1).long().to(device)
            )
            pred, (h, c) = model(feature, h0=h, c0=c)
            output_seq += vocab.to_tokens(torch.argmax(pred.reshape(-1), dim=0))
        return output_seq

    # 获取预测结果的索引
    predicted_indices = torch.argmax(predict, dim=1).cpu().numpy()
    # 将索引转换为词
    predicted_tokens = vocab.to_tokens(predicted_indices)

    return predicted_tokens
