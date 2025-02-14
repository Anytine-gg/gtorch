import torch
import torch.nn.functional as F
from numpy import exp
from torch import GradScaler, autocast, mode, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.datasets.LangDataset import LangDataset

from utils.mytorch import try_gpu
from utils.nlp.Vocab import Vocab, load_books
from utils.models.simple_transformer import TransformerDecoderOnly, Classify

seq_len = 64
batch_size = 256

train_dataset = LangDataset(
    books_path="/root/projs/python/mytorch/enbooks/1",
    seq_len=seq_len,
    min_freq=10,
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
        seq = [vocab[c] for c in input_seq]
        seq = torch.tensor(seq).reshape(1,-1).to(try_gpu())
        seq_out = input_seq
        for _ in range(seq_len):
            predict_out = model(seq)    
            predicted_indices = torch.argmax(predict_out, dim=-1).cpu().numpy()
            last_predict = torch.tensor([[predicted_indices[0,-1]]], device=seq.device)
            seq = torch.cat([seq,last_predict],dim=1).to(try_gpu()) 
            seq_out += vocab.to_tokens(last_predict)
        
        return seq_out


def train(model, begin=0, num_epoch=2000):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=0.01, params=model.parameters(),weight_decay=1e-4)
    scaler = GradScaler()
    for epoch in range(begin, num_epoch):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epoch}") as pbar:
            for feature, label in train_loader:
                feature = feature.to(try_gpu())
                label = label.to(try_gpu())
                model.zero_grad()

                with autocast("cuda"):
                    predict = model(feature)
                    predict = predict.permute(0,2,1)
                
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
                    "Do you know about ",
                    vocab,
                    seq_len=200,
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
            input_seq="Do you know about ",
            vocab=vocab,
            seq_len=100,
        )
    )


if __name__ == "__main__":
    input_sz = output_sz = vocab_sz = len(vocab)
    embedding_size = 64
    model = nn.Sequential(
        TransformerDecoderOnly(
            vocab_size=vocab_sz,
            hidden_size=embedding_size,
            nhead=4,
            num_layers=4,
            ffn_hidden_size=256,
            dropout=0.02,
            max_seqlen=1024
        ),
        Classify(embedding_size,vocab_sz)
    )
    # model.load_state_dict(
    #     torch.load(
    #         "/root/projs/python/mytorch/saved_models/lstm/enbooks/lstm_model1999.pth",
    #         weights_only=True,
    #     )
    # )
    model.to(try_gpu())
    # get_predict(model)
    train(model,0)
