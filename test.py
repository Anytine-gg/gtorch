import torch
import torch.nn.functional as F
from torch import  nn
import torch.optim.adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.datasets.LangDataset import LangDataset

from utils.mytorch import try_gpu
from utils.nlp.Vocab import Vocab, load_books, en_tokenize
import utils.models.transformer2 as myTF



if __name__ == "__main__":
    seq_len = 16
    batch_size = 16

    train_dataset = LangDataset(
        books_path="/root/projs/python/mytorch/enbooks/1",
        seq_len=seq_len,
        min_freq=100,
        lang="en",
    )

    vocab = train_dataset.vocab

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    model = myTF.GPTEncoder(
        vocab_size=len(vocab),
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        max_len=512,
        dropout=0.1,
    )
    model.to(try_gpu())
    print(try_gpu())
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    myTF.train_model(
        model,
        train_loader,
        optim,
        nn.CrossEntropyLoss(),
        try_gpu(),
        10
        )
    