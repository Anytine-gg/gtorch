import random
import torch
from torch.utils.data import Dataset
from ..nlp.Vocab import Vocab, load_books, tokenize
import numpy as np


class LangDataset(Dataset):
    def __init__(self, books_path, seq_len,min_freq):
        super().__init__()
        self.books = load_books(books_path)
        self.books = tokenize(books=self.books)
        self.vocab = Vocab(tokenized_books=self.books,min_freq=min_freq)
        self.seq_len = seq_len
        self.data = []
        self.target = []
        self.random_slice()

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.vocab[self.data[index]]),torch.tensor(self.vocab[self.target[index]])
