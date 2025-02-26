import random
import torch
from torch.utils.data import Dataset
from gtorch.nlp.Vocab import Vocab, load_books, tokenize, load_en_books
import numpy as np


class LangDataset(Dataset):
    def __init__(self, books_path, seq_len,min_freq,lang='en'):
        super().__init__()
        if lang == 'zh':
            self.books = load_books(books_path)
        else:
            self.books = load_en_books(books_path)
        # self.books = tokenize(books=self.books)
        self.vocab = Vocab(tokenized_books=self.books,min_freq=min_freq)
        self.seq_len = seq_len
        self.data = []
        self.target = []
        self.random_slice()

    def random_slice(self):
        shuffled_indices = list(range(len(self.books)))
        random.shuffle(shuffled_indices)
        rand_begin = random.randrange(0, self.seq_len)
        self.books[shuffled_indices[0]] = self.books[shuffled_indices[0]][rand_begin:]
        reshaped_books = []
        reshaped_targets = []
        for i in shuffled_indices:
            np_book = np.array(self.books[i])
            np_target = np_book[1:]
            book_len = len(np_book)
            target_len = book_len-1
            drop = book_len % self.seq_len
            if drop != 0:
                np_book = np_book[:-drop]
            drop = target_len % self.seq_len
            if drop != 0:
                np_target = np_target[:-drop]
            
            np_book = np_book.reshape((-1, self.seq_len))
            np_target = np_target.reshape((-1,self.seq_len))
            if len(np_book) != len(np_target):
                np_book = np_book[:-1]
            reshaped_books.append(np_book)
            reshaped_targets.append(np_target)
        self.data = np.concatenate(reshaped_books, axis=0)
        self.target = np.concatenate(reshaped_targets,axis=0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.vocab[self.data[index]]),torch.tensor(self.vocab[self.target[index]])
