import torch
from torch import nn
from utils.nlp.Vocab import Vocab,load_books


books = load_books('/root/projs/py/demo/books')
vocab = Vocab(books=books)
print(vocab[list('瀚')])
