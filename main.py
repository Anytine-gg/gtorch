from ctypes import util
from pytz import utc
import torch
from torch import mode, nn
from utils.nlp.Vocab import Vocab,load_books
from utils.datasets.LangDataset import LangDataset
from torch.utils.data import DataLoader,Dataset
from utils.models.LTSM import LSTM_demo
from utils.mytorch import try_gpu
import torch.nn.functional as F
seq_len = 30
batch_size = 64
train_dataset = LangDataset(books_path='/root/projs/py/demo/books',seq_len=seq_len,min_freq=3000)
vocab = train_dataset.vocab
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
print(len(vocab))
def train():
    
    num_epoch = 50
    ce_loss = nn.CrossEntropyLoss()
    input_sz = output_sz = vocab_sz = len(vocab)
    model = LSTM_demo(input_sz,256,1,output_sz)
    model.to(try_gpu())
    optimizer = torch.optim.Adam(lr=0.001,params=model.parameters())
    for epoch in range(num_epoch):
        model.train()
        for (feature,label) in train_loader:
            feature = feature.T
            feature = F.one_hot(feature,num_classes=vocab_sz).float()
            label = label.T
            label = label.reshape(-1)
            feature = feature.to(try_gpu())
            label = label.to(try_gpu())
            predict,(h,c) = model(feature)
            loss = ce_loss(predict,label)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss.item())
train()