import torch
from torch import nn


class LSTM_demo(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers,output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers)
        self.fc = nn.Linear(hidden_size,output_size)
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        
        # 前向传播 LSTM
        out, state = self.lstm(x, (h0, c0))
        
        out = self.fc(out)
        
        return out,state

if __name__ == '__main__':
    model = LSTM_demo(30,256,2,30)
    data = torch.randn(35,100,30)

    print(model(data)[0].shape)