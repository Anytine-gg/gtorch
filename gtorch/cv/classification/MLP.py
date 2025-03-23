import torch
from torch import nn
import torchsummary
import torchvision.models
import torchvision.transforms as transforms
import gtorch.utils.misc.plot as gplt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # 添加此行
device = 'cuda'
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 转换为tensor并归一化到[0,1]
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST数据集的均值和标准差
    ]
)
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024,2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048,2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048,256),
    nn.ReLU(inplace=True),
    nn.Linear(256,10),
)
net.to(device)
train_dataset = torchvision.datasets.MNIST(
    "./data", train=True, transform=transform
)
val_dataset = torchvision.datasets.MNIST(
    "./data", train=False, transform=transform
)
train_loader = DataLoader(train_dataset,256,True)
