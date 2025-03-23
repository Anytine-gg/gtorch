import torch
from torchvision.datasets import Caltech101
from torchvision import transforms
import numpy as np
from gtorch.utils.misc.plot import plot_img
import albumentations as A
from albumentations.pytorch import ToTensorV2
from gtorch.utils.datasets.VOCSegmentation_ import VOCSegmentation_
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch import nn
from tqdm import tqdm

model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model.to('cuda')
transform = transforms.Compose(
    [
        transforms.Resize([224,224]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 归一化
    ]
)

trainset = Caltech101(root="./data",  download=False, transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

classifier = nn.Sequential(
    nn.Linear(384,101)
)

classifier.to('cuda')
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(classifier.parameters(),lr=0.001)

def validate(model, classifier, val_loader, criterion):
    model.eval()
    classifier.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for image, label in val_loader:
            image = image.to('cuda')
            label = label.to('cuda')
            features = model(image)
            outputs = classifier(features)
            
            loss = criterion(outputs, label)
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total

# 训练循环
for epoch in range(100):
    model.eval()  # DINO模型设置为评估模式
    classifier.train()  # 分类器设置为训练模式
    
    # 训练阶段
    train_loss = 0
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for image, label in train_pbar:
        image = image.to('cuda')
        label = label.to('cuda')
        
        with torch.no_grad():
            features = model(image)
        pred = classifier(features)
        l = loss(pred, label)
        
        optim.zero_grad()
        l.backward()
        optim.step()
        
        train_loss += l.item()
        train_pbar.set_postfix({'train_loss': f'{l.item():.4f}'})
    
    # 验证阶段
    val_loss, val_acc = validate(model, classifier, val_loader, loss)
    
    # 打印训练和验证结果
    print(f'Epoch: {epoch}')
    print(f'Train Loss: {train_loss/len(train_loader):.4f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print('-' * 50)


    