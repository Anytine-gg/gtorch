import os
import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader

def get_voc_dataloaders(data_dir="./data", batch_size=8, num_workers=4, split_ratio=0.8):
    """
    加载2007 VOC数据集，并划分为训练集和验证集。
    
    参数:
        data_dir: VOC 数据集存放的目录
        batch_size: 每个 batch 的样本数
        num_workers: DataLoader 使用的工作线程数
        split_ratio: 训练集占整个训练验证集的比例
    返回:
        train_dataset, train_loader, val_dataset, val_loader
    """
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])
    
    # 加载 VOC2007 数据集，采用 trainval 模式
    voc_dataset = VOCDetection(root=data_dir, year="2007", image_set="trainval", download=False, transform=transform)
    
    dataset_size = len(voc_dataset)
    train_size = int(split_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    # 划分训练集和验证集
    train_dataset, val_dataset = torch.utils.data.random_split(voc_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dataset, train_loader, val_dataset, val_loader

if __name__ == "__main__":
    train_dataset, train_loader, val_dataset, val_loader = get_voc_dataloaders()
    print("训练集大小:", len(train_dataset))
    print("验证集大小:", len(val_dataset))