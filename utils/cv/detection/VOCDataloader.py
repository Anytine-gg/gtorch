from cProfile import label
import os
from re import L
from scipy import datasets
import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class MyVOCDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.transform = transform
        self.voc_dataset = VOCDetection(
            root=data_dir,
            year="2007",
            image_set="trainval",
            download=False,
            transform=None,
        )

    def __getitem__(self, index):
        image, target = self.voc_dataset[index]
        objects = target["annotation"]["object"]
        labels = [obj['name'] for obj in objects]
        if isinstance(objects, dict):
            objects = [objects]
        
        def get_bbox(obj):
            box = obj['bndbox']
            return [int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])]
        bboxes = [get_bbox(obj) for obj in objects]
        if self.transform:
            augmented = self.transform(image=image,bboxes=bboxes)
            image = augmented['image']
            bboxes = augmented['bboxes']
            
        return image,bboxes,labels
    

    def __len__(self):
        return len(self.voc_dataset)


def get_voc_dataloaders(
    data_dir="./data", batch_size=1, num_workers=4, split_ratio=0.8
):
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
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
        ]
    )

    # 加载 VOC2007 数据集，采用 trainval 模式
    voc_dataset = VOCDetection(
        root=data_dir,
        year="2007",
        image_set="trainval",
        download=False,
        transform=transform,
    )

    dataset_size = len(voc_dataset)
    train_size = int(split_ratio * dataset_size)
    val_size = dataset_size - train_size

    # 划分训练集和验证集
    train_dataset, val_dataset = torch.utils.data.random_split(
        voc_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_dataset, train_loader, val_dataset, val_loader


if __name__ == "__main__":
    # train_dataset, train_loader, val_dataset, val_loader = get_voc_dataloaders()
    # print("训练集大小:", len(train_dataset))
    # print("验证集大小:", len(val_dataset))
    # images, targets = next(iter(train_loader))
    # # 选取第一张图片, images 的形状为 (batch_size, C, H, W)
    # img = images[0]
    # # 将 tensor 转换为 numpy 数组, 同时将通道维从 C,H,W 转换为 H,W,C
    # img_np = img.permute(1, 2, 0).numpy()
    # print(targets)
    # # 显示图片
    # plt.imshow(img_np)
    # plt.axis("off")
    # plt.title("VOC Image")
    # plt.show()
    dataset = MyVOCDataset("./data", None)
    print(dataset[0])
