# 使用Albumentations的VOC分割数据集
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection, VOCSegmentation
import albumentations as A
import numpy as np

class VOCSegmentation_(Dataset):
    def __init__(
        self,
        root,
        year="2012",
        image_set="train",
        download=False,
        transform=None
    ):
        super().__init__()
        self.transform = transform
        self.voc_dataset = VOCSegmentation(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=None,
            target_transform=None,
        )

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, index):
        img, target = self.voc_dataset[index]
        img = np.array(img)
        target = np.array(target)
        if self.transform:
            augmented = self.transform(image=img, mask=target)
            img = augmented['image'].float()
            target = augmented['mask'].long()
        return img, target

