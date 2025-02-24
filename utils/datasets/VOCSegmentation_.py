# 使用Albumentations的VOC分割数据集
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection, VOCSegmentation


class VOCSegmentation(Dataset):
    def __init__(
        self,
        dir,
        year="2012",
        image_set="train",
        download=False,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.voc_dataset = VOCSegmentation(
            root=dir,
            year=year,
            image_set=image_set,
            download=download,
            transform=None,
            target_transform=None,
        )

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, index):
        pass
