import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from .VOCDetection_ import VOCDetection_


class YOLOv3_VOCDataset(Dataset):
    def __init__(
        self,
        dir="./data",
        year="2012",
        image_set="train",
        download=False,
        transform=None,
    ):
        super().__init__()
        self.voc_dataset = VOCDetection_(dir, year, image_set, download, transform)

    def __getitem__(self, index):
        image, bboxes, labels = self.voc_dataset[index]
        img_shape = image.shape
        
    def __len__(self):
        pass
