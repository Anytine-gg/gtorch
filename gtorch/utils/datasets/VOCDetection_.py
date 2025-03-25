# 使用Albumentations的Detection Dateset

from torchvision.datasets import VOCDetection, VOCSegmentation
from torch.utils.data import Dataset, DataLoader
import numpy as np

voc2012_labels = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19
}


# 2007不分train和val，用split分
class VOCDetection_(Dataset):
    def __init__(
        self,
        data_dir='./data',
        year="2012",
        image_set="train",
        download=False,
        transform=None,
    ):
        super().__init__()
        self.transform = transform
        self.voc_dataset = VOCDetection(
            root=data_dir,
            year=year,
            image_set=image_set,
            download=download,
            transform=None,
        )

    def __getitem__(self, index):  #xyxy格式
        image, target = self.voc_dataset[index]
        image = np.array(image)
        objects = target["annotation"]["object"]
        labels = [obj["name"] for obj in objects]
        if isinstance(objects, dict):
            objects = [objects]

        def get_bbox(obj):
            box = obj["bndbox"]
            return [
                int(box["xmin"]),
                int(box["ymin"]),
                int(box["xmax"]),
                int(box["ymax"]),
            ]

        bboxes = [get_bbox(obj) for obj in objects]
        if self.transform:
            augmented = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            labels = augmented["labels"]
        
        return image, bboxes, labels

    def __len__(self):
        return len(self.voc_dataset)
