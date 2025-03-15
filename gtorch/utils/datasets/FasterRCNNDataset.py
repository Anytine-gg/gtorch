import torch
from torch.utils.data import Dataset
from gtorch.utils.datasets.VOCDetection_ import VOCDetection_
from gtorch.cv.detection.tools import AnchorGenerator, calc_IoU_tensor_v2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FasterRCNNDataset(Dataset):
    def __init__(self, VOCDataset: VOCDetection_):
        super.__init__()
        self.VOCDataset = VOCDataset
        self.feat_size = (50, 38)

    def __len__(self):
        return len(self.VOCDataset)

    def __getitem__(self, idx):
        image, bboxes, labels = self.VOCDataset[idx]
        anchors = AnchorGenerator(self.feat_size, scale=[8, 16, 32], ratio=[0.5, 1, 2])
        print(anchors)
        return


if __name__ == "__main__":
    transform = A.Compose(
        [
            A.Resize(600, 800),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),  # ImageNet 均值
                std=(0.229, 0.224, 0.225),  # ImageNet 标准差
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    voc_dataset = VOCDetection_(image_set="train", transform=transform)
    dataset = FasterRCNNDataset(voc_dataset)
    dataset[0]
