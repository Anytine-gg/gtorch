import torch
from torch.utils.data import Dataset
from gtorch.utils.datasets.VOCDetection_ import VOCDetection_
from gtorch.cv.detection.tools import AnchorGenerator, calc_IoU_tensor_v2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F


class FasterRCNNDataset(Dataset):
    def __init__(self, VOCDataset: VOCDetection_):
        super().__init__()
        self.VOCDataset = VOCDataset
        self.feat_size = (50, 38)

    def __len__(self):
        return len(self.VOCDataset)

    def __getitem__(self, idx):
        image, bboxes, labels = self.VOCDataset[idx]
        anchors = AnchorGenerator(self.feat_size, scale=[8, 16, 32], ratio=[0.5, 1, 2])
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        iou = calc_IoU_tensor_v2(bboxes, anchors)
        IoUMax, argMax = torch.max(iou, dim=1)
        
        label_cls = torch.zeros_like(IoUMax,dtype=torch.long)
        label_cls[IoUMax >= 0.7] = 1
        
        # 将anchors 转为cx cy w h 的形式
        width = anchors[:, 2] - anchors[:, 0]
        height = anchors[:, 3] - anchors[:, 1]
        anchors[:, 0] = anchors[:, 0] + width / 2
        anchors[:, 1] = anchors[:, 1] + height / 2
        anchors[:, 2] = width
        anchors[:, 3] = height
        # 将bboxes 转为cx cy w h 的形式
        width = bboxes[:, 2] - bboxes[:, 0]
        height = bboxes[:, 3] - bboxes[:, 1]
        bboxes[:, 0] = bboxes[:, 0] + width / 2
        bboxes[:, 1] = bboxes[:, 1] + height / 2
        bboxes[:, 2] = width
        bboxes[:, 3] = height

        # 计算回归目标
        label_reg = torch.zeros_like(anchors)
        label_reg[label_cls == 1][:,:2] = bboxes[argMax[label_cls == 1], :2] - anchors[label_cls == 1][:,:2]
        label_reg[label_cls == 1][:,:2] /= anchors[label_cls == 1][:, 2:]
        label_reg[label_cls == 1][:,2:] = torch.log(bboxes[argMax[label_cls == 1], 2:] / anchors[label_cls == 1][:, 2:])
        
        return image, label_cls, label_reg


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
