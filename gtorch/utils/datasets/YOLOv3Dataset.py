from ast import arg
from networkx import center
from numpy import arange
from sympy import im
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from zmq import device
from gtorch.cv.detection.tools import calc_IoU_tensor
from gtorch.utils.datasets.VOCDetection_ import VOCDetection_
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from gtorch.utils.misc.plot import plot_bbox


class YOLOv3_Dataset(Dataset):
    def __init__(
        self,
        dataset=None,
        anchors=[
            [(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)],
        ],
    ):
        super().__init__()
        self.dataset = dataset
        self.anchors = anchors

    def __getitem__(self, index):
        image, bboxes, labels = self.dataset[index]
        # YOLOv3 输入416*416, bbox: [[xmin,ymin,xmax,ymax],...,]
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image to be torch.Tensor but got {type(image)}")
        if image.shape[-2:] != (416, 416):
            raise ValueError(
                f"Expected image height and width to be (416, 416) but got {image.shape[-2:]}"
            )
        stride1 = 416 // 52  # 8
        stride2 = 416 // 26  # 16
        stride3 = 416 // 13  # 32
        # 以cx,cy,width,height记录框
        num_of_bboxes = len(bboxes)
        bboxes = torch.tensor(bboxes).float()
        width = bboxes[:, 2] - bboxes[:, 0]
        height = bboxes[:, 3] - bboxes[:, 1]
        bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
        bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
        bboxes[:, 2] = width
        bboxes[:, 3] = height
        # 生成锚框的位置矩阵，shape为(9*num_of_bboxes,4),
        anchors = torch.tensor(self.anchors).float().reshape(9, 2)
        anchors = torch.cat([torch.zeros(9, 2), anchors], dim=1)
        anchors = anchors.repeat(num_of_bboxes, 1)
        bboxes = bboxes.repeat_interleave(9, dim=0)
        anchors[:, 0:2] = bboxes[:, 0:2]
        strides = torch.tensor([stride1, stride2, stride3]).int()
        strides = strides.repeat_interleave(3, dim=0).repeat(num_of_bboxes, 2).T
        # 确定中心点
        centers = anchors[:, 0:2] // strides
        anchors[:, 0:2] = centers * strides + strides / 2

        iou = calc_IoU_tensor(bboxes, anchors).reshape(-1, 9)
        argmax = torch.argmax(iou, dim=1)
        # 获取对应iou最大的中心点
        
        # TODO
        centers = centers.reshape(-1,9,2)[torch.arange(num_of_bboxes),argmax]
        return centers

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    from gtorch.utils.datasets.VOCDetection_ import VOCDetection_

    transform = transform = A.Compose(
        [
            A.LongestMaxSize(max_size=416),
            A.PadIfNeeded(
                min_height=416,
                min_width=416,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(128, 128, 128),
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

    dataset = VOCDetection_(transform=transform)
    dataset = YOLOv3_Dataset(dataset)
    print(dataset[0])
