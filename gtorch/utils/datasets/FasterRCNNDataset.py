import torch
from torch.utils.data import Dataset
from gtorch.utils.datasets.VOCDetection_ import VOCDetection_
from gtorch.cv.detection.tools import AnchorGenerator, calc_IoU_tensor_v2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from gtorch.utils.misc.plot import plot_bbox

class FasterRCNNDataset(Dataset):
    def __init__(self, VOCDataset: VOCDetection_):
        super().__init__()
        self.VOCDataset = VOCDataset
        self.feat_size = (38,50)
        self.anchors = AnchorGenerator(self.feat_size, scale=[8, 16, 32], ratio=[0.5, 1, 2],crop=True)
    def __len__(self):
        return len(self.VOCDataset)

    def __getitem__(self, idx):
        image, bboxes, labels = self.VOCDataset[idx]
        anchors = self.anchors.clone()
    
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
        # 确保每个bbox都分配一个anchor
        temp = torch.argmax(iou,dim=0)
        label_cls[temp] = 1
        
        for i in range(len(bboxes)):
            anchor_idx = temp[i]  # 第i个bbox对应的最佳anchor索引
            # 计算第i个bbox与其最佳anchor的回归目标
            label_reg[anchor_idx, :2] = (bboxes[i, :2] - anchors[anchor_idx, :2]) / anchors[anchor_idx, 2:]
            label_reg[anchor_idx, 2:] = torch.log(bboxes[i, 2:] / anchors[anchor_idx, 2:])
        
        return image, F.one_hot(label_cls,2), label_reg, anchors[label_cls == 1]


if __name__ == "__main__":
    transform = A.Compose(
        [
            A.Resize(600, 800),
            # A.Normalize(
            #     mean=(0.485, 0.456, 0.406),  # ImageNet 均值
            #     std=(0.229, 0.224, 0.225),  # ImageNet 标准差
            # ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    voc_dataset = VOCDetection_(image_set="train", transform=transform)
    dataset = FasterRCNNDataset(voc_dataset)
    dataset[0]
