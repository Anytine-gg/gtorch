import albumentations as A
import cv2
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from gtorch.cv.detection.tools import calc_IoU_tensor, yolo3_loss
from gtorch.utils.datasets.VOCDetection_ import VOCDetection_, voc2012_labels
from gtorch.cv.detection import get_default_anchor_size


class YOLOv3_Dataset(Dataset):
    def __init__(
        self,
        num_of_classes,
        dataset=None,
        anchor_size=get_default_anchor_size("yolov3"),
    ):
        super().__init__()
        self.dataset = dataset
        self.anchors = anchor_size
        self.num_of_classes = num_of_classes
        self.device = "cpu"  # cuda无法用于多线程

    def __getitem__(self, index):
        image, bboxes, labels = self.dataset[index]
        device = self.device
        image = image.to(device)
        feat_map1 = torch.zeros(
            3 * (5 + self.num_of_classes), 52, 52, device=device
        ).float()
        feat_map2 = torch.zeros(
            3 * (5 + self.num_of_classes), 26, 26, device=device
        ).float()
        feat_map3 = torch.zeros(
            3 * (5 + self.num_of_classes), 13, 13, device=device
        ).float()

        feat_map = [feat_map1, feat_map2, feat_map3]

        if len(labels) == 0:
            return image, feat_map1, feat_map2, feat_map3

        # YOLOv3 输入416*416, bbox: [[xmin,ymin,xmax,ymax],...,]
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected image to be torch.Tensor but got {type(image)}")
        if image.shape[-2:] != (416, 416):
            raise ValueError(
                f"Expected image height and width to be (416, 416) but got {image.shape[-2:]}"
            )
        # 将string的label转为long
        labels = torch.tensor(
            [voc2012_labels.get(item, 0) for item in labels], device=device
        ).long()
        stride1 = 416 // 52  # 8
        stride2 = 416 // 26  # 16
        stride3 = 416 // 13  # 32
        grid_strides = torch.tensor([stride1, stride2, stride3], device=device).int()
        # 以cx,cy,width,height记录框
        num_of_bboxes = len(bboxes)
        bboxes = torch.tensor(bboxes, device=device).float()
        width = bboxes[:, 2] - bboxes[:, 0]
        height = bboxes[:, 3] - bboxes[:, 1]
        bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
        bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
        bboxes[:, 2] = width
        bboxes[:, 3] = height
        bboxes_origin = torch.clone(bboxes)
        # 生成锚框的位置矩阵，shape为(9*num_of_bboxes,4),
        anchors = torch.tensor(self.anchors, device=device).float().reshape(9, 2)
        anchors = anchors.repeat(num_of_bboxes, 1)
        bboxes = bboxes.repeat_interleave(9, dim=0)
        anchors = torch.cat([bboxes[:, 0:2], anchors], dim=1)

        strides = torch.clone(grid_strides)
        strides = strides.repeat_interleave(3, dim=0).repeat(2, num_of_bboxes).T
        # 确定中心点所在的格子
        centers = anchors[:, 0:2] // strides
        
        anchors[:, 0:2] = centers * strides + strides / 2

        iou = calc_IoU_tensor(bboxes, anchors).reshape(-1, 9)  # 一行为一个bbox的9个IoU
        
        # 每一行对应一个负责预测bboxes的九个anchor的cx,cy,w,h
        anchors = anchors.reshape(-1, 9, 4)
        # 每一行对应一个负责预测bboxes的九个anchor的cx cy
        centers = centers.reshape(-1, 9, 2)

        
        max_iou, argmax = torch.max(iou, dim=1)

        # 生成一个bboxes_idx_iou tensor,每一行是一个bbox的x,y,w,h,idx,iou,id. 其中id是bboxes原来的排位
        # idx是负责预测的anchor的idx,iou是最大的iou. 在iou处排序,确保最大的在前,避免一个anchor预测多个bbox

        bboxes_idx_iou = torch.cat(
            [
                bboxes_origin,
                argmax.unsqueeze(1),
                max_iou.unsqueeze(1),
                labels.unsqueeze(1),
            ],
            dim=1,
        )
        
        #* anchors centers也要经过排序！！！
        sorted_indices = torch.argsort(bboxes_idx_iou[:, 5], dim=0, descending=True)
        bboxes_idx_iou = bboxes_idx_iou[sorted_indices]
        anchors = anchors[sorted_indices]
        centers = centers[sorted_indices]
        for i in range(num_of_bboxes):
            anchor_idx = bboxes_idx_iou[i, 4].long()
            gt_pos = bboxes_idx_iou[i, :2]
            gt_size = bboxes_idx_iou[i, 2:4]
            
            anchor = anchors[i, anchor_idx]

            anchor_pos = anchor[0:2]  # 负责预测anchor的中心
            anchor_size = anchor[2:4]  # 负责预测anchor的size
            
            stride = grid_strides[anchor_idx // 3].float()
            
            cxcy = centers[i,anchor_idx]
            cx, cy = cxcy.long()
            
            step = (anchor_idx % 3) * (5 + self.num_of_classes)
            if feat_map[anchor_idx // 3][step + 4, cx, cy] != 0:
                # 该anchor已被分配, iou从大到小排.
                continue

            txty = gt_pos / stride - cxcy  # sigmoid后,再与txty做loss
            twth = torch.log(gt_size / anchor_size)  # 不用再乘stride
            
            # txty = bboxes_idx_iou[i,:2]
            # twth = bboxes_idx_iou[i,2:4]
            # print(txty,twth)
            conf = 1.0

            label = bboxes_idx_iou[i, 6].long()
            feat_map[anchor_idx // 3][step : step + 2, cx, cy] = txty
            feat_map[anchor_idx // 3][step + 2 : step + 4, cx, cy] = twth
            feat_map[anchor_idx // 3][step + 4, cx, cy] = conf
            feat_map[anchor_idx // 3][step + 5 + label, cx, cy] = 1
        # feat_map1,2,3的尺寸分别是52,26,13(检测小，中，大物体)
        return image, feat_map1, feat_map2, feat_map3

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
    dataset = YOLOv3_Dataset(20, dataset)
    pred_feat = torch.randn(1, 75, 52, 52)
    dataset[6]
    # print(dataset[1][1].unsqueeze(0).shape)
    # print(yolo3_loss(pred_feat,dataset[1][1].unsqueeze(0)))
