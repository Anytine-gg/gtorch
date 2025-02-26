from cv2 import repeat
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from gtorch.utils.datasets.VOCDetection_ import VOCDetection_


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

        # # YOLOv3 输入416*416, bbox: [[xmin,ymin,xmax,ymax],...,]
        # if not isinstance(image, torch.Tensor):
        #     raise TypeError(f"Expected image to be torch.Tensor but got {type(image)}")
        # if image.shape[-2:] != (416, 416):
        #     raise ValueError(
        #         f"Expected image height and width to be (416, 416) but got {image.shape[-2:]}"
        #     )
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
        anchors = anchors.repeat(num_of_bboxes,1)
        
        
        cx = bboxes[:,0]
        cy = bboxes[:,1]

        
        return anchors

    def __len__(self):
        return len(self.dataset)

    def calc_iou(bboxes, anchors):
        pass


if __name__ == "__main__":
    from gtorch.utils.datasets.VOCDetection_ import VOCDetection_

    dataset = VOCDetection_()
    dataset = YOLOv3_Dataset(dataset)
    print(dataset[0])
