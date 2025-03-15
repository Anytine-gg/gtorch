from gtorch.cv.detection.RPN import RPNHead
import gtorch.cv.detection.tools as gtools
from gtorch.utils.datasets.VOCDetection_ import VOCDetection_
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from gtorch.utils.misc.plot import plot_bbox
import cv2
from gtorch.utils.datasets.FasterRCNNDataset import FasterRCNNDataset
transform = A.Compose(
        [
            A.Resize(600,800),
            # A.Normalize(
            #     mean=(0.485, 0.456, 0.406),  # ImageNet 均值
            #     std=(0.229, 0.224, 0.225),  # ImageNet 标准差
            # ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
if __name__ == '__main__':
    batch_size = 16
    train_dataset = VOCDetection_(image_set='train',transform=transform)   
    train_dataset = FasterRCNNDataset(train_dataset)
    image,label_cls,label_reg,anchors = train_dataset[0]
    plot_bbox(image,anchors,labels=None,format='xywh')