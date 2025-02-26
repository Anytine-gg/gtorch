import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection, VOCSegmentation
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from gtorch.utils.datasets.VOCDetection_ import VOCDetection_



def getVOCSeg(dir, train_transform, val_transform):
    pass


def detectionDemo():
    # 目标检测用例，加入bbox_params可以使得bbox也跟着变化
    transform = A.Compose(
        [
            #A.RandomCrop(width=256, height=256),
            A.LongestMaxSize(max_size=512),
            A.PadIfNeeded(
                min_height=512,
                min_width=512,
                border_mode=cv2.BORDER_CONSTANT,
                value=128,
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    dataset = VOCDetection_(
        './data',transform=transform
    )
    image, bboxes, labels = dataset[0]
    print(labels)
    # plot_bbox(image, bboxes, labels)


def SegmentationDemo():
    voc_dataset = VOCSegmentation(
        root="./data",
        year="2012",
        image_set="train",
        download=False,
        transform=transforms.ToTensor(),
        target_transform=None,
    )
    image, label = voc_dataset[0]
    np.set_printoptions(threshold=np.inf)

    print(np.array(label).shape)
    plot_img_seg(image,label)

    

if __name__ == "__main__":
    detectionDemo()
