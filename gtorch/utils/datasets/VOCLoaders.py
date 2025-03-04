import os
from turtle import down
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
from gtorch.utils.datasets.YOLOv3Dataset import YOLOv3_Dataset



def getVOC2012SegLoaders(dir, train_transform, val_transform):
    pass
def getVOC2012DetLoaders(train_transform,val_transform,batch_size,dir = './data',download=False,val_shuffle = False):
    """
        获取VOC2012目标检测数据集Dateset和Dataloader \n
        Returns: train_dataset,train_loader,val_dataset,val_loader
    """
    train_dataset = VOCDetection_(data_dir=dir,image_set='train',transform=train_transform,download=download)
    val_dataset = VOCDetection_(data_dir=dir,image_set='val',transform=val_transform,download=download)
    train_dataset = YOLOv3_Dataset(20,dataset=train_dataset)
    val_dataset = YOLOv3_Dataset(20,dataset=val_dataset)
    train_loader=  DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    val_loader=  DataLoader(val_dataset,batch_size=batch_size,shuffle=val_shuffle,num_workers=4)
    return train_dataset,train_loader,val_dataset,val_loader
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
