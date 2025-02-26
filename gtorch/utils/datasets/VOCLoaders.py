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


def plot_bbox(image, bboxes, labels):
    # 如果 image 是 torch.Tensor，则转换为 numpy 数组
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    # 如果 image shape 为 (C, H, W) 且通道数为 1 或 3，转换为 (H, W, C)
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))

    # 如果像素值在 [0,1]，放缩到 [0,255]
    if image.dtype != np.uint8 or image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # 假设传入的 image 是 RGB 图像，
    # OpenCV 的绘图函数默认按 BGR 理解，所以先把 RGB 转为 BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 复制一份用于绘制
    image_draw = image_bgr.copy()

    for bbox, label in zip(bboxes, labels):
        # 假设 bbox 格式为 [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = map(int, bbox)
        # 绘制边框，颜色为蓝色（BGR 下蓝色为 (255,0,0)），线宽2
        cv2.rectangle(
            image_draw, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=2
        )
        cv2.putText(
            image_draw,
            label,
            (xmin, max(ymin - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            thickness=2,
        )

    # 绘制结束后将图像从 BGR 转回 RGB 再用 pyplot 显示
    image_rgb = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("Image with Bboxes and Labels")
    plt.show()


def plot_img_seg(image, label):
    # 如果 image 是 torch.Tensor，则转为 numpy 且调整通道顺序（如果是 [C,H,W]）
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

    # 如果 label 是 torch.Tensor，则转为 numpy；如果 shape 为[1,H,W]则去除 channel 维度
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy()
        if label.ndim == 3 and label.shape[0] == 1:
            label = label.squeeze(0)

    # 绘制图像和掩码
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)

    axs[0].axis("off")

    axs[1].imshow(label, cmap="jet")

    axs[1].axis("off")

    plt.show()


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
    plot_bbox(image, bboxes, labels)


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
