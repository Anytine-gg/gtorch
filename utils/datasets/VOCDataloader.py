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


def plot(image, bboxes, labels):
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


class VOCDetection_(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.transform = transform
        self.voc_dataset = VOCDetection(
            root=data_dir,
            year="2007",
            image_set="trainval",
            download=False,
            transform=None,
        )

    def __getitem__(self, index):
        image, target = self.voc_dataset[index]
        image = np.array(image)
        objects = target["annotation"]["object"]
        labels = [obj["name"] for obj in objects]
        if isinstance(objects, dict):
            objects = [objects]

        def get_bbox(obj):
            box = obj["bndbox"]
            return [
                int(box["xmin"]),
                int(box["ymin"]),
                int(box["xmax"]),
                int(box["ymax"]),
            ]

        bboxes = [get_bbox(obj) for obj in objects]
        if self.transform:
            augmented = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            labels = augmented["labels"]

        return image, bboxes, labels

    def __len__(self):
        return len(self.voc_dataset)

def getVOCSegDataset(data_dir="./data", batch_size=4, num_workers=4):
    # 图像转换：转为 tensor 并归一化
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 分割标注转换：将标注转换为 numpy 数组，再转为 long 类型 tensor
    def target_transform(target):
        # target 本身为 PIL Image, 数值范围通常为 0~255，因此直接转换为 int 类别
        target = transforms.ToTensor()(target)
        # ToTensor 之后 target 形状为 (1, H, W)，将其转换为 (H, W) 的 long tensor
        target = torch.squeeze(target, 0)
        return (target * 255).long()

    # 构建训练集（VOCSegmentation 要求数据集目录符合VOC格式）
    train_dataset = VOCSegmentation(
        root=data_dir,
        year="2012",
        image_set="train",
        download=True,
        transform=img_transform,
        target_transform=target_transform
    )
    # 构建验证集
    val_dataset = VOCSegmentation(
        root=data_dir,
        year="2012",
        image_set="val",
        download=True,
        transform=img_transform,
        target_transform=target_transform
    )

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, train_loader, val_dataset, val_loader

if __name__ == "__main__":
    # train_dataset, train_loader, val_dataset, val_loader = get_voc_dataloaders()
    # print("训练集大小:", len(train_dataset))
    # print("验证集大小:", len(val_dataset))
    # images, targets = next(iter(train_loader))
    # # 选取第一张图片, images 的形状为 (batch_size, C, H, W)
    # img = images[0]
    # # 将 tensor 转换为 numpy 数组, 同时将通道维从 C,H,W 转换为 H,W,C
    # img_np = img.permute(1, 2, 0).numpy()
    # print(targets)
    # # 显示图片
    # plt.imshow(img_np)
    # plt.axis("off")
    # plt.title("VOC Image")
    # plt.show()
    transform = A.Compose(
        [
            A.RandomCrop(width=256, height=256),
            A.LongestMaxSize(max_size=512),
            A.PadIfNeeded(
                min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT, value=128
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    dataset = VOCDetection_("./data", transform)
    image, bboxes, labels = dataset[0]

    plot(image, bboxes, labels)
