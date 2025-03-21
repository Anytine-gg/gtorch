import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
import cv2
def plot_bbox(image, bboxes, labels, format='xyxy'):
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
    if labels is None:
        labels = [""] * len(bboxes)
    for bbox, label in zip(bboxes, labels):
        if format == 'xyxy':
            # 假设 bbox 格式为 [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = map(int, bbox)
        else:
            # 假设 bbox 格式为 [cx, cy, w, h]
            cx, cy, w, h = map(int, bbox)
            xmin = int(cx - w / 2)
            ymin = int(cy - h / 2)
            xmax = int(cx + w / 2)
            ymax = int(cy + h / 2)
        # 绘制边框，颜色为蓝色（BGR 下蓝色为 (255,0,0)），线宽1
        cv2.rectangle(
            image_draw, (xmin, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1
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


def plot_seg(image, label):
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

def plot_img(image):
    """显示图像，支持numpy array和torch tensor格式
    
    Args:
        image: numpy array或torch tensor格式的图像
              支持(H,W), (H,W,C)或(C,H,W)格式
    """
    
    # 转换为numpy array
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # 处理通道顺序
    if len(image.shape) == 3:
        if image.shape[0] in [1, 3]:  # (C,H,W)格式
            image = np.transpose(image, (1, 2, 0))
        if image.shape[2] == 1:  # (H,W,1)格式
            image = image.squeeze()
            
    # 处理值范围
    if image.max() <= 1.0:
        image = image * 255
    image = image.astype(np.uint8)
    
    # 显示图像
    plt.figure(figsize=(8, 8))
    if len(image.shape) == 2 or image.shape[-1] == 1:  # 灰度图
        plt.imshow(image, cmap='gray')
    else:  # RGB图
        plt.imshow(image)
    plt.axis('off')
    plt.show()