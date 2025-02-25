from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import random_split, DataLoader
from utils.datasets.VOCLoaders import get_voc_dataloaders


train_dataset, train_loader, val_dataset, val_loader = get_voc_dataloaders(
    batch_size=1, num_workers=4
)


# 查看数据集中的一张图像和标注
image, target = next(iter(train_loader))
image = image[0]
target = target[0]
print(image.shape)  # 图像形状 (C, H, W)
print(target)  # 标注信息


fig, ax = plt.subplots(1)
ax.imshow(image.permute(1, 2, 0).numpy())


objects = target["annotation"]["object"]
if isinstance(objects, dict):
    objects = [objects]
# 遍历每个目标
for obj in objects:
    # 边界框信息存储在 obj["bndbox"]
    bndbox = obj["bndbox"]
    xmin = int(bndbox["xmin"])
    ymin = int(bndbox["ymin"])
    xmax = int(bndbox["xmax"])
    ymax = int(bndbox["ymax"])
    width = xmax - xmin
    height = ymax - ymin
    # 创建矩形框
    rect = patches.Rectangle(
        (xmin, ymin), width, height, linewidth=2, edgecolor="r", facecolor="none"
    )
    ax.add_patch(rect)
    # 可选：显示类别名称
    label = obj["name"]
    ax.text(xmin, ymin, label, color="yellow", fontsize=12, backgroundcolor="red")


plt.axis("off")
plt.show()
