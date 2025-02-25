import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot(image,bboxes,labels):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).numpy())


    objects = bboxes
    
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
