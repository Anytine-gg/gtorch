from gtorch.utils.datasets.YOLOv3Dataset import YOLOv3_VOCDataset
from gtorch.utils.misc.plot import plot_bbox
if __name__ == '__main__':
    dataset = YOLOv3_VOCDataset(download=False)
    plot_bbox(*dataset[0])