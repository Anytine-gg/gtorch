from gtorch.utils.datasets.YOLOv3Dataset import YOLOv3_VOCDataset
from gtorch.misc.plot import plot_bbox
import albumentations as A

if __name__ == '__main__':
    dataset = YOLOv3_VOCDataset()
    plot_bbox(*dataset[2])