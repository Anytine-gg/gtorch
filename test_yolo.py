from gtorch.utils.datasets.YOLOv3Dataset import YOLOv3_Dataset
from gtorch.utils.datasets.VOCDetection_ import VOCDetection_
from gtorch.utils.misc.plot import plot_bbox
if __name__ == '__main__':
    dataset = VOCDetection_()
    dataset = YOLOv3_Dataset(dataset=dataset)
    print(dataset[0])