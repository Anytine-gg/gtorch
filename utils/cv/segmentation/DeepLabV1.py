import torch
import numpy as np
from torchvision import transforms
from torch import nn
import torchsummary
import torchvision
import torchvision.models as models


class VGG16_LargeFOV(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        features = vgg.features
        features[4] = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        features[9] = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        features[16] = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        features[23] = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        features[30] = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
if __name__ == '__main__':
    vgg = models.vgg16()
    print(vgg.features)
