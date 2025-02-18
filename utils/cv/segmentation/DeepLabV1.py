import torch
import numpy as np
from torchvision import transforms
from torch import nn
import torchsummary
import torchvision
import torchvision.models as models
from AtrousConv import AtrousConv
import torch.nn.functional as F

class VGG16_LargeFOV(nn.Module):
    def __init__(self,out_channels,dropout=0.1):
        super().__init__()
        vgg = models.vgg16(pretrained=False)
        features = vgg.features
        features[4] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        features[9] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        features[16] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        features[23] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        features[30] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        features[24] = AtrousConv(512, 512, kernel_size=3, stride=1, dilation=2)
        features[26] = AtrousConv(512, 512, kernel_size=3, stride=1, dilation=2)
        features[28] = AtrousConv(512, 512, kernel_size=3, stride=1, dilation=2)
        
        self.sequential = nn.Sequential(features[:31])
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.atrousconv = AtrousConv(512, 1024, kernel_size=3, stride=1, dilation=12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.conv = nn.Conv2d(1024,1024,kernel_size=1)
        self.classify = nn.Conv2d(1024,out_channels,kernel_size=1)

    def forward(self,x):
        x = self.sequential(x)
        x = self.avgpool(x)
        x = F.relu(self.atrousconv(x))
        x = self.dropout1(x)
        x = F.relu(self.conv(x))
        x = self.dropout2(x)
        x = self.classify(x)
        
        
if __name__ == "__main__":
    model = VGG16_LargeFOV(16)
    img = torch.rand(3,224,224)
    torchsummary.summary(model,(3,224,224))
    
