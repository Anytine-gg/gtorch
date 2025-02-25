import torch
from torch import nn
import torchvision
import torchsummary
import torch.nn.functional as F


class SegNet(nn.Module):
    def __init__(self, out_channels, pretrained=False):
        super().__init__()
        vgg16 = torchvision.models.vgg16_bn(pretrained=pretrained)
        # 去掉maxpooling的vgg layer
        self.layer1 = vgg16.features[0:6]
        self.layer2 = vgg16.features[7:13]
        self.layer3 = vgg16.features[14:23]
        self.layer4 = vgg16.features[24:33]
        self.layer5 = vgg16.features[34:43]
        self.upsample5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.upsample4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.upsample3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.layer1(x)
        x, idx1 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2)
        x = self.layer2(x)
        x, idx2 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2)
        x = self.layer3(x)
        x, idx3 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2)
        x = self.layer4(x)
        x, idx4 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2)
        x = self.layer5(x)
        x, idx5 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2)
        x = F.max_unpool2d(x,idx5,2,2)
        x = self.upsample5(x)
        x = F.max_unpool2d(x,idx4,2,2)
        x = self.upsample4(x)
        x = F.max_unpool2d(x,idx3,2,2)
        x = self.upsample3(x)
        x = F.max_unpool2d(x,idx2,2,2)
        x = self.upsample2(x)
        x = F.max_unpool2d(x,idx1,2,2)
        x = self.upsample1(x)
        return x


if __name__ == "__main__":
    net = SegNet(4)
    torchsummary.summary(net, (3, 224, 224), -1, "cpu")
