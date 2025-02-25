from numpy import pad
from torch import nn
import torchvision
import torchsummary
from gtorch.cv.segmentation.AtrousConv import AtrousConv

import torch.nn.functional as F



class AtrousBottleneck(nn.Module):
    def __init__(
        self, in_channels, out_channels, padding=None, dilation=1, projection=False
    ):
        super().__init__()
        self.proj = projection
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = AtrousConv(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if self.proj:
            self.conv4 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.proj:
            x = self.bn4(self.conv4(x))
        return self.relu(x + y)


class DeepLabV2(nn.Module):
    def __init__(self, out_channels, pretrained=False):
        super().__init__()
        resnet101 = torchvision.models.resnet101(pretrained=pretrained)
        self.layer1 = resnet101.layer1
        self.layer2 = resnet101.layer2
        self.stage0 = nn.Sequential(*list(resnet101.children())[:4])
        self.layer3 = nn.Sequential(
            AtrousBottleneck(512, 1024, padding=2, dilation=2, projection=True),
            *[
                AtrousBottleneck(1024, 1024, padding=2, dilation=2, projection=False)
                for _ in range(22)
            ]
        )
        self.layer4 = nn.Sequential(
            AtrousBottleneck(1024, 2048, padding=4, dilation=4, projection=True),
            AtrousBottleneck(2048, 2048, padding=4, dilation=4, projection=False),
            AtrousBottleneck(2048, 2048, padding=4, dilation=4, projection=False),
        )
        self.aspp1 = AtrousConv(2048, out_channels, 3, 1, padding=6, dilation=6)
        self.aspp2 = AtrousConv(2048, out_channels, 3, 1, padding=12, dilation=12)
        self.aspp3 = AtrousConv(2048, out_channels, 3, 1, padding=18, dilation=18)
        self.aspp4 = AtrousConv(2048, out_channels, 3, 1, padding=24, dilation=24)

    def forward(self, x):
        x = self.stage0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp1(x) + self.aspp2(x) + self.aspp3(x) + self.aspp4(x)
        x = F.interpolate(
            x, scale_factor=8, mode="bilinear", align_corners=False
        )  # 上采样到原图
        return x

if __name__ == "__main__":
    resnet = torchvision.models.resnet101()
    model = DeepLabV2(3)
    torchsummary.summary(model, (3, 224, 224))
