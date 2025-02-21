import torch
from torch import nn
import torchvision
import torchsummary
from AtrousConv import AtrousConv


class Bottleneck1(nn.Module):
    def __init__(self, in_channels, out_channels,stride,padding=None ,dilation=1):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = AtrousConv(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            padding=padding
        )

    def forward():
        pass


class Bottleneck2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward():
        pass


class DeepLabV2(nn.Module):
    def __init__(self, out_channels, pretrained=False):
        super().__init__()
        resnet101 = torchvision.models.resnet101(pretrained=pretrained)
        self.layer1 = resnet101.layer1
        self.layer2 = resnet101.layer2
        self.stage0 = nn.Sequential(*list(resnet101.children())[:4])

    def forward(self, x):
        x = self.stage0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


if __name__ == "__main__":
    resnet = torchvision.models.resnet101()
    model = DeepLabV2(3)
    torchsummary.summary(model, (3, 224, 224))
