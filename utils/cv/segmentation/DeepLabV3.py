from numpy import pad
import torch
from torch import nn
import torchsummary
import torchvision
from AtrousConv import AtrousConv
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations = [6,12,18],dropout=0.1):
        super().__init__()
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.path2 = nn.Sequential(
            # kernel size 3, stride 1 dilation 12 padding: auto(None)
            AtrousConv(in_channels, out_channels, 3, 1, None, dilations[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.path3 = nn.Sequential(
            AtrousConv(in_channels, out_channels, 3, 1, None, dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.path4 = nn.Sequential(
            AtrousConv(in_channels, out_channels, 3, 1, None, dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.path5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(size=(60, 60), mode="bilinear", align_corners=True),
        )
        self.conv = nn.Conv2d(out_channels * 5, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out4 = self.path4(x)
        out5 = self.path5(x)
        output = torch.concat([out1, out2, out3, out4, out5], dim=1)
        output = self.bn(self.conv(output))
        output = self.dropout(self.relu(output))
        return output


class DeepLabV3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        backbone = torchvision.models.resnet50(
            replace_stride_with_dilation=[False, True, True]
        )
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.aspp = ASPP(2048, 256,[12,24,36])
        self.conv = nn.Conv2d(256, 256, 3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.proj = nn.Conv2d(256, out_channels, 1)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.proj(x)
        x = self.upsample(x)
        return x


if __name__ == "__main__":
    model = DeepLabV3(3, 3)
    torchsummary.summary(model, (3, 480, 480), -1, "cpu")
