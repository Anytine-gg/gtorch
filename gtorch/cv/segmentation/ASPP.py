import torch
from torch import nn
from gtorch.cv.segmentation.AtrousConv import AtrousConv
import torch.nn.functional as F
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18], dropout=0.1):
        super().__init__()
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.path2 = nn.Sequential(
            # kernel size 3, stride 1 dilation 12 padding: auto(None)
            AtrousConv(in_channels, out_channels, 3, 1, None, dilations[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.path3 = nn.Sequential(
            AtrousConv(in_channels, out_channels, 3, 1, None, dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.path4 = nn.Sequential(
            AtrousConv(in_channels, out_channels, 3, 1, None, dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.path5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 需要动态upsample
        )
        self.conv = nn.Conv2d(out_channels * 5, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.path1(x)
        out2 = self.path2(x)
        out3 = self.path3(x)
        out4 = self.path4(x)
        out5 = self.path5(x)
        size = out1.shape[2:]
        out5 = F.interpolate(out5, size=size, mode="bilinear", align_corners=True)

        output = torch.concat([out1, out2, out3, out4, out5], dim=1)
        output = self.bn(self.conv(output))
        output = self.dropout(self.relu(output))
        return output
