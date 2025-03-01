import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, proj=False):
        super().__init__()
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels,out_channels,1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if proj:
            self.proj_conv = nn.Conv2d(in_channels,out_channels,1,stride=stride,bias=False)
            self.proj_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.proj = proj
    def forward(self,x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.proj:
            x = self.proj_bn(self.proj_conv(x))
        return self.relu(x+y)
if __name__ == '__main__':
    tensor = torch.rand(3,64,224,224)
    net = Bottleneck(64,256,stride=2,proj=True)

    print(net(tensor).shape)