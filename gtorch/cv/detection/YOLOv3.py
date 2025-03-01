from matplotlib.pyplot import sca
import torch
from torch import nn
import torchsummary
from gtorch.models.Bottleneck import Bottleneck


class YOLOv3(nn.Module):
    def __init__(self, num_of_classes, in_channels=3):
        super().__init__()
        out_channels = 3*(5+num_of_classes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, 32, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.res = Bottleneck(32, 64, 2, proj=True)
        self.seq1 = nn.Sequential(
            Bottleneck(64, 128, 2, proj=True),
            Bottleneck(128, 128, 1, proj=False),
        )
        self.seq2 = nn.Sequential(
            Bottleneck(128, 256, 2, proj=True),
            Bottleneck(256, 256, 1, proj=False),
            Bottleneck(256, 256, 1, proj=False),
            Bottleneck(256, 256, 1, proj=False),
            Bottleneck(256, 256, 1, proj=False),
            Bottleneck(256, 256, 1, proj=False),
            Bottleneck(256, 256, 1, proj=False),
            Bottleneck(256, 256, 1, proj=False)
        )
        self.seq3 = nn.Sequential(
            Bottleneck(256, 512, 2, proj=True),
            Bottleneck(512, 512, 1, proj=False),
            Bottleneck(512, 512, 1, proj=False),
            Bottleneck(512, 512, 1, proj=False),
            Bottleneck(512, 512, 1, proj=False),
            Bottleneck(512, 512, 1, proj=False),
            Bottleneck(512, 512, 1, proj=False),
            Bottleneck(512, 512, 1, proj=False)
        )
        self.seq4 = nn.Sequential(
            Bottleneck(512, 1024, 2, proj=True),
            Bottleneck(1024, 1024, 1, proj=False),
            Bottleneck(1024, 1024, 1, proj=False),
            Bottleneck(1024, 1024, 1, proj=False)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 最近邻上采样

        self.conv_map3 = nn.Sequential(
            nn.Conv2d(1024,1024,3,padding=1,bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024,out_channels,1)
        )
        self.conv_upsample_from_map3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(1024,256,1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_map2 = nn.Sequential(
            nn.Conv2d(512+256)
        )
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.res(x)
        x = self.seq1(x)
        x1 = self.seq2(x)
        
        x2 = self.seq3(x1)
        
        x3 = self.seq4(x2)
        x2_up = self.conv_upsample_from_map3(x3)
        x2_up = torch.cat([x2,x2_up],dim=1)
        
        
        return self.conv_map3(x3)
if __name__ == '__main__':
    net = YOLOv3(20,3)
    net.to('cuda')
    tensor = torch.rand(3,3,416,416).to('cuda')
    print(net(tensor).shape)