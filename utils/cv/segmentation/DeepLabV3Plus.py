from turtle import forward
from matplotlib.pyplot import cla
import torch
from torch import nn
import torchsummary
import torchvision
from AtrousConv import AtrousConv
import torch.nn.functional as F
from DeepLabV3 import ASPP


class DeepLabEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet50(
            replace_stride_with_dilation=[False, False, True]  # 仅改动layer4为空洞卷积
        )
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.aspp = ASPP(2048, 256, [6, 12, 18])

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)  # 256通道(低级特征)
        y = self.layer2(x)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.aspp(y)
        return x, y


class DeepLabDecoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        # 将Low-level features从256调到48，使得aspp占主导
        self.proj = nn.Conv2d(256, 48, 1)
        self.output = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, out_channels, 1, stride=1),
        )

    def forward(self, low_feat, aspp_feat):
        low_feat = self.proj(low_feat)
        aspp_feat = self.upsample(aspp_feat)
        output = torch.concat([low_feat, aspp_feat], dim=1)
        output = self.output(output)
        return output

class DeepLabV3Plus(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.encoder = DeepLabEncoder()
        self.decoder = DeepLabDecoder(out_channels)
    def forward(self,x):
        x,y = self.encoder(x)
        output = self.decoder(x,y)
        output = F.interpolate(output, scale_factor=4, mode='bilinear', align_corners=True)
        return output

        
if __name__ == "__main__":
    model = DeepLabV3Plus(5)
    tensor = torch.randn(2,3,480,480)
    print(model(tensor).shape)
    #torchsummary.summary(model, (3, 480, 480), -1, "cpu")
