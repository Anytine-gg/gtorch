from torch import nn
import torchsummary
import torchvision
import torch.nn.functional as F
from gtorch.cv.segmentation.ASPP import ASPP

class DeepLabV3(nn.Module):
    def __init__(self, out_channels, pretrained=False):
        super().__init__()
        backbone = torchvision.models.resnet50(
            replace_stride_with_dilation=[
                False,
                True,
                True,
            ],  # 改动layer3 layer4为空洞卷积(不下采样)
            pretrained=pretrained,
        )
        backbone.conv1
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.aspp = ASPP(2048, 256, [12, 24, 36])
        self.conv = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.proj = nn.Conv2d(256, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
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
    model = DeepLabV3(3)
    torchsummary.summary(model, (3, 480, 480), -1, "cpu")
