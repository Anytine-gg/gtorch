import torch
from torch import nn
import torchsummary
from gtorch.utils.datasets.VOCDetection_ import VOCDetection_
from gtorch.utils.datasets.YOLOv3Dataset import YOLOv3_Dataset
from gtorch.cv.detection.tools import yolo3_loss
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
class DarknetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        # 当下采样（stride>1）或者通道数不匹配时，使用1×1卷积投影
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.leaky_relu(x + out)


class ConvSet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        half_channels = in_channels // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, half_channels, 1, bias=False),
            nn.BatchNorm2d(half_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(half_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, half_channels, 1, bias=False),
            nn.BatchNorm2d(half_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(half_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class YOLOv3(nn.Module):
    def __init__(self, num_of_classes, in_channels=3):
        super().__init__()
        out_channels = 3 * (5 + num_of_classes)

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            DarknetResidualBlock(64, 64, 1),
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            DarknetResidualBlock(128, 128, 1), DarknetResidualBlock(128, 128, 1)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            DarknetResidualBlock(256, 256, 1),
            DarknetResidualBlock(256, 256, 1),
            DarknetResidualBlock(256, 256, 1),
            DarknetResidualBlock(256, 256, 1),
            DarknetResidualBlock(256, 256, 1),
            DarknetResidualBlock(256, 256, 1),
            DarknetResidualBlock(256, 256, 1),
            DarknetResidualBlock(256, 256, 1),
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            DarknetResidualBlock(512, 512, 1),
            DarknetResidualBlock(512, 512, 1),
            DarknetResidualBlock(512, 512, 1),
            DarknetResidualBlock(512, 512, 1),
            DarknetResidualBlock(512, 512, 1),
            DarknetResidualBlock(512, 512, 1),
            DarknetResidualBlock(512, 512, 1),
            DarknetResidualBlock(512, 512, 1),
        )
        self.downsample4 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            DarknetResidualBlock(1024, 1024, 1),
            DarknetResidualBlock(1024, 1024, 1),
            DarknetResidualBlock(1024, 1024, 1),
            DarknetResidualBlock(1024, 1024, 1),
        )
        self.conv_set3 = ConvSet(1024, 512)
        self.get_map3 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, out_channels, 1),
        )
        self.upsample_conv_3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.conv_set2 = ConvSet(768, 256)
        self.get_map2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, out_channels, 1),
        )
        self.upsample_conv_2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.conv_set1 = ConvSet(384, 128)
        self.get_map1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.downsample1(x)
        x = self.layer2(x)
        x = self.downsample2(x)
        layer3_out = self.layer3(x)
        x = self.downsample3(layer3_out)
        layer4_out = self.layer4(x)
        x = self.downsample4(layer4_out)
        layer5_out = self.layer5(x)
        layer5_out = self.conv_set3(layer5_out)
        layer4_up = self.upsample_conv_3(layer5_out)
        layer4_out = torch.cat([layer4_out,layer4_up],dim=1)
        layer4_out = self.conv_set2(layer4_out)
        layer3_up = self.upsample_conv_2(layer4_out)
        layer3_out = torch.cat([layer3_out,layer3_up],dim=1)
        layer3_out = self.conv_set1(layer3_out)
        feat_map1 = self.get_map1(layer3_out)
        feat_map2 = self.get_map2(layer4_out)
        feat_map3 = self.get_map3(layer5_out)
        return feat_map1,feat_map2,feat_map3


if __name__ == "__main__":
    net = YOLOv3(20, 3)
    net.to("cuda")
    tensor = torch.rand(3, 3, 416, 416).to("cuda")
    epoch = 100
    
    transform = transform = A.Compose(
        [
            A.LongestMaxSize(max_size=416),
            A.PadIfNeeded(
                min_height=416,
                min_width=416,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(128, 128, 128),
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

    dataset = VOCDetection_(transform=transform)
    dataset = YOLOv3_Dataset(20, dataset,device='cuda')
    
    
    for i in range(epoch):
        # image,feat_map1,feat_map2,feat_map3 = 
        