import torch
import numpy as np
from torch import nn
import torchsummary

class UnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = UnetBlock(in_channel, 64)
        self.conv2 = UnetBlock(64, 128)
        self.conv3 = UnetBlock(128, 256)
        self.conv4 = UnetBlock(256, 512)
        self.conv5 = UnetBlock(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = UnetBlock(1024, 512)
        self.upconv2 = UnetBlock(512, 256)
        self.upconv3 = UnetBlock(256, 128)
        self.upconv4 = UnetBlock(128, 64)
        self.merge = nn.Conv2d(64, out_channel, kernel_size=1)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
    def crop_and_merge(self, x, target):
        target = torch.cat((x, target), dim=1)
        return target
        
    def forward(self, x):
        x1 = self.conv1(x)
        down1 = self.maxPool(x1)
        x2 = self.conv2(down1)
        down2 = self.maxPool(x2)
        x3 = self.conv3(down2)
        down3 = self.maxPool(x3)
        x4 = self.conv4(down3)
        down4 = self.maxPool(x4)
        x5 = self.conv5(down4)
        up = self.up1(x5)
        up = self.crop_and_merge(x4, up)
        up = self.upconv1(up)
        up = self.up2(up)
        up = self.crop_and_merge(x3, up)
        up = self.upconv2(up)
        up = self.up3(up)
        up = self.crop_and_merge(x2, up)
        up = self.upconv3(up)
        up = self.up4(up)
        up = self.crop_and_merge(x1, up)
        up = self.upconv4(up)
        up = self.merge(up)
        return up


if __name__ == "__main__":
    tensor = torch.randn(1, 3, 1256, 1256).to("mps")
    net = UNet(3, 4)
    #net.to("mps")
    #print(net(tensor).shape)
    torchsummary.summary(net,input_size=(3,256,256))
