import torch
from torch import nn
import torchsummary


class AlexNet(nn.Module):
    def __init__(self, in_channels,nClasses):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, 11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96,256,5,padding=2,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256,384,3,padding=1,stride=1),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384,384,3,padding=1,stride=1),
            nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384,256,3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256,4096,6,stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.layer7 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.layer8 = nn.Linear(4096,nClasses)
       
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x


if __name__ == "__main__":
    net = AlexNet(3,1000)
    tensor = torch.randn(1, 3, 227, 227)
    print(net(tensor).shape)
