import torch
from torch import nn
import torchvision
import torchsummary
import torch.nn.functional as F
class UpSampling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,pooling_indices,features):
        
        pass
class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16_bn()
        # 去掉maxpooling的vgg layer
        self.layer1 = vgg16.features[0:6]
        self.layer2 = vgg16.features[7:13]
        self.layer3 = vgg16.features[14:23]
        self.layer4 = vgg16.features[24:33]
        self.layer5 = vgg16.features[34:43]
    def forward(self,x):
        x = self.layer1(x)
        x,idx1 = F.max_pool1d_with_indices(x,kernel_size=2,stride=2)
        x = self.layer2(x)
        x,idx2 = F.max_pool2d_with_indices(x,kernel_size=2,stride=2)
        x = self.layer3(x)
        x,idx3 = F.max_pool2d_with_indices(x,kernel_size=2,stride=2)
        x = self.layer4(x)
        x,idx4 = F.max_pool2d_with_indices(x,kernel_size=2,stride=2)
        x = self.layer5(x)
        x,idx5 = F.max_pool2d_with_indices(x,kernel_size=2,stride=2)
        return x
    
if __name__ == '__main__':
    net = SegNet()
    torchsummary.summary(net,(3,224,224),-1,'cpu')