import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary
# from model.VGGUNet.vgg16 import VGG16
from torch.utils.tensorboard import SummaryWriter


class Double_CBR(nn.Module):
    """
    Double_CBR 是 Conv BN Relu 两次堆叠
    """

    def __init__(self, in_channel, out_channel, is_pooling=False):
        super(Double_CBR, self).__init__()
        self.is_pooling = is_pooling
        self.CBR2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1,),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU6(),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU6()
        )
        if self.is_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        if self.is_pooling:
            x = self.pool(x)
        x = self.CBR2(x)
        return x


class Unet_Skip_Up(nn.Module):
    """
    Unet_Skip_Up 是 Unet 跳跃链接模块
    """

    def __init__(self, in_channel, out_channel):
        super(Unet_Skip_Up, self).__init__()
        self.CBR2 = Double_CBR(in_channel, out_channel, is_pooling=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, feat_encoder, feat_up):
        feat_up = self.upsample(feat_up)
        output_feat = torch.cat([feat_encoder, feat_up], dim=1)
        output_feat = self.CBR2(output_feat)
        return output_feat


class Unet_Encoder(nn.Module):
    def __init__(self, in_channel=3):
        super(Unet_Encoder, self).__init__()
        self.CBR_512 = Double_CBR(in_channel, 32, is_pooling=False)
        self.CBR_256 = Double_CBR(32, 64, is_pooling=True)
        self.CBR_128 = Double_CBR(64, 128, is_pooling=True)
        self.CBR_64 = Double_CBR(128, 256, is_pooling=True)
        self.CBR_32 = Double_CBR(256, 256, is_pooling=True)

    def forward(self, x):
        feat_512 = self.CBR_512(x)
        feat_256 = self.CBR_256(feat_512)
        feat_128 = self.CBR_128(feat_256)
        feat_64 = self.CBR_64(feat_128)
        feat_32 = self.CBR_32(feat_64)

        return feat_512, feat_256, feat_128, feat_64, feat_32


class Att_block(nn.Module):
    def __init__(self, channel_g, channel_x, channel_mid):
        super(Att_block, self).__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(channel_g, channel_mid, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(channel_mid)
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(channel_x, channel_mid, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(channel_mid)
        )
        self.weight = nn.Sequential(
            nn.Conv2d(channel_mid, 1, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, g, x):
        g1 = self.upsample(g)
        g1 = self.Wg(g1)
        x1 = self.Wx(x)
        psi = F.relu6(g1 + x1)
        weight = self.weight(psi)

        return weight * x


class Attention_Unet_Vgg(nn.Module):
    def __init__(self, n_classes=2, in_channels=3):
        super(Attention_Unet_Vgg, self).__init__()
        print(self.__doc__)
        self.encoder = Unet_Encoder(in_channels)
        self.skip_up_64 = Unet_Skip_Up(512, 128)
        self.skip_up_128 = Unet_Skip_Up(256, 64)
        self.skip_up_256 = Unet_Skip_Up(128, 32)
        self.skip_up_512 = Unet_Skip_Up(64, 32)
        self.att_block_64 = Att_block(256, 256, 128)
        self.att_block_128 = Att_block(128, 128, 64)
        self.att_block_256 = Att_block(64, 64, 32)
        self.cls_conv = nn.Conv2d(32, n_classes, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        feat_512, feat_256, feat_128, feat_64, feat_32 = self.encoder(x)
        feat_64_w = self.att_block_64(feat_32, feat_64)
        up_64 = self.skip_up_64(feat_64_w, feat_32)

        feat_128_w = self.att_block_128(up_64, feat_128)
        up_128 = self.skip_up_128(feat_128_w, up_64)

        feat_256_w = self.att_block_256(up_128, feat_256)
        up_256 = self.skip_up_256(feat_256_w, up_128)

        up_512 = self.skip_up_512(feat_512, up_256)

        results = self.cls_conv(up_512)

        # argmax = torch.argmax(results, dim=1)
        return results
        # if self.training:
        #     return results, argmax
        # else:
        #     return argmax


if __name__ == "__main__":
    attention_unet_vgg = Attention_Unet_Vgg(n_classes=3, in_channels=3)
    summary(attention_unet_vgg, (3, 512, 512), device='cpu')

    # dummy_input = torch.randn(1, 3, 512, 512)
    # outputs = base_unet_vgg_official(dummy_input)
    # print(outputs.shape)

    # writer = SummaryWriter("arch_plot/" + base_unet_vgg_official._get_name())
    # writer.add_graph(base_unet_vgg_official, torch.randn((1, 3, 512, 512)))
    # writer.close()
