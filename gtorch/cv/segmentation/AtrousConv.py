import torch
import torch.nn as nn


class AtrousConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
    ):
        super(AtrousConv, self).__init__()
        if padding is None:  # 自动计算大小输入输出不变
            padding = kernel_size + (dilation - 1) * (kernel_size - 1) - 1
            padding //= 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )

    def forward(self, x):
        return self.conv(x)


if __name__ == "__main__":

    atrous_conv = AtrousConv(
        in_channels=3, out_channels=16, kernel_size=5, dilation=2, padding=None
    )

    input_tensor = torch.randn(1, 3, 64, 64)  # (batch_size, channels, height, width)

    output_tensor = atrous_conv(input_tensor)

    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output_tensor.shape)
