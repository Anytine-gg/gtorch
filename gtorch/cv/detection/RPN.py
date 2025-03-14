import torch
from torch import nn
import torch.nn.functional as F


class RPNHead(nn.Module):
    def __init__(self, in_channels, nAnchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.cls = nn.Conv2d(256, 2 * nAnchors, 1)
        self.bbox_reg = nn.Conv2d(256, 4 * nAnchors, 1)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
               nn.init.normal_(m.weight, std=0.01)
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls(x)
        reg = self.bbox_reg(x)
        return logits, reg
