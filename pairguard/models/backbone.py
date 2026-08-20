from __future__ import annotations
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class SiameseMobileNetV3Small(nn.Module):
    def __init__(self, pretrained=False, proj_channels=(24, 32, 48, 64)):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.encoder = mobilenet_v3_small(weights=weights).features
        self.capture = {0: (2, 16), 1: (4, 16), 2: (8, 24), 4: (16, 40)}
        self.proj = nn.ModuleDict()
        for stride, in_channels, out_channels in zip(
            (2, 4, 8, 16),
            (16, 16, 24, 40),
            proj_channels,
        ):
            self.proj[str(stride)] = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )

    def encode_one(self, x):
        outputs = {}
        z = x
        for index, block in enumerate(self.encoder):
            z = block(z)
            if index in self.capture:
                stride, _ = self.capture[index]
                outputs[stride] = self.proj[str(stride)](z)
            if len(outputs) == 4:
                break
        return outputs

    def forward(self, img1, img2):
        return self.encode_one(img1), self.encode_one(img2)
