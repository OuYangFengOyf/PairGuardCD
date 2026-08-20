import torch
from torch import nn

class EIFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, feature1, feature2):
        summed = feature1 + feature2
        difference = (feature1 - feature2).abs()
        product = feature1 * feature2
        return self.net(torch.cat([summed, difference, product], dim=1))

class MultiScaleEIFusion(nn.Module):
    def __init__(self, channels=(24, 32, 48, 64), strides=(2, 4, 8, 16)):
        super().__init__()
        self.blocks = nn.ModuleDict(
            {
                str(stride): EIFusionBlock(channel)
                for stride, channel in zip(strides, channels)
            }
        )

    def forward(self, features1, features2):
        return {
            stride: self.blocks[str(stride)](
                features1[stride],
                features2[stride],
            )
            for stride in features1
        }
