from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .backbone import SiameseMobileNetV3Small

class SeedCDNet(nn.Module):
    def __init__(self, pretrained_backbone=False):
        super().__init__()
        self.backbone = SiameseMobileNetV3Small(pretrained=pretrained_backbone)
        self.f16 = nn.Conv2d(64 * 3, 48, 3, padding=1)
        self.f8 = nn.Conv2d(48 * 3 + 48, 48, 3, padding=1)
        self.f4 = nn.Conv2d(32 * 3 + 48, 32, 3, padding=1)
        self.f2 = nn.Conv2d(24 * 3 + 32, 24, 3, padding=1)
        self.head = nn.Conv2d(24, 1, 1)

    @staticmethod
    def ordered(src, tgt):
        return torch.cat([src, tgt, tgt - src], dim=1)

    def forward(self, src, tgt):
        fs, ft = self.backbone(src, tgt)
        d16 = F.silu(self.f16(self.ordered(fs[16], ft[16])))
        d8 = F.silu(
            self.f8(
                torch.cat(
                    [
                        self.ordered(fs[8], ft[8]),
                        F.interpolate(d16, size=fs[8].shape[-2:], mode="bilinear", align_corners=False),
                    ],
                    dim=1,
                )
            )
        )
        d4 = F.silu(
            self.f4(
                torch.cat(
                    [
                        self.ordered(fs[4], ft[4]),
                        F.interpolate(d8, size=fs[4].shape[-2:], mode="bilinear", align_corners=False),
                    ],
                    dim=1,
                )
            )
        )
        d2 = F.silu(
            self.f2(
                torch.cat(
                    [
                        self.ordered(fs[2], ft[2]),
                        F.interpolate(d4, size=fs[2].shape[-2:], mode="bilinear", align_corners=False),
                    ],
                    dim=1,
                )
            )
        )
        return F.interpolate(self.head(d2), size=src.shape[-2:], mode="bilinear", align_corners=False)
