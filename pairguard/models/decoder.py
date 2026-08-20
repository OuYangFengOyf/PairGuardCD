import torch
from torch import nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class RelationGuidedDecoder(nn.Module):
    def __init__(self, channels=(24, 32, 48, 64), z_channels=24):
        super().__init__()
        c2, c4, c8, c16 = channels
        self.d16 = DecoderBlock(c16, 64)
        self.d8 = DecoderBlock(c8 + 64 + z_channels, 64)
        self.d4 = DecoderBlock(c4 + 64 + z_channels, 48)
        self.d2 = DecoderBlock(c2 + 48, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, fused, relation_code, out_size):
        d16 = self.d16(fused[16])
        d16_up = F.interpolate(
            d16,
            size=fused[8].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        z8 = F.interpolate(
            relation_code,
            size=fused[8].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        d8 = self.d8(torch.cat([fused[8], d16_up, z8], dim=1))

        d8_up = F.interpolate(
            d8,
            size=fused[4].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        z4 = F.interpolate(
            relation_code,
            size=fused[4].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        d4 = self.d4(torch.cat([fused[4], d8_up, z4], dim=1))

        d4_up = F.interpolate(
            d4,
            size=fused[2].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        d2 = self.d2(torch.cat([fused[2], d4_up], dim=1))
        base_logit = F.interpolate(
            self.head(d2),
            size=out_size,
            mode="bilinear",
            align_corners=False,
        )
        return {
            "D2": d2,
            "base_logit": base_logit,
            "base_prob": torch.sigmoid(base_logit),
        }
