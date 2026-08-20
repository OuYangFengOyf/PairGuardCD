from __future__ import annotations
import math
from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F

def offset_table(radius, device=None, dtype=torch.float32):
    values = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    return torch.tensor(values, device=device, dtype=dtype)

def expected_offset(logits, radius, has_null=False):
    if has_null:
        probability = F.softmax(logits, dim=1)
        nonnull = probability[:, :-1]
        null = probability[:, -1:]
        nonnull = nonnull / (nonnull.sum(dim=1, keepdim=True) + 1e-6)
    else:
        nonnull = F.softmax(logits, dim=1)
        null = torch.zeros_like(nonnull[:, :1])
    table = offset_table(radius, logits.device, logits.dtype)
    dx = (nonnull * table[:, 0].view(1, -1, 1, 1)).sum(1, keepdim=True)
    dy = (nonnull * table[:, 1].view(1, -1, 1, 1)).sum(1, keepdim=True)
    entropy = -(nonnull.clamp_min(1e-8).log() * nonnull).sum(1, keepdim=True)
    confidence = 1.0 - entropy / max(1e-6, math.log(float(nonnull.shape[1])))
    return dx, dy, null, confidence.clamp(0.0, 1.0)

class DirectionalRelationHead(nn.Module):
    def __init__(
        self,
        c8=48,
        c16=64,
        coarse_classes=170,
        fine_classes=25,
        z_channels=24,
        coarse_radius=6,
        fine_radius=2,
    ):
        super().__init__()
        self.coarse_radius = coarse_radius
        self.fine_radius = fine_radius
        self.coarse12 = self._head(c16 * 3, coarse_classes)
        self.coarse21 = self._head(c16 * 3, coarse_classes)
        self.fine12 = self._head(c8 * 3 + 4, fine_classes)
        self.fine21 = self._head(c8 * 3 + 4, fine_classes)
        self.zproj = nn.Sequential(
            nn.Conv2d(12, z_channels, 1, bias=False),
            nn.BatchNorm2d(z_channels),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _head(in_channels, out_channels):
        hidden = max(64, in_channels // 2)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
        )

    @staticmethod
    def ordered(src, tgt):
        return torch.cat([src, tgt, tgt - src], dim=1)

    def _fine_input(self, src8, tgt8, coarse):
        dx, dy, null, confidence = expected_offset(coarse, self.coarse_radius, True)
        cue = torch.cat([dx, dy, null, confidence], dim=1)
        cue = F.interpolate(cue, size=src8.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([self.ordered(src8, tgt8), cue], dim=1)

    def forward(self, f1: Dict[int, torch.Tensor], f2: Dict[int, torch.Tensor]):
        coarse12 = self.coarse12(self.ordered(f1[16], f2[16]))
        coarse21 = self.coarse21(self.ordered(f2[16], f1[16]))
        fine12 = self.fine12(self._fine_input(f1[8], f2[8], coarse12))
        fine21 = self.fine21(self._fine_input(f2[8], f1[8], coarse21))

        dx12, dy12, null12, conf12 = expected_offset(coarse12, self.coarse_radius, True)
        dx21, dy21, null21, conf21 = expected_offset(coarse21, self.coarse_radius, True)
        fdx12, fdy12, _, _ = expected_offset(fine12, self.fine_radius, False)
        fdx21, fdy21, _, _ = expected_offset(fine21, self.fine_radius, False)

        coarse_parts = [dx12, dy12, null12, conf12, dx21, dy21, null21, conf21]
        coarse_parts = [
            F.interpolate(item, size=fine12.shape[-2:], mode="bilinear", align_corners=False)
            for item in coarse_parts
        ]
        relation_code = self.zproj(
            torch.cat(coarse_parts + [fdx12, fdy12, fdx21, fdy21], dim=1)
        )

        return {
            "coarse_12": coarse12,
            "coarse_21": coarse21,
            "fine_12": fine12,
            "fine_21": fine21,
            "ZR": relation_code,
            "offset12": torch.cat(coarse_parts[:2], dim=1),
            "offset21": torch.cat(coarse_parts[4:6], dim=1),
        }
