from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

class DINOv2FeatureExtractor:
    def __init__(self, model, patch_size=14, device: Optional[str] = None):
        self.model = model.eval()
        self.patch_size = int(patch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    @classmethod
    def from_torch_hub(cls, variant="dinov2_vits14", device=None):
        model = torch.hub.load("facebookresearch/dinov2", variant)
        return cls(model, patch_size=14, device=device)

    @staticmethod
    def _prepare(image_rgb, patch_size, device):
        x = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        h, w = x.shape[-2:]
        hh = max(patch_size, (h // patch_size) * patch_size)
        ww = max(patch_size, (w // patch_size) * patch_size)
        if (hh, ww) != (h, w):
            x = F.interpolate(x, size=(hh, ww), mode="bilinear", align_corners=False)
        return x.to(device), hh, ww

    @torch.no_grad()
    def __call__(self, image_rgb):
        x, h, w = self._prepare(image_rgb, self.patch_size, self.device)
        out = self.model.forward_features(x) if hasattr(self.model, "forward_features") else self.model(x)
        if isinstance(out, dict):
            if "x_norm_patchtokens" in out:
                out = out["x_norm_patchtokens"]
            else:
                out = next(value for value in out.values() if torch.is_tensor(value))
        if out.dim() == 4:
            feat = out
        elif out.dim() == 3:
            gh, gw = h // self.patch_size, w // self.patch_size
            if out.shape[1] == gh * gw + 1:
                out = out[:, 1:]
            feat = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], gh, gw)
        else:
            raise ValueError(f"Unsupported DINOv2 output shape: {tuple(out.shape)}")
        return F.normalize(feat.float(), dim=1)
