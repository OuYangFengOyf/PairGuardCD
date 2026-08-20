from __future__ import annotations
import numpy as np
import cv2
import torch
import torch.nn.functional as F

def normalized_abs_difference(img1, img2):
    a = img1.astype(np.float32) / 255.0
    b = img2.astype(np.float32) / 255.0
    diff = np.abs(a - b).mean(axis=2)
    low, high = np.percentile(diff, [1, 99])
    return np.clip((diff - low) / (high - low + 1e-6), 0.0, 1.0).astype(np.float32)

def _shift_tensor(x, dx, dy):
    out = torch.zeros_like(x)
    h, w = x.shape[-2:]
    sx1 = max(0, -dx)
    sx2 = min(w, w - dx)
    sy1 = max(0, -dy)
    sy2 = min(h, h - dy)
    tx1 = max(0, dx)
    tx2 = tx1 + (sx2 - sx1)
    ty1 = max(0, dy)
    ty2 = ty1 + (sy2 - sy1)
    if sx2 > sx1 and sy2 > sy1:
        out[..., ty1:ty2, tx1:tx2] = x[..., sy1:sy2, sx1:sx2]
    return out

@torch.no_grad()
def local_correspondence_evidence(zsrc, ztgt, out_size, radius=4, min_similarity=0.35):
    similarities = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = _shift_tensor(ztgt, dx, dy)
            similarities.append((zsrc * shifted).sum(dim=1, keepdim=True))
    volume = torch.cat(similarities, dim=1)
    best = volume.max(dim=1, keepdim=True).values
    difference = (1.0 - best).clamp(0.0, 2.0) / 2.0
    valid = (best >= min_similarity).float()
    difference = F.interpolate(difference, size=out_size, mode="bilinear", align_corners=False)
    valid = F.interpolate(valid, size=out_size, mode="nearest")
    return difference[0, 0].cpu().numpy().astype(np.float32), valid[0, 0].cpu().numpy().astype(np.uint8)

def global_phase_translation(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    shift, response = cv2.phaseCorrelate(g1, g2)
    return float(shift[0]), float(shift[1]), float(np.clip(response, 0.0, 1.0))
