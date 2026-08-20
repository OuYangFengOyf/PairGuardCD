from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F

def offset_list(radius):
    return [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]

def _shifted(target, dx, dy):
    batch, channels, height, width = target.shape
    output = torch.zeros_like(target)
    src_x1 = max(0, dx)
    src_x2 = min(width, width + dx)
    src_y1 = max(0, dy)
    src_y2 = min(height, height + dy)
    dst_x1 = max(0, -dx)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y1 = max(0, -dy)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    if src_x2 > src_x1 and src_y2 > src_y1:
        output[:, :, dst_y1:dst_y2, dst_x1:dst_x2] = target[:, :, src_y1:src_y2, src_x1:src_x2]
    return output

def offset_affinity_volume(source, target, radius, temperature=0.07):
    volumes = []
    for dx, dy in offset_list(radius):
        shifted = _shifted(target, dx, dy)
        volumes.append((source * shifted).sum(1, keepdim=True) / temperature)
    return torch.cat(volumes, dim=1)

def coarse_index_to_offset(index, radius):
    width = 2 * radius + 1
    dy = torch.div(index, width, rounding_mode="floor") - radius
    dx = index % width - radius
    return dx, dy

def _sample_variable_offset(target, dx, dy):
    batch, _, height, width = target.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=target.device),
        torch.arange(width, device=target.device),
        indexing="ij",
    )
    xx = xx.float().view(1, height, width).expand(batch, -1, -1) + dx.float()
    yy = yy.float().view(1, height, width).expand(batch, -1, -1) + dy.float()
    gx = 2.0 * xx / max(1, width - 1) - 1.0
    gy = 2.0 * yy / max(1, height - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(target, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

def fine_affinity_volume(source, target, center_dx, center_dy, radius, temperature=0.07):
    volumes = []
    for residual_dx, residual_dy in offset_list(radius):
        sampled = _sample_variable_offset(
            target,
            center_dx + residual_dx,
            center_dy + residual_dy,
        )
        volumes.append((source * sampled).sum(1, keepdim=True) / temperature)
    return torch.cat(volumes, dim=1)

def normalized_entropy(probability):
    entropy = -(probability.clamp_min(1e-8).log() * probability).sum(1, keepdim=True)
    return entropy / np.log(probability.shape[1])

def geometric_bias(logits, dx_prior, dy_prior, quality, radius, weight=0.5, sigma=3.0):
    bias = []
    for dx, dy in offset_list(radius):
        distance = (dx_prior - dx).abs() + (dy_prior - dy).abs()
        bias.append(-(weight / sigma) * quality * distance)
    return logits + torch.cat(bias, dim=1)

class OffsetNullRelationTeacher:
    def __init__(self, coarse_radius=6, fine_radius=2, temperature=0.07):
        self.coarse_radius = int(coarse_radius)
        self.fine_radius = int(fine_radius)
        self.temperature = float(temperature)

    def _direction(
        self,
        source_coarse,
        target_coarse,
        source_fine,
        target_fine,
        valid_coarse,
        valid_fine,
        certified_null_coarse,
        certified_null_fine,
        utility_coarse,
        utility_fine,
        geometry,
        prefix,
    ):
        coarse_logits = offset_affinity_volume(
            source_coarse,
            target_coarse,
            self.coarse_radius,
            self.temperature,
        )
        if geometry is not None:
            coarse_logits = geometric_bias(
                coarse_logits,
                geometry[f"dx{prefix}"],
                geometry[f"dy{prefix}"],
                geometry[f"q{prefix}"],
                self.coarse_radius,
            )

        nonnull_probability = F.softmax(coarse_logits, dim=1)
        valid_coarse = valid_coarse.float()
        certified_null_coarse = certified_null_coarse.float()
        coarse_probability = torch.cat(
            [
                nonnull_probability * valid_coarse,
                certified_null_coarse,
            ],
            dim=1,
        )
        unknown = valid_coarse + certified_null_coarse <= 0
        coarse_probability[:, 0:1] = torch.where(
            unknown,
            torch.ones_like(coarse_probability[:, 0:1]),
            coarse_probability[:, 0:1],
        )
        coarse_probability = coarse_probability / (
            coarse_probability.sum(1, keepdim=True) + 1e-6
        )

        coarse_index = nonnull_probability.argmax(1)
        coarse_dx, coarse_dy = coarse_index_to_offset(coarse_index, self.coarse_radius)
        fine_height, fine_width = source_fine.shape[-2:]
        center_dx = F.interpolate(
            coarse_dx.unsqueeze(1).float(),
            size=(fine_height, fine_width),
            mode="nearest",
        ).squeeze(1) * 2.0
        center_dy = F.interpolate(
            coarse_dy.unsqueeze(1).float(),
            size=(fine_height, fine_width),
            mode="nearest",
        ).squeeze(1) * 2.0

        fine_logits = fine_affinity_volume(
            source_fine,
            target_fine,
            center_dx,
            center_dy,
            self.fine_radius,
            self.temperature,
        )
        fine_probability = F.softmax(fine_logits, dim=1) * valid_fine.float()

        coarse_reliability = (1.0 - normalized_entropy(nonnull_probability)) * valid_coarse
        if utility_coarse is not None:
            coarse_reliability = coarse_reliability * utility_coarse
        if geometry is not None:
            geometric_quality = geometry[f"q{prefix}"]
            coarse_reliability = coarse_reliability * torch.where(
                geometric_quality > 0,
                geometric_quality,
                torch.ones_like(geometric_quality),
            )
            null_reliability = geometry.get(
                f"null_r{prefix}",
                certified_null_coarse,
            )
        else:
            null_reliability = certified_null_coarse
        coarse_reliability = coarse_reliability + certified_null_coarse * null_reliability

        fine_nonnull = fine_probability / (fine_probability.sum(1, keepdim=True) + 1e-6)
        fine_reliability = (1.0 - normalized_entropy(fine_nonnull)) * valid_fine.float()
        if utility_fine is not None:
            fine_reliability = fine_reliability * utility_fine

        return {
            "coarse": coarse_probability,
            "fine": fine_probability,
            "r_coarse": coarse_reliability.clamp(0.0, 1.0),
            "r_fine": fine_reliability.clamp(0.0, 1.0),
            "dx": coarse_dx.unsqueeze(1).float(),
            "dy": coarse_dy.unsqueeze(1).float(),
        }

    def build(
        self,
        z1_coarse,
        z2_coarse,
        z1_fine,
        z2_fine,
        valid_coarse_12,
        valid_coarse_21,
        valid_fine_12,
        valid_fine_21,
        null_coarse_12,
        null_coarse_21,
        null_fine_12,
        null_fine_21,
        utility_coarse_12,
        utility_coarse_21,
        utility_fine_12,
        utility_fine_21,
        geom_prior: Optional[Dict[str, torch.Tensor]] = None,
    ):
        direction12 = self._direction(
            z1_coarse,
            z2_coarse,
            z1_fine,
            z2_fine,
            valid_coarse_12,
            valid_fine_12,
            null_coarse_12,
            null_fine_12,
            utility_coarse_12,
            utility_fine_12,
            geom_prior,
            "12",
        )
        direction21 = self._direction(
            z2_coarse,
            z1_coarse,
            z2_fine,
            z1_fine,
            valid_coarse_21,
            valid_fine_21,
            null_coarse_21,
            null_fine_21,
            utility_coarse_21,
            utility_fine_21,
            geom_prior,
            "21",
        )
        return {
            "coarse_12": direction12["coarse"],
            "coarse_21": direction21["coarse"],
            "fine_12": direction12["fine"],
            "fine_21": direction21["fine"],
            "rT_coarse_12": direction12["r_coarse"],
            "rT_coarse_21": direction21["r_coarse"],
            "rT_fine_12": direction12["r_fine"],
            "rT_fine_21": direction21["r_fine"],
            "teacher_dx12": direction12["dx"],
            "teacher_dy12": direction12["dy"],
            "teacher_dx21": direction21["dx"],
            "teacher_dy21": direction21["dy"],
        }
