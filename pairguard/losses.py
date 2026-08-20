from __future__ import annotations
import torch
import torch.nn.functional as F
from pairguard.models.relation import expected_offset

def dice_loss(probability, target, mask=None, eps=1e-6):
    if mask is None:
        mask = torch.ones_like(target)
    probability = probability * mask
    target = target * mask
    intersection = (probability * target).sum((1, 2, 3))
    denominator = probability.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()

def bce_dice_from_logit(logit, target, mask=None):
    if mask is None:
        mask = torch.ones_like(target)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    bce = (bce * mask).sum() / (mask.sum() + 1e-6)
    return bce + dice_loss(torch.sigmoid(logit), target, mask)


def seed_recall_loss(logit, target, pos_weight=2.0, alpha=0.3, beta=0.7, eps=1e-6):
    weight = torch.tensor([pos_weight], device=logit.device, dtype=logit.dtype)
    bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=weight)
    probability = torch.sigmoid(logit)
    tp = (probability * target).sum((1, 2, 3))
    fp = (probability * (1.0 - target)).sum((1, 2, 3))
    fn = ((1.0 - probability) * target).sum((1, 2, 3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return bce + (1.0 - tversky).mean()

def masked_kl(student_logits, teacher_probability, reliability):
    if reliability.dim() == 3:
        reliability = reliability.unsqueeze(1)
    student_log_probability = F.log_softmax(student_logits, dim=1)
    kl = F.kl_div(student_log_probability, teacher_probability, reduction="none").sum(1, keepdim=True)
    return (kl * reliability).sum() / (reliability.sum() + 1e-6)

def endpoint_loss(student_logits, teacher_probability, reliability, radius=6, has_null=True):
    student_dx, student_dy, _, _ = expected_offset(student_logits, radius, has_null)
    teacher_logit = torch.log(teacher_probability.clamp_min(1e-8))
    teacher_dx, teacher_dy, teacher_null, _ = expected_offset(teacher_logit, radius, has_null)
    if reliability.dim() == 3:
        reliability = reliability.unsqueeze(1)
    valid = reliability * (1.0 - teacher_null if has_null else 1.0)
    endpoint = torch.sqrt(
        (student_dx - teacher_dx).square() + (student_dy - teacher_dy).square() + 1e-6
    )
    return (endpoint * valid).sum() / (valid.sum() + 1e-6)

def _warp_field(field, dx, dy):
    batch, _, height, width = field.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=field.device),
        torch.arange(width, device=field.device),
        indexing="ij",
    )
    xx = xx.float().view(1, height, width).expand(batch, -1, -1) + dx[:, 0]
    yy = yy.float().view(1, height, width).expand(batch, -1, -1) + dy[:, 0]
    gx = 2.0 * xx / max(1, width - 1) - 1.0
    gy = 2.0 * yy / max(1, height - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

def relation_cycle_loss(coarse12, coarse21, radius=6):
    dx12, dy12, null12, _ = expected_offset(coarse12, radius, True)
    dx21, dy21, null21, _ = expected_offset(coarse21, radius, True)
    reverse = torch.cat([dx21, dy21], dim=1)
    reverse_at_target = _warp_field(reverse, dx12, dy12)
    cycle = torch.sqrt(
        (dx12 + reverse_at_target[:, 0:1]).square()
        + (dy12 + reverse_at_target[:, 1:2]).square()
        + 1e-6
    )
    valid = (1.0 - null12) * (1.0 - _warp_field(null21, dx12, dy12))
    return (cycle * valid).sum() / (valid.sum() + 1e-6)

def rtec_typed_loss(output, target):
    base_probability = output["base_prob"].detach()
    size = output["RFN"].shape[-2:]
    target_small = F.interpolate(target, size=size, mode="nearest")
    base_small = F.interpolate(base_probability, size=size, mode="bilinear", align_corners=False)
    false_negative_target = target_small * (1.0 - base_small)
    false_positive_target = (1.0 - target_small) * base_small
    type_loss = F.binary_cross_entropy(output["RFN"], false_negative_target)
    type_loss += F.binary_cross_entropy(output["RFP"], false_positive_target)
    exclusion_loss = (output["RFN"] * output["RFP"]).mean()
    magnitude_loss = output["delta_logit"].abs().mean()
    final_loss = bce_dice_from_logit(output["final_logit"], target)
    return type_loss + final_loss + 0.05 * exclusion_loss + 0.01 * magnitude_loss
