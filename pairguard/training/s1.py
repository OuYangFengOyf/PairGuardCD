import torch
import torch.nn.functional as F
from pairguard.losses import (
    bce_dice_from_logit,
    masked_kl,
    relation_cycle_loss,
    endpoint_loss,
)
from pairguard.models.relation import offset_table
from .common import move_batch

def _target(batch, key, size=None):
    tensor = batch[key]
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    if size is not None and tensor.shape[-2:] != size:
        tensor = F.interpolate(tensor, size=size, mode="nearest")
    return tensor

def _recenter_fine_target(teacher_fine, teacher_coarse, student_coarse, fine_radius=2, coarse_radius=6):
    batch, channels, height, width = teacher_fine.shape
    coarse_classes = teacher_coarse.shape[1] - 1
    teacher_index = teacher_coarse[:, :coarse_classes].argmax(dim=1)
    student_index = student_coarse[:, :coarse_classes].argmax(dim=1)
    student_full_index = student_coarse.argmax(dim=1)
    student_nonnull = student_full_index != coarse_classes

    coarse_offsets = offset_table(coarse_radius, teacher_fine.device, torch.long)
    teacher_dx = coarse_offsets[teacher_index, 0]
    teacher_dy = coarse_offsets[teacher_index, 1]
    student_dx = coarse_offsets[student_index, 0]
    student_dy = coarse_offsets[student_index, 1]

    shift_x = 2 * (teacher_dx - student_dx)
    shift_y = 2 * (teacher_dy - student_dy)
    center_valid = student_nonnull & (shift_x.abs() <= fine_radius) & (shift_y.abs() <= fine_radius)

    fine_size = teacher_fine.shape[-2:]
    shift_x = F.interpolate(shift_x.unsqueeze(1).float(), size=fine_size, mode="nearest").squeeze(1).long()
    shift_y = F.interpolate(shift_y.unsqueeze(1).float(), size=fine_size, mode="nearest").squeeze(1).long()
    center_valid = F.interpolate(center_valid.unsqueeze(1).float(), size=fine_size, mode="nearest").squeeze(1).bool()

    fine_offsets = offset_table(fine_radius, teacher_fine.device, torch.long)
    width_classes = 2 * fine_radius + 1
    output = torch.zeros_like(teacher_fine)

    for out_index, (rdx, rdy) in enumerate(fine_offsets.tolist()):
        source_dx = rdx - shift_x
        source_dy = rdy - shift_y
        source_valid = (
            (source_dx >= -fine_radius)
            & (source_dx <= fine_radius)
            & (source_dy >= -fine_radius)
            & (source_dy <= fine_radius)
            & center_valid
        )
        source_index = (source_dy + fine_radius) * width_classes + (source_dx + fine_radius)
        source_index = source_index.clamp(0, channels - 1)
        gathered = torch.gather(
            teacher_fine,
            1,
            source_index.unsqueeze(1),
        ).squeeze(1)
        output[:, out_index] = torch.where(
            source_valid,
            gathered,
            teacher_fine[:, out_index],
        )

    output = output / (output.sum(dim=1, keepdim=True) + 1e-6)
    return output

def train_s1_epoch(
    model,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    device,
    epoch=0,
    total_epochs=100,
    lambda_pl=1.0,
    lambda_rel=0.5,
    lambda_f=1.0,
    lambda_cyc=0.1,
    lambda_disp=0.1,
):
    model.train()
    total = 0.0
    unlabeled_iterator = iter(unlabeled_loader)

    for labeled_batch in labeled_loader:
        try:
            unlabeled_batch = next(unlabeled_iterator)
        except StopIteration:
            unlabeled_iterator = iter(unlabeled_loader)
            unlabeled_batch = next(unlabeled_iterator)

        labeled_batch = move_batch(labeled_batch, device)
        unlabeled_batch = move_batch(unlabeled_batch, device)

        labeled_output = model(
            labeled_batch["img1"],
            labeled_batch["img2"],
            use_rtec=False,
        )
        unlabeled_output = model(
            unlabeled_batch["img1"],
            unlabeled_batch["img2"],
            use_rtec=False,
        )

        supervised_loss = bce_dice_from_logit(
            labeled_output["base_logit"],
            labeled_batch["label"],
        )

        pseudo_label = _target(
            unlabeled_batch,
            "Ye",
            unlabeled_output["base_logit"].shape[-2:],
        )
        pseudo_valid = _target(
            unlabeled_batch,
            "Vpl",
            unlabeled_output["base_logit"].shape[-2:],
        )
        pseudo_loss = bce_dice_from_logit(
            unlabeled_output["base_logit"],
            pseudo_label,
            pseudo_valid,
        )

        coarse12 = _target(
            unlabeled_batch,
            "coarse_12",
            unlabeled_output["coarse_12"].shape[-2:],
        )
        coarse21 = _target(
            unlabeled_batch,
            "coarse_21",
            unlabeled_output["coarse_21"].shape[-2:],
        )
        fine12 = _target(
            unlabeled_batch,
            "fine_12",
            unlabeled_output["fine_12"].shape[-2:],
        )
        fine21 = _target(
            unlabeled_batch,
            "fine_21",
            unlabeled_output["fine_21"].shape[-2:],
        )
        coarse_r12 = _target(
            unlabeled_batch,
            "rT_coarse_12",
            unlabeled_output["coarse_12"].shape[-2:],
        )
        coarse_r21 = _target(
            unlabeled_batch,
            "rT_coarse_21",
            unlabeled_output["coarse_21"].shape[-2:],
        )
        fine_r12 = _target(
            unlabeled_batch,
            "rT_fine_12",
            unlabeled_output["fine_12"].shape[-2:],
        )
        fine_r21 = _target(
            unlabeled_batch,
            "rT_fine_21",
            unlabeled_output["fine_21"].shape[-2:],
        )

        if epoch >= total_epochs // 2:
            fine12 = _recenter_fine_target(
                fine12,
                coarse12,
                unlabeled_output["coarse_12"].detach(),
            )
            fine21 = _recenter_fine_target(
                fine21,
                coarse21,
                unlabeled_output["coarse_21"].detach(),
            )

        coarse_loss = masked_kl(
            unlabeled_output["coarse_12"],
            coarse12,
            coarse_r12,
        )
        coarse_loss += masked_kl(
            unlabeled_output["coarse_21"],
            coarse21,
            coarse_r21,
        )

        fine_loss = masked_kl(
            unlabeled_output["fine_12"],
            fine12,
            fine_r12,
        )
        fine_loss += masked_kl(
            unlabeled_output["fine_21"],
            fine21,
            fine_r21,
        )

        displacement_loss = endpoint_loss(
            unlabeled_output["coarse_12"],
            coarse12,
            coarse_r12,
            6,
            True,
        )
        displacement_loss += endpoint_loss(
            unlabeled_output["coarse_21"],
            coarse21,
            coarse_r21,
            6,
            True,
        )

        cycle_loss = relation_cycle_loss(
            unlabeled_output["coarse_12"],
            unlabeled_output["coarse_21"],
            6,
        )

        relation_loss = (
            coarse_loss
            + lambda_f * fine_loss
            + lambda_disp * displacement_loss
            + lambda_cyc * cycle_loss
        )
        loss = supervised_loss + lambda_pl * pseudo_loss + lambda_rel * relation_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item())

    return total / max(1, len(labeled_loader))
