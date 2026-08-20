import sys
import math
import argparse
from pathlib import Path as _PGPath
_PG_ROOT = _PGPath(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

from pathlib import Path
import torch
from pairguard.datasets import PairListDataset
from pairguard.models import PairGuardCD
from pairguard.metrics import BinaryChangeMeter
from pairguard.training.common import make_loader, save_checkpoint, load_checkpoint, move_batch
from pairguard.training.s0 import train_s0_epoch
from pairguard.training.s1 import train_s1_epoch
from pairguard.training.s2 import train_s2_epoch

def build_optimizer(model, stage, backbone_lr, head_lr):
    for parameter in model.parameters():
        parameter.requires_grad = True

    if stage == "s0":
        for parameter in model.relation.parameters():
            parameter.requires_grad = False
        for parameter in model.rtec.parameters():
            parameter.requires_grad = False
    elif stage == "s1":
        for parameter in model.rtec.parameters():
            parameter.requires_grad = False
    elif stage == "s2":
        model.freeze_except_rtec()

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    backbone = [
        parameter
        for name, parameter in trainable
        if name.startswith("backbone.")
    ]
    heads = [
        parameter
        for name, parameter in trainable
        if not name.startswith("backbone.")
    ]

    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    if heads:
        groups.append({"params": heads, "lr": head_lr})
    return torch.optim.AdamW(groups, weight_decay=1e-4)

def lr_factor(epoch, total, warmup):
    if warmup > 0 and epoch < warmup:
        return max(1e-3, (epoch + 1) / warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

@torch.no_grad()
def evaluate(model, loader, device, stage):
    model.eval()
    meter = BinaryChangeMeter()
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(
            batch["img1"],
            batch["img2"],
            use_rtec=stage == "s2",
            use_relation=stage != "s0",
        )
        meter.update(output["final_prob"], batch["label"])
    return meter.compute()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["s0", "s1", "s2"], required=True)
    parser.add_argument("--labeled-manifest", required=True)
    parser.add_argument("--unlabeled-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--resume")
    parser.add_argument("--save", default="checkpoints/pairguard.pt")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PairGuardCD(pretrained_backbone=(args.pretrained_backbone and not args.resume)).to(device)
    if args.resume:
        load_checkpoint(args.resume, model, map_location=device)

    labeled_loader = make_loader(
        PairListDataset(args.labeled_manifest, augment="full"),
        args.batch_size,
        True,
    )

    unlabeled_loader = None
    if args.stage == "s1":
        if not args.unlabeled_manifest:
            raise ValueError("--unlabeled-manifest is required for Stage S1.")
        unlabeled_loader = make_loader(
            PairListDataset(args.unlabeled_manifest, augment="photometric"),
            args.batch_size,
            True,
        )

    val_loader = None
    if args.val_manifest:
        val_loader = make_loader(
            PairListDataset(args.val_manifest, augment="none"),
            args.batch_size,
            False,
        )

    optimizer = build_optimizer(model, args.stage, args.backbone_lr, args.head_lr)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    best_f1 = -1.0
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        factor = lr_factor(epoch, args.epochs, args.warmup_epochs)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = max(1e-6, base_lr * factor)

        if args.stage == "s0":
            loss = train_s0_epoch(model, labeled_loader, optimizer, device)
        elif args.stage == "s1":
            loss = train_s1_epoch(model, labeled_loader, unlabeled_loader, optimizer, device, epoch, args.epochs)
        else:
            loss = train_s2_epoch(model, labeled_loader, optimizer, device)

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, args.stage)
            current_f1 = metrics["F1"]
            if current_f1 > best_f1:
                best_f1 = current_f1
                save_checkpoint(
                    save_path,
                    model,
                    optimizer,
                    {"stage": args.stage, "epoch": epoch + 1, "metrics": metrics},
                )
            print(
                f"epoch={epoch + 1}/{args.epochs} loss={loss:.6f} "
                f"F1={metrics['F1']:.4f} IoU={metrics['IoU']:.4f}"
            )
        else:
            save_checkpoint(
                save_path,
                model,
                optimizer,
                {"stage": args.stage, "epoch": epoch + 1},
            )
            print(f"epoch={epoch + 1}/{args.epochs} loss={loss:.6f}")

    print(f"Checkpoint saved: {save_path}")

if __name__ == "__main__":
    main()
