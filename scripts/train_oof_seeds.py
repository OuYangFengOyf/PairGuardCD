import sys
from pathlib import Path as _PGPath
_PG_ROOT = _PGPath(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
from pairguard.datasets import PairListDataset, IMAGENET_MEAN, IMAGENET_STD
from pairguard.models.seed_cd import SeedCDNet
from pairguard.losses import seed_recall_loss
from pairguard.offline_teacher.oof_seed import make_group_folds

def train_model(dataset, indices, device, epochs=40, batch_size=8, lr=3e-4):
    model = SeedCDNet(pretrained_backbone=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    for _ in range(epochs):
        model.train()
        for batch in loader:
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)
            label = batch["label"].to(device)
            logit = model(img1, img2)
            loss = seed_recall_loss(logit, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()

def load_image(path):
    tensor = TF.to_tensor(Image.open(path).convert("RGB"))
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor.unsqueeze(0)

@torch.no_grad()
def predict_full(model, path1, path2, device, swap=False, crop_size=256, stride=128):
    first_path, second_path = (path2, path1) if swap else (path1, path2)
    img1 = load_image(first_path).to(device)
    img2 = load_image(second_path).to(device)
    height, width = img1.shape[-2:]

    if height <= crop_size and width <= crop_size:
        pad_h = max(0, crop_size - height)
        pad_w = max(0, crop_size - width)
        patch1 = F.pad(img1, (0, pad_w, 0, pad_h))
        patch2 = F.pad(img2, (0, pad_w, 0, pad_h))
        probability = torch.sigmoid(model(patch1, patch2))[..., :height, :width]
        return probability[0, 0].cpu().numpy().astype(np.float32)

    prediction = torch.zeros((1, 1, height, width), device=device)
    count = torch.zeros_like(prediction)
    ys = list(range(0, max(1, height - crop_size + 1), stride))
    xs = list(range(0, max(1, width - crop_size + 1), stride))
    if ys[-1] != max(0, height - crop_size):
        ys.append(max(0, height - crop_size))
    if xs[-1] != max(0, width - crop_size):
        xs.append(max(0, width - crop_size))

    for y in ys:
        for x in xs:
            patch1 = img1[..., y : y + crop_size, x : x + crop_size]
            patch2 = img2[..., y : y + crop_size, x : x + crop_size]
            patch_h, patch_w = patch1.shape[-2:]
            if patch_h < crop_size or patch_w < crop_size:
                pad = (0, crop_size - patch_w, 0, crop_size - patch_h)
                patch1 = F.pad(patch1, pad)
                patch2 = F.pad(patch2, pad)
            probability = torch.sigmoid(model(patch1, patch2))[..., :patch_h, :patch_w]
            prediction[..., y : y + patch_h, x : x + patch_w] += probability
            count[..., y : y + patch_h, x : x + patch_w] += 1

    return (prediction / count.clamp_min(1))[0, 0].cpu().numpy().astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-manifest", required=True)
    parser.add_argument("--unlabeled-manifest")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PairListDataset(
        args.labeled_manifest,
        size=args.crop_size,
        augment="full",
        crop_training=True,
    )
    folds = make_group_folds(train_dataset.group_ids, args.folds, args.seed)
    unlabeled_dataset = (
        PairListDataset(args.unlabeled_manifest, size=args.crop_size, augment="none")
        if args.unlabeled_manifest
        else None
    )
    accumulated12 = [[] for _ in range(len(unlabeled_dataset))] if unlabeled_dataset is not None else None
    accumulated21 = [[] for _ in range(len(unlabeled_dataset))] if unlabeled_dataset is not None else None
    all_indices = set(range(len(train_dataset)))

    for fold_id, held_indices in enumerate(folds):
        train_indices = sorted(all_indices - set(held_indices))
        model = train_model(
            train_dataset,
            train_indices,
            device,
            args.epochs,
            args.batch_size,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "heldout_indices": held_indices,
                "heldout_groups": sorted({train_dataset.group_ids[i] for i in held_indices}),
            },
            out_dir / f"fold{fold_id}.pt",
        )

        for index in held_indices:
            path1, path2, _, _, _ = train_dataset.records[index]
            np.save(
                out_dir / f"labeled_{index:06d}_seed12.npy",
                predict_full(model, path1, path2, device, False, args.crop_size, args.stride),
            )
            np.save(
                out_dir / f"labeled_{index:06d}_seed21.npy",
                predict_full(model, path1, path2, device, True, args.crop_size, args.stride),
            )

        if unlabeled_dataset is not None:
            for index, record in enumerate(unlabeled_dataset.records):
                path1, path2, _, _, _ = record
                accumulated12[index].append(
                    predict_full(model, path1, path2, device, False, args.crop_size, args.stride)
                )
                accumulated21[index].append(
                    predict_full(model, path1, path2, device, True, args.crop_size, args.stride)
                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if unlabeled_dataset is not None:
        for index in range(len(unlabeled_dataset)):
            np.save(
                out_dir / f"unlabeled_{index:06d}_seed12.npy",
                np.mean(accumulated12[index], axis=0).astype(np.float32),
            )
            np.save(
                out_dir / f"unlabeled_{index:06d}_seed21.npy",
                np.mean(accumulated21[index], axis=0).astype(np.float32),
            )

    print(f"OOF seed generation completed: {out_dir}")

if __name__ == "__main__":
    main()
