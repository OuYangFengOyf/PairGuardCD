import sys
from pathlib import Path as _PGPath
_PG_ROOT = _PGPath(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from pairguard.offline_teacher import (
    OfflineTeacher,
    SAM2ProposalAdapter,
    DINOv2FeatureExtractor,
    load_selector_ensemble,
)

def read_manifest(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [item.strip() for item in line.split(",")]
        while len(parts) < 5:
            parts.append("")
        if not parts[2] or not parts[3]:
            raise ValueError("Teacher manifest requires OOF seed12 and seed21 paths.")
        rows.append(parts[:5])
    return rows

def resize_seed(seed, size):
    width, height = size
    if seed.shape == (height, width):
        return seed.astype(np.float32)
    image = Image.fromarray(seed.astype(np.float32), mode="F")
    return np.asarray(image.resize((width, height), resample=Image.Resampling.BILINEAR), dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--output-manifest")
    parser.add_argument("--sam2-config", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--dino-variant", default="dinov2_vits14")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    sam = SAM2ProposalAdapter.from_checkpoint(
        args.sam2_config,
        args.sam2_checkpoint,
        device=args.device,
        max_proposals=3,
    )
    dino = DINOv2FeatureExtractor.from_torch_hub(args.dino_variant, device=args.device)
    selector = load_selector_ensemble(args.selector_checkpoint)
    teacher = OfflineTeacher(dino, sam, selector)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_lines = []

    for index, (path1, path2, seed12_path, seed21_path, group_id) in enumerate(read_manifest(args.manifest)):
        img1 = np.asarray(Image.open(path1).convert("RGB"))
        img2 = np.asarray(Image.open(path2).convert("RGB"))
        if img1.shape != img2.shape:
            raise ValueError(f"Image size mismatch: {path1} vs {path2}")
        height, width = img1.shape[:2]
        seed12 = resize_seed(np.load(seed12_path), (width, height))
        seed21 = resize_seed(np.load(seed21_path), (width, height))
        cache, metadata = teacher.build(img1, img2, seed12, seed21)
        cache_path = out_dir / f"{index:06d}.npz"
        teacher.save(cache_path, cache)
        output_lines.append(f"{path1},{path2},,{cache_path},{group_id or index}")
        print(index, metadata)

    if args.output_manifest:
        output_path = Path(args.output_manifest)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    print(f"Teacher cache generation completed: {out_dir}")

if __name__ == "__main__":
    main()
