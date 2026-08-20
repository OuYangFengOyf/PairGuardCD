import sys
from pathlib import Path as _PGPath
_PG_ROOT = _PGPath(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from pairguard.offline_teacher.sam2_adapter import SAM2ProposalAdapter
from pairguard.offline_teacher.dinov2_adapter import DINOv2FeatureExtractor
from pairguard.offline_teacher.c2f_scpt import C2FSCPT
from pairguard.offline_teacher.correspondence import local_correspondence_evidence
from pairguard.offline_teacher.proposal_builder import ProposalHypothesisBuilder
from pairguard.offline_teacher.narp_us import proposal_features, make_utility_target

def resize_seed(seed, size):
    width, height = size
    if seed.shape == (height, width):
        return seed.astype(np.float32)
    image = Image.fromarray(seed.astype(np.float32), mode="F")
    return np.asarray(image.resize((width, height), resample=Image.Resampling.BILINEAR), dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sam2-config", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
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
    scpt = C2FSCPT()
    proposal_builder = ProposalHypothesisBuilder(sam)

    features = []
    relative_utility = []
    usability = []
    groups = []

    lines = [
        line.strip()
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    for record_index, line in enumerate(lines):
        parts = [item.strip() for item in line.split(",")]
        while len(parts) < 6:
            parts.append("")
        path1, path2, label_path, seed12_path, seed21_path, group_id = parts[:6]
        if not seed12_path or not seed21_path:
            raise ValueError("NARP-US record construction requires pair-level OOF seeds.")

        img1 = np.asarray(Image.open(path1).convert("RGB"))
        img2 = np.asarray(Image.open(path2).convert("RGB"))
        if img1.shape != img2.shape:
            raise ValueError(f"Image size mismatch: {path1} vs {path2}")
        height, width = img1.shape[:2]
        gt = np.asarray(Image.open(label_path).convert("L"), dtype=np.float32) / 255.0
        if gt.shape != (height, width):
            gt = np.asarray(
                Image.fromarray(gt, mode="F").resize((width, height), resample=Image.Resampling.NEAREST),
                dtype=np.float32,
            )

        seed12 = resize_seed(np.load(seed12_path), (width, height))
        seed21 = resize_seed(np.load(seed21_path), (width, height))
        z1 = dino(img1)
        z2 = dino(img2)
        corr12, valid12 = local_correspondence_evidence(z1, z2, seed12.shape)
        corr21, valid21 = local_correspondence_evidence(z2, z1, seed21.shape)
        score12, trans12 = scpt(img1, img2, seed12, corr12, valid12)
        score21, trans21 = scpt(img2, img1, seed21, corr21, valid21)

        hypotheses = proposal_builder.build_direction(img1, img2, seed12, score12, trans12, "12")
        hypotheses += proposal_builder.build_direction(img2, img1, seed21, score21, trans21, "21")

        current_group = group_id or str(record_index)
        for hypothesis in hypotheses:
            if hypothesis.proposal is None:
                continue
            utility, usable = make_utility_target(hypothesis.proposal, seed12, gt)
            features.append(proposal_features(hypothesis))
            relative_utility.append(utility)
            usability.append(usable)
            groups.append(current_group)

        print(record_index, len(hypotheses))

    if not features:
        raise RuntimeError("No NARP-US proposal records were generated.")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(features),
        relative_utility=np.asarray(relative_utility, dtype=np.float32),
        usable=np.asarray(usability, dtype=np.float32),
        groups=np.asarray(groups, dtype=object),
    )
    print(f"NARP-US records saved: {output}")

if __name__ == "__main__":
    main()
