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
import torchvision.transforms.functional as TF
from pairguard.datasets import IMAGENET_MEAN, IMAGENET_STD
from pairguard.models import PairGuardCD
from pairguard.metrics import BinaryChangeMeter
from pairguard.training.common import load_checkpoint
from pairguard.inference import sliding_window_predict

def load_image(path):
    image = TF.to_tensor(Image.open(path).convert("RGB"))
    return TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0)

def load_mask(path):
    array = (np.asarray(Image.open(path).convert("L")) > 127).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PairGuardCD(False).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    meter = BinaryChangeMeter()

    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [item.strip() for item in line.split(",")]
        img1 = load_image(parts[0]).to(device)
        img2 = load_image(parts[1]).to(device)
        target = load_mask(parts[2]).to(device)
        prediction = sliding_window_predict(
            model,
            img1,
            img2,
            crop_size=args.crop_size,
            stride=args.stride,
            use_rtec=True,
        )
        valid = None
        if len(parts) > 3 and parts[3]:
            valid_array = (np.asarray(Image.open(parts[3]).convert("L")) > 0).astype(np.float32)
            valid = torch.from_numpy(valid_array).unsqueeze(0).unsqueeze(0).to(device)
        meter.update(prediction, target, valid)

    for name, value in meter.compute().items():
        print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main()
