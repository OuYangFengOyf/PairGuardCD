import sys
from pathlib import Path as _PGPath
_PG_ROOT = _PGPath(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

import argparse
import torch
from pairguard.datasets import PairListDataset
from pairguard.models import PairGuardCD
from pairguard.metrics import BinaryChangeMeter
from pairguard.training.common import make_loader, load_checkpoint, move_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = PairListDataset(args.manifest, augment="none")
    loader = make_loader(dataset, args.batch_size, False)
    model = PairGuardCD().to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    meter = BinaryChangeMeter()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(batch["img1"], batch["img2"], use_rtec=True)
            meter.update(output["final_prob"], batch["label"])

    for name, value in meter.compute().items():
        print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main()
