import sys
from pathlib import Path as _PGPath
_PG_ROOT = _PGPath(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

import argparse
import numpy as np
import torch
from pairguard.offline_teacher.narp_us import cross_fit_selectors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = np.load(args.records, allow_pickle=True)
    models, oof_rel, oof_use, group_folds = cross_fit_selectors(
        data["features"],
        data["relative_utility"],
        data["usable"],
        data["groups"],
        args.folds,
        args.seed,
        args.device,
    )
    checkpoint = {
        "selectors": [model.state_dict() for model in models],
        "oof_rel": oof_rel,
        "oof_use": oof_use,
        "group_folds": [np.asarray(fold, dtype=object) for fold in group_folds],
        "input_dim": 18,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    print(f"NARP-US selector checkpoint saved: {args.out}")

if __name__ == "__main__":
    main()
