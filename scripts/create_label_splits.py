import sys
from pathlib import Path as _PGPath
_PG_ROOT = _PGPath(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

from pathlib import Path
import argparse
import numpy as np

def read_manifest(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [item.strip() for item in line.split(",")]
        while len(parts) < 5:
            parts.append("")
        if not parts[2]:
            raise ValueError("The source manifest must include ground-truth labels.")
        records.append(parts[:5])
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--labeled-out", required=True)
    parser.add_argument("--unlabeled-out", required=True)
    args = parser.parse_args()

    if not 0.0 < args.ratio <= 1.0:
        raise ValueError("--ratio must be in (0, 1].")

    records = read_manifest(args.manifest)
    group_ids = np.asarray([record[4] or str(index) for index, record in enumerate(records)], dtype=object)
    unique_groups = np.unique(group_ids)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_groups)
    labeled_count = max(1, int(round(len(unique_groups) * args.ratio)))
    labeled_groups = set(unique_groups[:labeled_count].tolist())

    labeled_lines = []
    unlabeled_lines = []
    for index, record in enumerate(records):
        group_id = record[4] or str(index)
        img1, img2, label, cache, _ = record
        if group_id in labeled_groups:
            labeled_lines.append(f"{img1},{img2},{label},{cache},{group_id}")
        else:
            unlabeled_lines.append(f"{img1},{img2},,{cache},{group_id}")

    labeled_path = Path(args.labeled_out)
    unlabeled_path = Path(args.unlabeled_out)
    labeled_path.parent.mkdir(parents=True, exist_ok=True)
    unlabeled_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_path.write_text("\n".join(labeled_lines) + "\n", encoding="utf-8")
    unlabeled_path.write_text("\n".join(unlabeled_lines) + "\n", encoding="utf-8")

    print(f"Labeled groups: {len(labeled_groups)}/{len(unique_groups)}")
    print(f"Labeled manifest: {labeled_path}")
    print(f"Unlabeled manifest: {unlabeled_path}")

if __name__ == "__main__":
    main()
