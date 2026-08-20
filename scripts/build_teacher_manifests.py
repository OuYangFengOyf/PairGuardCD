import sys
from pathlib import Path as _PGPath
_PG_ROOT = _PGPath(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

from pathlib import Path
import argparse

def read_student_manifest(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [item.strip() for item in line.split(",")]
        while len(parts) < 5:
            parts.append("")
        records.append(parts[:5])
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-manifest", required=True)
    parser.add_argument("--unlabeled-manifest", required=True)
    parser.add_argument("--seed-dir", required=True)
    parser.add_argument("--narp-out", required=True)
    parser.add_argument("--teacher-out", required=True)
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    labeled = read_student_manifest(args.labeled_manifest)
    unlabeled = read_student_manifest(args.unlabeled_manifest)

    narp_lines = []
    for index, (img1, img2, label, _, group_id) in enumerate(labeled):
        seed12 = seed_dir / f"labeled_{index:06d}_seed12.npy"
        seed21 = seed_dir / f"labeled_{index:06d}_seed21.npy"
        narp_lines.append(f"{img1},{img2},{label},{seed12},{seed21},{group_id or index}")

    teacher_lines = []
    for index, (img1, img2, _, _, group_id) in enumerate(unlabeled):
        seed12 = seed_dir / f"unlabeled_{index:06d}_seed12.npy"
        seed21 = seed_dir / f"unlabeled_{index:06d}_seed21.npy"
        teacher_lines.append(f"{img1},{img2},{seed12},{seed21},{group_id or index}")

    narp_path = Path(args.narp_out)
    teacher_path = Path(args.teacher_out)
    narp_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_path.parent.mkdir(parents=True, exist_ok=True)
    narp_path.write_text("\n".join(narp_lines) + "\n", encoding="utf-8")
    teacher_path.write_text("\n".join(teacher_lines) + "\n", encoding="utf-8")
    print(f"NARP-US manifest: {narp_path}")
    print(f"Teacher cache manifest: {teacher_path}")

if __name__ == "__main__":
    main()
