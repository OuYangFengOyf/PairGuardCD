import sys
from pathlib import Path as _PGPath
_PG_ROOT = _PGPath(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

from pathlib import Path
import argparse
import numpy as np
import cv2
from PIL import Image

def warp(image, kind, magnitude, rng):
    height, width = image.shape[:2]

    if kind == "translation":
        dx = rng.uniform(-magnitude, magnitude)
        dy = rng.uniform(-magnitude, magnitude)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        output = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = cv2.warpAffine(
            np.ones((height, width), dtype=np.uint8),
            matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
        )

    elif kind == "rotation":
        angle = rng.uniform(-magnitude, magnitude)
        matrix = cv2.getRotationMatrix2D(
            (width / 2, height / 2),
            angle,
            1.0,
        )
        output = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = cv2.warpAffine(
            np.ones((height, width), dtype=np.uint8),
            matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
        )

    elif kind == "homography":
        displacement = float(magnitude)
        source = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        target = source + rng.uniform(
            -displacement,
            displacement,
            source.shape,
        ).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(source, target)
        output = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = cv2.warpPerspective(
            np.ones((height, width), dtype=np.uint8),
            matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
        )

    elif kind == "local":
        yy, xx = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing="ij",
        )
        dx = rng.normal(0, 1, (height, width)).astype(np.float32)
        dy = rng.normal(0, 1, (height, width)).astype(np.float32)
        dx = cv2.GaussianBlur(dx, (0, 0), 20)
        dy = cv2.GaussianBlur(dy, (0, 0), 20)
        dx = dx / (np.std(dx) + 1e-6) * magnitude / 3.0
        dy = dy / (np.std(dy) + 1e-6) * magnitude / 3.0
        map_x = (xx + dx).astype(np.float32)
        map_y = (yy + dy).astype(np.float32)
        output = cv2.remap(
            image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = cv2.remap(
            np.ones((height, width), dtype=np.uint8),
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )

    else:
        raise ValueError(kind)

    return output, (valid > 0).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--kind",
        choices=["translation", "rotation", "homography", "local"],
        default="translation",
    )
    parser.add_argument("--magnitude", type=float, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(Image.open(args.image).convert("RGB"))
    warped, valid = warp(
        image,
        args.kind,
        args.magnitude,
        np.random.default_rng(args.seed),
    )
    Image.fromarray(warped).save(output_dir / "warped.png")
    Image.fromarray(valid * 255).save(output_dir / "valid_mask.png")
    print(output_dir)

if __name__ == "__main__":
    main()
