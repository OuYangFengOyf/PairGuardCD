import sys
import time
import argparse
from pathlib import Path as _PGPath
_PG_ROOT = _PGPath(__file__).resolve().parents[1]
if str(_PG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PG_ROOT))

import torch
from pairguard.models import PairGuardCD
from pairguard.training.common import load_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PairGuardCD(False).to(device).eval()
    img1 = torch.randn(1, 3, args.size, args.size, device=device)
    img2 = torch.randn_like(img1)

    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)

    params = sum(parameter.numel() for parameter in model.parameters()) / 1e6

    with torch.no_grad():
        for _ in range(args.warmup):
            model(img1, img2)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            model(img1, img2)
        if device.type == "cuda":
            torch.cuda.synchronize()

    latency = (time.perf_counter() - start) / args.iters
    print(f"Params(M): {params:.3f}")
    print(f"Latency(ms): {latency * 1000:.3f}")
    print(f"FPS: {1.0 / latency:.2f}")

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return

    flops = FlopCountAnalysis(model, (img1, img2)).total() / 1e9
    print(f"FLOPs(G): {flops:.3f}")

if __name__ == "__main__":
    main()
