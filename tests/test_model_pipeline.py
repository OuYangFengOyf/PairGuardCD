import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from pairguard.models import PairGuardCD

def test_pairguard_forward():
    torch.manual_seed(0)
    model = PairGuardCD(False)
    img1 = torch.rand(1, 3, 128, 128)
    img2 = torch.rand(1, 3, 128, 128)

    output = model(img1, img2, use_rtec=True, use_relation=True)
    assert output["final_prob"].shape == (1, 1, 128, 128)
    assert output["coarse_12"].shape[1] == 170
    assert output["coarse_21"].shape[1] == 170
    assert output["fine_12"].shape[1] == 25
    assert output["fine_21"].shape[1] == 25
    assert output["ZR"].shape[1] == 24

    base_output = model(img1, img2, use_rtec=False, use_relation=False)
    assert base_output["base_prob"].shape == (1, 1, 128, 128)
    assert base_output["ZR"].shape[1] == 24

if __name__ == "__main__":
    test_pairguard_forward()
