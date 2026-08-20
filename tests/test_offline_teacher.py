import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from pairguard.offline_teacher.cache_builder import OfflineTeacher
from pairguard.offline_teacher.sam2_adapter import SAM2ProposalAdapter
from pairguard.offline_teacher.narp_us import UtilitySelector

class DeterministicPredictor:
    def set_image(self, image):
        self.image = image

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
        height, width = self.image.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        masks = np.zeros((3, height, width), dtype=np.uint8)
        masks[0, y1:y2, x1:x2] = 1
        margin = 2
        masks[1, max(0, y1 - margin):min(height, y2 + margin), max(0, x1 - margin):min(width, x2 + margin)] = 1
        masks[2] = masks[0]
        scores = np.asarray([0.95, 0.85, 0.75], dtype=np.float32)
        return masks, scores, None

class FeatureExtractor:
    def __call__(self, image):
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        tensor = F.avg_pool2d(tensor, 8, 8)
        return F.normalize(tensor, dim=1)

def accepted_selector():
    selector = UtilitySelector()
    for parameter in selector.parameters():
        torch.nn.init.zeros_(parameter)
    selector.relative_utility.bias.data.fill_(0.20)
    selector.usability.bias.data.fill_(2.00)
    return selector.eval()

def test_offline_teacher_cache():
    size = 128
    img1 = np.zeros((size, size, 3), dtype=np.uint8)
    img2 = img1.copy()
    img1[32:64, 30:60] = 180
    img2[34:66, 32:62] = 180
    img2[78:105, 80:110] = 220

    seed12 = np.zeros((size, size), dtype=np.float32)
    seed21 = np.zeros_like(seed12)
    seed12[75:108, 77:113] = 0.95
    seed21[75:108, 77:113] = 0.95

    teacher = OfflineTeacher(
        FeatureExtractor(),
        SAM2ProposalAdapter(DeterministicPredictor(), max_proposals=3),
        accepted_selector(),
    )
    cache, metadata = teacher.build(img1, img2, seed12, seed21)

    assert cache["Ye"].shape == (1, size, size)
    assert cache["Vpl"].shape == (1, size, size)
    assert cache["coarse_12"].shape[0] == 170
    assert cache["coarse_21"].shape[0] == 170
    assert cache["fine_12"].shape[0] == 25
    assert cache["fine_21"].shape[0] == 25
    assert "rT_coarse_12" in cache
    assert "rT_fine_12" in cache
    assert metadata["num_hypotheses"] >= 0

if __name__ == "__main__":
    test_offline_teacher_cache()
