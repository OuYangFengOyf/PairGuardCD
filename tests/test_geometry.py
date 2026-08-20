import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from pairguard.offline_teacher.geometry import transform_points, transform_box

def test_transform_geometry():
    matrix = np.array([[1, 0, 5], [0, 1, -3]], dtype=np.float32)
    point = transform_points(np.array([[10, 10]], dtype=np.float32), matrix)[0]
    assert np.allclose(point, [15, 7])
    box = transform_box((10, 10, 20, 20), matrix, 100, 100)
    assert box == (15, 7, 25, 17)

if __name__ == "__main__":
    test_transform_geometry()
