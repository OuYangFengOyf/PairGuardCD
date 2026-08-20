from __future__ import annotations
from typing import Optional, Sequence
import numpy as np

class SAM2ProposalAdapter:
    def __init__(self, predictor, max_proposals=3):
        if predictor is None:
            raise ValueError("A SAM2ImagePredictor instance is required.")
        self.predictor = predictor
        self.max_proposals = int(max_proposals)

    @classmethod
    def from_checkpoint(cls, model_cfg, checkpoint, device="cuda", max_proposals=3):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError("Install the official SAM2 package before constructing the offline teacher.") from exc
        model = build_sam2(model_cfg, checkpoint, device=device)
        predictor = SAM2ImagePredictor(model)
        return cls(predictor, max_proposals=max_proposals)

    def propose(self, image_rgb, box, positive_point=None, negative_points: Optional[Sequence]=None):
        self.predictor.set_image(image_rgb)
        coords = []
        labels = []
        if positive_point is not None:
            coords.append(positive_point)
            labels.append(1)
        for point in negative_points or []:
            coords.append(point)
            labels.append(0)
        point_coords = np.asarray(coords, dtype=np.float32) if coords else None
        point_labels = np.asarray(labels, dtype=np.int32) if labels else None
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=np.asarray(box, dtype=np.float32),
            multimask_output=True,
        )
        order = np.argsort(-np.asarray(scores))
        proposals = []
        for idx in order[: self.max_proposals]:
            mask = (masks[idx] > 0).astype(np.uint8)
            if mask.any():
                proposals.append(mask)
        return proposals
