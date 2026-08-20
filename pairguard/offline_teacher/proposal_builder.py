from __future__ import annotations
import numpy as np
import cv2
from .narp_us import Hypothesis, RelationState
from .geometry import warp_mask

def _boundary(mask):
    return cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )

def _overlap(a, b):
    numerator = ((a > 0) & (b > 0)).sum()
    denominator = (a > 0).sum()
    return float(numerator / (denominator + 1e-6))

def _search_completeness(box, width, height, reliability):
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    inside = (
        max(0, min(width, x2) - max(0, x1))
        * max(0, min(height, y2) - max(0, y1))
    )
    return float(np.clip(reliability * inside / area, 0.0, 1.0))

def _best_source_proposal(proposals, seed):
    if not proposals:
        return None
    return max(
        proposals,
        key=lambda mask: _overlap(mask, seed > 0.25),
    )

def _best_counterpart(proposals, source_mask, matrix, shape):
    best_mask = None
    best_back = None
    best_evidence = 0.0
    for proposal in proposals:
        mapped_back = warp_mask(
            proposal,
            matrix,
            shape,
            inverse=True,
        )
        evidence = _overlap(mapped_back, source_mask)
        if evidence > best_evidence:
            best_mask = proposal
            best_back = mapped_back
            best_evidence = evidence
    return best_mask, best_back, float(best_evidence)

class ProposalHypothesisBuilder:
    def __init__(self, sam_adapter, counterpart_threshold=0.15):
        if sam_adapter is None:
            raise ValueError("A SAM2 proposal adapter is required.")
        self.sam = sam_adapter
        self.counterpart_threshold = float(counterpart_threshold)

    def build_direction(
        self,
        src,
        tgt,
        seed,
        score,
        transports,
        direction,
    ):
        height, width = seed.shape
        hypotheses = []

        for transport in transports:
            source_proposals = self.sam.propose(
                src,
                transport.box_src,
                transport.pos_src,
                transport.neg_src,
            )
            target_proposals = self.sam.propose(
                tgt,
                transport.box_tgt,
                transport.pos_tgt,
                transport.neg_tgt,
            )
            source_mask = _best_source_proposal(
                source_proposals,
                seed,
            )
            if source_mask is None:
                continue

            search_reliability = _search_completeness(
                transport.box_tgt,
                width,
                height,
                transport.reliability,
            )
            target_mask, target_back, counterpart_evidence = _best_counterpart(
                target_proposals,
                source_mask,
                transport.matrix,
                (height, width),
            )

            if target_mask is not None and counterpart_evidence >= self.counterpart_threshold:
                disagreement = np.abs(
                    source_mask.astype(np.float32)
                    - target_back.astype(np.float32)
                )
                proposal = np.clip(
                    0.70 * disagreement
                    + 0.30 * score * source_mask,
                    0.0,
                    1.0,
                )
                state = RelationState.OBJECT_OBJECT
                bidirectional_consistency = counterpart_evidence
                missing_ratio = 0.0
            else:
                target_mask = None
                proposal = np.clip(
                    source_mask.astype(np.float32) * score,
                    0.0,
                    1.0,
                )
                state = (
                    RelationState.OBJECT_NULL
                    if direction == "12"
                    else RelationState.NULL_OBJECT
                )
                bidirectional_consistency = 0.0
                missing_ratio = 0.5

            boundary_consistency = _overlap(
                _boundary(source_mask),
                _boundary(seed > 0.25),
            )

            if direction == "21":
                proposal = warp_mask(
                    proposal,
                    transport.matrix,
                    (height, width),
                    inverse=False,
                )

            hypotheses.append(
                Hypothesis(
                    mask1=source_mask if direction == "12" else target_mask,
                    mask2=target_mask if direction == "12" else source_mask,
                    state=state,
                    q_src=float(
                        np.clip(
                            _overlap(source_mask, seed > 0.25),
                            0.2,
                            1.0,
                        )
                    ),
                    search_reliability=search_reliability,
                    counterpart_evidence=counterpart_evidence,
                    transform_reliability=transport.reliability,
                    seed_overlap=_overlap(
                        source_mask,
                        seed > 0.25,
                    ),
                    aligned_difference=float(
                        score[source_mask > 0].mean()
                    )
                    if (source_mask > 0).any()
                    else 0.0,
                    bidirectional_consistency=bidirectional_consistency,
                    boundary_consistency=boundary_consistency,
                    missing_ratio=missing_ratio,
                    proposal_scale=float(source_mask.mean()),
                    proposal=proposal,
                )
            )

        return hypotheses
