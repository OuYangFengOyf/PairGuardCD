from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2
from .correspondence import normalized_abs_difference
from .geometry import (
    estimate_transform_hierarchy,
    transform_points,
    transform_box,
)

@dataclass
class TransportResult:
    component_mask: np.ndarray
    stable_mask: np.ndarray
    box_src: Tuple[int, int, int, int]
    box_tgt: Tuple[int, int, int, int]
    pos_src: Tuple[int, int]
    pos_tgt: Tuple[int, int]
    neg_src: List[Tuple[int, int]]
    neg_tgt: List[Tuple[int, int]]
    matrix: np.ndarray
    reliability: float
    model: str

def combine_candidate_evidence(
    seed,
    appearance_difference,
    correspondence_difference=None,
    alpha=1.0,
    beta=0.7,
    gamma=0.7,
):
    numerator = alpha * seed + beta * appearance_difference
    denominator = alpha + beta
    if correspondence_difference is not None:
        numerator = numerator + gamma * correspondence_difference
        denominator = denominator + gamma
    return (numerator / (denominator + 1e-6)).astype(np.float32)

def connected_components(score, threshold=0.45, min_area=16):
    binary = (score >= threshold).astype(np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = []
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area:
            components.append((labels == index).astype(np.uint8))
    return components

def stable_ring(
    component,
    seed,
    difference,
    correspondence_valid=None,
    inner=3,
    outer=16,
    max_seed=0.25,
    max_difference=0.30,
):
    kernel_inner = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (inner * 2 + 1, inner * 2 + 1),
    )
    kernel_outer = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (outer * 2 + 1, outer * 2 + 1),
    )
    inner_region = cv2.dilate(component, kernel_inner)
    outer_region = cv2.dilate(component, kernel_outer)
    ring = (outer_region > 0) & (inner_region == 0)
    valid = ring & (seed < max_seed) & (difference < max_difference)
    if correspondence_valid is not None:
        valid &= correspondence_valid.astype(bool)
    return valid.astype(np.uint8)

def _bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 1, 1
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() + 1),
        int(ys.max() + 1),
    )

def _centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0
    return int(round(xs.mean())), int(round(ys.mean()))

def _negative_points(stable_mask, count=4):
    ys, xs = np.where(stable_mask > 0)
    if len(xs) == 0:
        return []
    indices = np.linspace(
        0,
        len(xs) - 1,
        min(count, len(xs)),
        dtype=int,
    )
    return [(int(xs[index]), int(ys[index])) for index in indices]

class C2FSCPT:
    def __init__(
        self,
        threshold=0.45,
        min_area=16,
        alpha=1.0,
        beta=0.7,
        gamma=0.7,
    ):
        self.threshold = float(threshold)
        self.min_area = int(min_area)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def __call__(
        self,
        img1,
        img2,
        seed,
        correspondence_difference=None,
        correspondence_valid=None,
    ):
        appearance_difference = normalized_abs_difference(img1, img2)
        score = combine_candidate_evidence(
            seed,
            appearance_difference,
            correspondence_difference,
            self.alpha,
            self.beta,
            self.gamma,
        )
        components = connected_components(
            score,
            self.threshold,
            self.min_area,
        )
        height, width = seed.shape
        results = []

        for component in components:
            stable = stable_ring(
                component,
                seed,
                appearance_difference,
                correspondence_valid,
            )
            estimate = estimate_transform_hierarchy(
                img1,
                img2,
                stable,
            )
            source_box = _bbox(component)
            cx, cy = _centroid(component)
            target_xy = transform_points(
                np.asarray([[cx, cy]], dtype=np.float32),
                estimate.matrix,
            )[0]
            tx = int(np.clip(round(float(target_xy[0])), 0, width - 1))
            ty = int(np.clip(round(float(target_xy[1])), 0, height - 1))
            uncertainty = max(2.0, (1.0 - estimate.reliability) * 10.0)
            target_box = transform_box(
                source_box,
                estimate.matrix,
                width,
                height,
                pad=uncertainty,
            )
            negative_source = _negative_points(stable, 4)
            negative_target = []
            if negative_source:
                mapped = transform_points(
                    np.asarray(negative_source, dtype=np.float32),
                    estimate.matrix,
                )
                for px, py in mapped:
                    negative_target.append(
                        (
                            int(np.clip(round(px), 0, width - 1)),
                            int(np.clip(round(py), 0, height - 1)),
                        )
                    )

            results.append(
                TransportResult(
                    component,
                    stable,
                    source_box,
                    target_box,
                    (cx, cy),
                    (tx, ty),
                    negative_source,
                    negative_target,
                    estimate.matrix,
                    estimate.reliability,
                    estimate.model,
                )
            )

        return score, results
