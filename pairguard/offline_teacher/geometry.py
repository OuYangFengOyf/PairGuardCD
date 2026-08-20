from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2

@dataclass
class TransformEstimate:
    matrix: np.ndarray
    reliability: float
    model: str
    inliers: int = 0
    matches: int = 0

    @property
    def dx(self):
        return float(self.matrix[0, 2])

    @property
    def dy(self):
        return float(self.matrix[1, 2])

def identity_transform(reliability=0.0, model="identity"):
    matrix = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    return TransformEstimate(matrix, reliability, model)

def phase_translation(img1, img2, mask: Optional[np.ndarray] = None):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if mask is not None:
        weight = mask.astype(np.float32)
        gray1 = gray1 * weight
        gray2 = gray2 * weight
    try:
        (dx, dy), response = cv2.phaseCorrelate(gray1, gray2)
    except cv2.error:
        return identity_transform(0.0, "phase_unavailable")
    if not np.isfinite(dx) or not np.isfinite(dy):
        return identity_transform(0.0, "phase_unavailable")
    matrix = np.asarray(
        [[1.0, 0.0, dx], [0.0, 1.0, dy]],
        dtype=np.float32,
    )
    return TransformEstimate(
        matrix,
        float(np.clip(response, 0.0, 1.0)),
        "translation_phase",
    )

def _orb_matches(
    img1,
    img2,
    mask1: Optional[np.ndarray],
    mask2: Optional[np.ndarray],
    max_features=1200,
):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(
        nfeatures=max_features,
        fastThreshold=8,
        edgeThreshold=15,
    )
    keypoints1, descriptors1 = orb.detectAndCompute(
        gray1,
        (mask1 * 255).astype(np.uint8) if mask1 is not None else None,
    )
    keypoints2, descriptors2 = orb.detectAndCompute(
        gray2,
        (mask2 * 255).astype(np.uint8) if mask2 is not None else None,
    )
    if (
        descriptors1 is None
        or descriptors2 is None
        or len(keypoints1) < 4
        or len(keypoints2) < 4
    ):
        return None, None, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good.append(first)

    if len(good) < 4:
        return None, None, len(good)

    points1 = np.float32(
        [keypoints1[match.queryIdx].pt for match in good]
    )
    points2 = np.float32(
        [keypoints2[match.trainIdx].pt for match in good]
    )
    return points1, points2, len(good)

def estimate_similarity(
    img1,
    img2,
    mask1: Optional[np.ndarray] = None,
    mask2: Optional[np.ndarray] = None,
):
    points1, points2, matches = _orb_matches(
        img1,
        img2,
        mask1,
        mask2,
    )
    if points1 is None:
        return None

    matrix, inlier = cv2.estimateAffinePartial2D(
        points1,
        points2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.995,
    )
    if matrix is None:
        return None

    inliers = int(inlier.sum()) if inlier is not None else 0
    ratio = inliers / max(1, matches)
    reliability = float(
        np.clip(
            ratio * min(1.0, matches / 20.0),
            0.0,
            1.0,
        )
    )
    return TransformEstimate(
        matrix.astype(np.float32),
        reliability,
        "similarity_orb",
        inliers,
        matches,
    )

def estimate_affine(
    img1,
    img2,
    mask1: Optional[np.ndarray] = None,
    mask2: Optional[np.ndarray] = None,
):
    points1, points2, matches = _orb_matches(
        img1,
        img2,
        mask1,
        mask2,
    )
    if points1 is None or len(points1) < 6:
        return None

    matrix, inlier = cv2.estimateAffine2D(
        points1,
        points2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.995,
    )
    if matrix is None:
        return None

    inliers = int(inlier.sum()) if inlier is not None else 0
    ratio = inliers / max(1, matches)
    reliability = float(
        np.clip(
            ratio * min(1.0, matches / 24.0),
            0.0,
            1.0,
        )
    )
    return TransformEstimate(
        matrix.astype(np.float32),
        reliability,
        "affine_orb",
        inliers,
        matches,
    )

def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((len(points), 1), dtype=np.float32)
    homogeneous = np.concatenate([points, ones], axis=1)
    return homogeneous @ matrix.T

def transform_box(
    box: Tuple[int, int, int, int],
    matrix,
    width,
    height,
    pad=0.0,
):
    x1, y1, x2, y2 = box
    corners = np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    mapped = transform_points(corners, matrix)
    target_x1 = max(0, int(np.floor(mapped[:, 0].min() - pad)))
    target_y1 = max(0, int(np.floor(mapped[:, 1].min() - pad)))
    target_x2 = min(width, int(np.ceil(mapped[:, 0].max() + pad)))
    target_y2 = min(height, int(np.ceil(mapped[:, 1].max() + pad)))
    return (
        target_x1,
        target_y1,
        max(target_x1 + 1, target_x2),
        max(target_y1 + 1, target_y2),
    )

def estimate_transform_hierarchy(
    img1,
    img2,
    stable_mask: Optional[np.ndarray],
    min_reliability=0.12,
):
    for estimator in (estimate_affine, estimate_similarity):
        estimate = estimator(
            img1,
            img2,
            stable_mask,
            stable_mask,
        )
        if estimate is not None and estimate.reliability >= min_reliability:
            return estimate

    estimate = phase_translation(
        img1,
        img2,
        stable_mask,
    )
    if estimate.reliability >= min_reliability * 0.6:
        return estimate

    global_estimate = phase_translation(
        img1,
        img2,
        None,
    )
    if global_estimate.reliability >= min_reliability * 0.5:
        global_estimate.reliability *= 0.75
        global_estimate.model = "translation_global_phase"
        return global_estimate

    return identity_transform(0.0, "no_transport")

def warp_mask(
    mask,
    matrix,
    out_shape,
    inverse=False,
):
    height, width = out_shape
    transform = cv2.invertAffineTransform(matrix) if inverse else matrix
    return cv2.warpAffine(
        mask.astype(np.float32),
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
