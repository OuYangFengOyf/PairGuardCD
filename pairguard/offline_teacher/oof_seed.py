from __future__ import annotations
import numpy as np

def make_group_folds(group_ids, k=5, seed=2026):
    groups = np.asarray(group_ids, dtype=object)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    folds = []
    for held_groups in np.array_split(unique_groups, k):
        folds.append(np.flatnonzero(np.isin(groups, held_groups)).tolist())
    return folds

def mean_unlabeled_prediction(predictions):
    return np.stack(predictions, axis=0).mean(axis=0).astype(np.float32)
