from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class RelationState(str, Enum):
    OBJECT_OBJECT = "oo"
    OBJECT_NULL = "o_null"
    NULL_OBJECT = "null_o"
    UNKNOWN = "unknown"
    REJECTED = "rejected"

@dataclass
class Hypothesis:
    mask1: Optional[np.ndarray]
    mask2: Optional[np.ndarray]
    state: RelationState
    q_src: float = 1.0
    search_reliability: float = 1.0
    counterpart_evidence: float = 0.0
    transform_reliability: float = 0.0
    seed_overlap: float = 0.0
    aligned_difference: float = 0.0
    bidirectional_consistency: float = 0.0
    boundary_consistency: float = 0.0
    missing_ratio: float = 0.0
    proposal_scale: float = 0.0
    group_id: str = ""
    proposal: Optional[np.ndarray] = None

def iou(a, b):
    if a is None or b is None:
        return 0.0
    aa = a > 0
    bb = b > 0
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    return float(inter / (union + 1e-6))

def f1_binary(pred, gt):
    p = pred > 0.5
    g = gt > 0.5
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    return float(2 * tp / (2 * tp + fp + fn + 1e-6))

def null_reliability(q_src, search_reliability, counterpart_evidence):
    return float(np.clip(q_src * search_reliability * (1.0 - counterpart_evidence), 0.0, 1.0))

def proposal_features(h):
    m1, m2 = h.mask1, h.mask2
    a1 = float(m1.mean()) if m1 is not None else 0.0
    a2 = float(m2.mean()) if m2 is not None else 0.0
    state = {
        RelationState.OBJECT_OBJECT: (1, 0, 0, 0),
        RelationState.OBJECT_NULL: (0, 1, 0, 0),
        RelationState.NULL_OBJECT: (0, 0, 1, 0),
        RelationState.UNKNOWN: (0, 0, 0, 1),
        RelationState.REJECTED: (0, 0, 0, 1),
    }[h.state]
    values = [
        h.q_src,
        h.search_reliability,
        h.counterpart_evidence,
        h.transform_reliability,
        h.seed_overlap,
        h.aligned_difference,
        h.bidirectional_consistency,
        h.boundary_consistency,
        h.missing_ratio,
        a1,
        a2,
        abs(a1 - a2),
        iou(m1, m2),
        *state,
        h.proposal_scale,
    ]
    return np.asarray(values, dtype=np.float32)

class UtilitySelector(nn.Module):
    def __init__(self, in_dim=18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.relative_utility = nn.Linear(32, 1)
        self.usability = nn.Linear(32, 1)

    def forward(self, x):
        h = self.net(x)
        return self.relative_utility(h).squeeze(-1), torch.sigmoid(self.usability(h)).squeeze(-1)

class UtilitySelectorEnsemble(nn.Module):
    def __init__(self, selectors):
        super().__init__()
        if not selectors:
            raise ValueError("At least one utility selector is required.")
        self.selectors = nn.ModuleList(selectors)

    def forward(self, x):
        rel = []
        use = []
        for selector in self.selectors:
            r, u = selector(x)
            rel.append(r)
            use.append(u)
        return torch.stack(rel, dim=0).mean(dim=0), torch.stack(use, dim=0).mean(dim=0)

def fit_selector(features, relative_utility, usable, epochs=150, lr=1e-3, device="cpu"):
    x = torch.as_tensor(features, dtype=torch.float32)
    utility = torch.as_tensor(relative_utility, dtype=torch.float32)
    usability = torch.as_tensor(usable, dtype=torch.float32)
    model = UtilitySelector(x.shape[1]).to(device)
    dataset = TensorDataset(x, utility, usability)
    loader = DataLoader(dataset, batch_size=min(128, len(dataset)), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    regression_loss = nn.SmoothL1Loss()
    classification_loss = nn.BCELoss()
    for _ in range(epochs):
        model.train()
        for xb, ub, sb in loader:
            xb = xb.to(device)
            ub = ub.to(device)
            sb = sb.to(device)
            pred_utility, pred_usability = model(xb)
            loss = regression_loss(pred_utility, ub) + classification_loss(pred_usability, sb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.cpu().eval()

def cross_fit_selectors(features, relative_utility, usable, groups, k=5, seed=2026, device="cpu"):
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    group_folds = np.array_split(unique_groups, k)
    models = []
    oof_rel = np.zeros(len(groups), dtype=np.float32)
    oof_use = np.zeros(len(groups), dtype=np.float32)
    for fold in group_folds:
        held_mask = np.isin(groups, fold)
        train_mask = ~held_mask
        model = fit_selector(
            np.asarray(features)[train_mask],
            np.asarray(relative_utility)[train_mask],
            np.asarray(usable)[train_mask],
            device=device,
        )
        models.append(model)
        if held_mask.any():
            with torch.no_grad():
                rel, use = model(torch.as_tensor(np.asarray(features)[held_mask], dtype=torch.float32))
            oof_rel[held_mask] = rel.numpy()
            oof_use[held_mask] = use.numpy()
    return models, oof_rel, oof_use, group_folds

def load_selector_ensemble(checkpoint_path, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    states = checkpoint["selectors"]
    selectors = []
    for state in states:
        model = UtilitySelector()
        model.load_state_dict(state)
        model.eval()
        selectors.append(model)
    return UtilitySelectorEnsemble(selectors).eval()

class NARPUS:
    def __init__(self, selector, rel_threshold=0.05, use_threshold=0.70, null_threshold=0.60):
        if selector is None:
            raise ValueError("A trained utility selector is required.")
        self.selector = selector
        self.rel_threshold = float(rel_threshold)
        self.use_threshold = float(use_threshold)
        self.null_threshold = float(null_threshold)

    def certify_state(self, h):
        if h.state in (RelationState.OBJECT_NULL, RelationState.NULL_OBJECT):
            reliability = null_reliability(h.q_src, h.search_reliability, h.counterpart_evidence)
            if reliability < self.null_threshold:
                return RelationState.UNKNOWN
        return h.state

    def score(self, h):
        with torch.no_grad():
            utility, usability = self.selector(torch.from_numpy(proposal_features(h)).unsqueeze(0))
        return float(utility.item()), float(usability.item())

    def select(self, hypotheses):
        accepted = []
        unknown = []
        rejected = []
        for h in hypotheses:
            h.state = self.certify_state(h)
            if h.state == RelationState.UNKNOWN:
                unknown.append(h)
                continue
            utility, usability = self.score(h)
            if utility >= self.rel_threshold and usability >= self.use_threshold:
                accepted.append((h, utility, usability))
            else:
                h.state = RelationState.REJECTED
                rejected.append((h, utility, usability))
        return accepted, unknown, rejected

def make_utility_target(proposal, seed, gt, usable_f1=0.65):
    proposal_score = f1_binary(proposal, gt)
    seed_score = f1_binary(seed, gt)
    return proposal_score - seed_score, float(proposal_score >= usable_f1)

def build_pseudo_label(
    shape,
    accepted,
    seed,
    strict_background_mask=None,
    strict_bg_threshold=0.10,
):
    h, w = shape
    ye = np.zeros((h, w), dtype=np.float32)
    valid = np.zeros((h, w), dtype=np.uint8)
    occupied = np.zeros((h, w), dtype=np.uint8)
    for item in accepted:
        hyp = item[0]
        if hyp.proposal is not None:
            positive = (hyp.proposal > 0.5).astype(np.uint8)
        else:
            masks = [m for m in (hyp.mask1, hyp.mask2) if m is not None]
            if not masks:
                continue
            positive = np.maximum.reduce([m.astype(np.uint8) for m in masks])
        ye[positive > 0] = 1.0
        valid[positive > 0] = 1
        occupied[positive > 0] = 1

    if strict_background_mask is None:
        background = seed < strict_bg_threshold
    else:
        background = strict_background_mask.astype(bool)
    background &= occupied == 0
    valid[background] = 1
    return ye, valid
