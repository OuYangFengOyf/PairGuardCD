from __future__ import annotations
import torch

class BinaryChangeMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(self, probability, target, valid_mask=None, threshold=0.5):
        pred = probability >= threshold
        truth = target >= 0.5
        if valid_mask is not None:
            valid = valid_mask > 0
            pred = pred[valid]
            truth = truth[valid]
        self.tp += int((pred & truth).sum().item())
        self.fp += int((pred & ~truth).sum().item())
        self.fn += int((~pred & truth).sum().item())
        self.tn += int((~pred & ~truth).sum().item())

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-9)
        recall = self.tp / (self.tp + self.fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-9)
        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "IoU": iou,
        }
