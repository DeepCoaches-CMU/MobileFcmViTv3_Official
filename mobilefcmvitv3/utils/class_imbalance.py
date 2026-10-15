"""
mobilefcmvitv3/utils/class_imbalance.py

Utilities for handling BUSI class imbalance.

BUSI class distribution (full dataset):
  benign:    437  (56.0%)
  malignant: 210  (26.9%)
  normal:    133  (17.1%)

Inverse-frequency weights: w_c = N / (C * n_c)
  benign:    780 / (3 * 437) = 0.595
  malignant: 780 / (3 * 210) = 1.238
  normal:    780 / (3 * 133) = 1.955
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

# BUSI inverse-frequency class weights (benign=0, malignant=1, normal=2)
BUSI_CLASS_WEIGHTS = torch.tensor([0.595, 1.238, 1.955])

# ── Cost-sensitive misclassification matrix ───────────────────────────────────
# Class-weighted CE penalizes by TRUE class only (symmetric across whichever
# wrong class it lands on). Error analysis on tune_05_augmix's held-out test
# predictions showed the clinically worst errors are direction-specific:
# malignant cases downgraded to benign (2/32 test errors) or, worse, to normal
# (3/32) — a cancer flagged as "no lesion at all" gets no follow-up whatsoever.
# Benign lesions called normal are a lesser version of the same problem.
# False alarms (benign/normal called malignant) are comparatively cheap —
# unnecessary follow-up, not a missed cancer — and are left unpenalized here so
# this doesn't just trade one bias for another.
#
# Built from an urgency ranking (normal=0, benign=1, malignant=2): cost(i,j) =
# 1 + max(0, urgency(i) - urgency(j)), i.e. only "downgrading" errors (truth
# more urgent than prediction) are penalized beyond standard CE.
#   rows = true class, cols = predicted class (benign=0, malignant=1, normal=2)
BUSI_COST_MATRIX = torch.tensor([
    # pred: benign  malignant  normal
    [   1.0,        1.0,        2.0],  # true: benign
    [   2.0,        1.0,        3.0],  # true: malignant
    [   1.0,        1.0,        1.0],  # true: normal
])


class WeightedSoftTargetCrossEntropy(nn.Module):
    """
    Cross-entropy loss for soft (mixup) targets with per-class weights.
    Compatible with timm's Mixup which produces soft label tensors.
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer('weight', weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        weighted = -(target * log_probs * self.weight.unsqueeze(0))
        return weighted.sum(dim=-1).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Focuses training on hard, misclassified examples.
    Supports per-class weights and label smoothing.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer('weight', weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(x, target, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class CostSensitiveCrossEntropy(nn.Module):
    """
    Cross-entropy scaled by a per-sample misclassification cost:
        L = cost[true, argmax(logits)] * CE(logits, true)

    argmax is used only to look up the scalar cost (treated as a constant,
    like p_t in FocalLoss) — gradients still flow through the CE term itself,
    so this doesn't require backpropagating through a discrete argmax.

    Args:
        cost_matrix: (C, C) — cost_matrix[i, j] = cost of predicting class j
            when the true class is i. Diagonal (correct predictions) should
            be 1.0 so this reduces to standard (optionally weighted) CE when
            the model is already right.
        weight:      optional per-class weights (stacks with cost_matrix)
        label_smoothing: forwarded to the underlying cross_entropy
    """
    def __init__(self, cost_matrix: torch.Tensor, weight: torch.Tensor = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer('cost_matrix', cost_matrix)
        self.label_smoothing = label_smoothing
        self.register_buffer('weight', weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(x, target, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        with torch.no_grad():
            pred = x.argmax(dim=-1)
            cost = self.cost_matrix[target, pred]
        return (cost * ce).mean()


def build_loss_fn(smoothing: float = 0.1,
                  mixup_active: bool = False,
                  use_class_weights: bool = True,
                  num_classes: int = 3,
                  focal_gamma: float = 0.0,
                  cost_sensitive_beta: float = 0.0,
                  device: str = 'cuda') -> nn.Module:
    """
    Build the appropriate loss function based on training configuration.

    Args:
        smoothing:      label smoothing epsilon
        mixup_active:   whether mixup/cutmix is active (soft targets)
        use_class_weights: apply BUSI inverse-frequency class weights
        num_classes:    number of output classes
        focal_gamma:    focal loss gamma (0 = standard cross-entropy)
        cost_sensitive_beta: scales BUSI_COST_MATRIX's off-diagonal penalties
            (0 = disabled/standard CE, 1.0 = matrix as defined). Incompatible
            with mixup (soft targets have no single argmax to cost-lookup) —
            mixup_active takes priority if both are set, matching focal_gamma's
            existing precedence.
        device:         device string
    """
    weights = BUSI_CLASS_WEIGHTS.to(device) if (use_class_weights and num_classes == 3) else None

    if mixup_active:
        if weights is not None:
            loss_fn = WeightedSoftTargetCrossEntropy(weight=weights)
        else:
            loss_fn = SoftTargetCrossEntropy()
    elif cost_sensitive_beta > 0 and num_classes == 3:
        cost_matrix = 1.0 + cost_sensitive_beta * (BUSI_COST_MATRIX.to(device) - 1.0)
        loss_fn = CostSensitiveCrossEntropy(cost_matrix=cost_matrix, weight=weights,
                                            label_smoothing=smoothing)
    elif focal_gamma > 0:
        loss_fn = FocalLoss(gamma=focal_gamma, weight=weights, label_smoothing=smoothing)
    elif smoothing > 0:
        if weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
        else:
            loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weights) if weights is not None \
            else nn.CrossEntropyLoss()

    if weights is not None:
        print(f"Using class-weighted loss: {weights.tolist()}")

    return loss_fn.to(device)
