"""
losses/robust_losses.py
Noise-robust loss functions, all supporting per-sample (reduction='none') mode
so Co-Teaching, DivideMix, and small-loss selection can inspect individual losses.

Implemented:
  CrossEntropyLoss          — baseline (noise-vulnerable)
  SymmetricCrossEntropyLoss — SCE  [Wang et al. ICCV 2019]
  GeneralizedCrossEntropyLoss — GCE [Zhang & Sabuncu NeurIPS 2018]
  MAELoss                   — provably noise-tolerant [Ghosh et al. AAAI 2017]
  BootstrappingLoss         — label refurbishment [Reed et al. ICLR 2015]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

EPS = 1e-7


# ── Helper ─────────────────────────────────────────────────────────────────────
def _onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes).float()


# ── 1. Standard CE ─────────────────────────────────────────────────────────────
class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int, reduction: str = "none"):
        super().__init__()
        self.num_classes = num_classes
        self.reduction   = reduction
        self._ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, labels, weights=None):
        loss = self._ce(logits, labels)
        if weights is not None:
            loss = loss * weights
        return _reduce(loss, self.reduction)


# ── 2. Symmetric CE (SCE) ──────────────────────────────────────────────────────
class SymmetricCrossEntropyLoss(nn.Module):
    """
    SCE = α · CE(p, q) + β · RCE(q, p)

    The Reverse CE term prevents the model from assigning full probability
    to wrong targets. Empirically superior to CE at ≥20% label noise.

    Reference: Wang et al., "Symmetric Cross Entropy for Robust Learning
               with Noisy Labels", ICCV 2019.
    """
    def __init__(self, num_classes: int, alpha: float = 0.1, beta: float = 1.0,
                 reduction: str = "none", A: float = -6.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha;  self.beta = beta
        self.reduction = reduction;  self.A = A
        self._ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, labels, weights=None):
        # Forward CE
        ce = self._ce(logits, labels)

        # Reverse CE
        pred     = F.softmax(logits, dim=1).clamp(min=EPS)
        oh       = _onehot(labels, self.num_classes).to(logits.device)
        log_oh   = torch.clamp(oh, min=EPS).log().clamp(min=self.A)
        rce      = -(pred * log_oh).sum(dim=1)

        loss = self.alpha * ce + self.beta * rce
        if weights is not None:
            loss = loss * weights
        return _reduce(loss, self.reduction)


# ── 3. Generalized CE (GCE) ────────────────────────────────────────────────────
class GeneralizedCrossEntropyLoss(nn.Module):
    """
    GCE(f, y) = (1 - f_y^q) / q
    q=0.7 interpolates between MAE (q→0) and CE (q→1).

    Reference: Zhang & Sabuncu, NeurIPS 2018.
    """
    def __init__(self, num_classes: int, q: float = 0.7, reduction: str = "none"):
        super().__init__()
        self.num_classes = num_classes
        self.q = q;  self.reduction = reduction

    def forward(self, logits, labels, weights=None):
        pred = F.softmax(logits, dim=1).clamp(min=EPS)
        p_y  = (pred * _onehot(labels, self.num_classes).to(logits.device)).sum(dim=1)
        loss = (1.0 - p_y ** self.q) / self.q
        if weights is not None:
            loss = loss * weights
        return _reduce(loss, self.reduction)


# ── 4. MAE Loss ────────────────────────────────────────────────────────────────
class MAELoss(nn.Module):
    """
    L_MAE = 1 - f_y   (provably noise-tolerant under symmetric noise τ < (c-1)/c)

    Reference: Ghosh et al., AAAI 2017.
    """
    def __init__(self, num_classes: int, reduction: str = "none"):
        super().__init__()
        self.num_classes = num_classes
        self.reduction   = reduction

    def forward(self, logits, labels, weights=None):
        pred = F.softmax(logits, dim=1)
        p_y  = (pred * _onehot(labels, self.num_classes).to(logits.device)).sum(dim=1)
        loss = 1.0 - p_y
        if weights is not None:
            loss = loss * weights
        return _reduce(loss, self.reduction)


# ── 5. Bootstrapping (Label Refurbishment) ─────────────────────────────────────
class BootstrappingLoss(nn.Module):
    """
    y_corrected = β · y_noisy + (1-β) · f(x)
    Soft CE against refurbished target reduces noise damage over time.

    Reference: Reed et al., ICLR 2015.
    """
    def __init__(self, num_classes: int, beta: float = 0.8, reduction: str = "none"):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta;  self.reduction = reduction

    def update_beta(self, new_beta: float):
        self.beta = new_beta

    def forward(self, logits, labels, weights=None):
        pred       = F.softmax(logits, dim=1)
        oh         = _onehot(labels, self.num_classes).to(logits.device)
        corrected  = self.beta * oh + (1.0 - self.beta) * pred.detach()
        loss       = -(corrected * torch.log(pred + EPS)).sum(dim=1)
        if weights is not None:
            loss = loss * weights
        return _reduce(loss, self.reduction)


# ── Factory ────────────────────────────────────────────────────────────────────
def build_loss(cfg) -> nn.Module:
    nc = cfg.data.num_classes
    t  = cfg.training.loss_type
    if   t == "ce":  return CrossEntropyLoss(nc, "none")
    elif t == "sce": return SymmetricCrossEntropyLoss(nc, cfg.training.sce_alpha, cfg.training.sce_beta, "none")
    elif t == "gce": return GeneralizedCrossEntropyLoss(nc, cfg.training.gce_q,   "none")
    elif t == "mae": return MAELoss(nc, "none")
    raise ValueError(f"Unknown loss_type: {t}")


@torch.no_grad()
def compute_per_sample_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Plain CE per sample — used for Co-Teaching / GMM selection."""
    return F.cross_entropy(logits, labels, reduction="none")


# ── Util ───────────────────────────────────────────────────────────────────────
def _reduce(loss: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":  return loss.mean()
    if mode == "sum":   return loss.sum()
    return loss   # "none"
