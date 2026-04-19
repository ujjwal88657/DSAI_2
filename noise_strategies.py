"""
training/noise_strategies.py
All noise-robust sample-selection and label-correction strategies.

  SmallLossTrick                — keep lowest-loss examples [Jiang et al. ICML 2018]
  CoTeaching                    — dual-network cross-selection [Han et al. NeurIPS 2018]
  GaussianMixtureNoiseSeparator — GMM on loss distribution [Li et al. ICLR 2020]
  LabelRefurbishmentStore       — EMA prediction store [Song et al. ICML 2019]
  NoiseRateEstimator            — tracks GMM estimates across epochs
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
from losses.robust_losses import compute_per_sample_loss


# ── 1. Small-Loss Trick ────────────────────────────────────────────────────────
class SmallLossTrick:
    """
    Treat the lowest keep_ratio% loss examples in a batch as clean.
    Keep ratio is linearly annealed from 1.0 → keep_ratio_final after start_epoch.
    """
    def __init__(self, keep_ratio_initial=1.0, keep_ratio_final=0.70,
                 start_epoch=3, total_epochs=10):
        self.keep_ratio_initial = keep_ratio_initial
        self.keep_ratio_final   = keep_ratio_final
        self.start_epoch        = start_epoch
        self.total_epochs       = total_epochs

    def get_keep_ratio(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return self.keep_ratio_initial
        prog  = (epoch - self.start_epoch) / max(1, self.total_epochs - self.start_epoch)
        ratio = self.keep_ratio_initial - prog * (self.keep_ratio_initial - self.keep_ratio_final)
        return max(ratio, self.keep_ratio_final)

    def select(self, losses: torch.Tensor, epoch: int) -> torch.Tensor:
        """Return indices of lowest-loss keep_ratio% examples."""
        k   = max(1, int(len(losses) * self.get_keep_ratio(epoch)))
        idx = losses.argsort()
        return idx[:k]


# ── 2. Co-Teaching ────────────────────────────────────────────────────────────
class CoTeaching:
    """
    Each network selects small-loss examples FROM ITS OWN BATCH and hands
    them to the PEER network for a gradient update.
    Gradually increases the fraction of examples dropped (forget rate schedule).

    Reference: Han et al., NeurIPS 2018.
    """
    def __init__(self, forget_rate=0.20, num_gradual=5, total_epochs=10, exponent=1.0):
        self.forget_rate   = forget_rate
        self.num_gradual   = num_gradual
        self.total_epochs  = total_epochs
        self.exponent      = exponent
        self._schedule     = self._build_schedule()

    def _build_schedule(self) -> np.ndarray:
        s = np.ones(self.total_epochs) * self.forget_rate
        s[:self.num_gradual] = np.linspace(0, self.forget_rate ** self.exponent, self.num_gradual)
        return s

    def get_forget_rate(self, epoch: int) -> float:
        return float(self._schedule[min(epoch, len(self._schedule) - 1)])

    def select(self, losses: torch.Tensor, epoch: int) -> torch.Tensor:
        """Indices of examples to KEEP (lowest loss)."""
        fr  = self.get_forget_rate(epoch)
        k   = max(1, int(len(losses) * (1.0 - fr)))
        return losses.argsort()[:k]


# ── 3. GMM Noise Separator (DivideMix-style) ──────────────────────────────────
class GaussianMixtureNoiseSeparator:
    """
    Fits a 2-component 1-D GMM to per-sample loss values.
    Component with smaller mean → clean; larger mean → noisy.
    Returns per-sample p_clean probability.

    Reference: Li et al., ICLR 2020.
    """
    def __init__(self, p_threshold: float = 0.5):
        self.p_threshold = p_threshold
        self.mu = self.sigma = self.pi = None

    def fit_predict(self, losses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        losses = losses.ravel()
        med = np.median(losses)
        lo, hi = losses[losses <= med], losses[losses > med]
        mu    = np.array([lo.mean(), hi.mean()])
        sigma = np.array([lo.std() + 1e-6, hi.std() + 1e-6])
        pi    = np.array([0.5, 0.5])

        for _ in range(60):                                # EM
            p0 = pi[0] * norm.pdf(losses, mu[0], sigma[0]) + 1e-10
            p1 = pi[1] * norm.pdf(losses, mu[1], sigma[1]) + 1e-10
            r0 = p0 / (p0 + p1);  r1 = 1.0 - r0
            n0, n1 = r0.sum(), r1.sum()
            mu[0]    = (r0 * losses).sum() / (n0 + 1e-10)
            mu[1]    = (r1 * losses).sum() / (n1 + 1e-10)
            sigma[0] = np.sqrt((r0 * (losses - mu[0]) ** 2).sum() / (n0 + 1e-10)) + 1e-6
            sigma[1] = np.sqrt((r1 * (losses - mu[1]) ** 2).sum() / (n1 + 1e-10)) + 1e-6
            pi[0]    = n0 / len(losses);  pi[1] = 1.0 - pi[0]

        if mu[0] > mu[1]:                                  # enforce component 0 = clean
            mu, sigma, pi = mu[[1, 0]], sigma[[1, 0]], pi[[1, 0]]

        self.mu, self.sigma, self.pi = mu, sigma, pi

        p_clean_num = pi[0] * norm.pdf(losses, mu[0], sigma[0]) + 1e-10
        p_total     = p_clean_num + pi[1] * norm.pdf(losses, mu[1], sigma[1]) + 1e-10
        p_clean     = p_clean_num / p_total
        is_clean    = p_clean > self.p_threshold
        est_nr      = float(1.0 - is_clean.mean())
        return p_clean, is_clean, est_nr

    def get_stats(self) -> Dict:
        if self.mu is None:
            return {}
        return {
            "mu_clean":    float(self.mu[0]),    "mu_noisy":    float(self.mu[1]),
            "sigma_clean": float(self.sigma[0]), "sigma_noisy": float(self.sigma[1]),
            "pi_clean":    float(self.pi[0]),
        }


# ── 4. Label Refurbishment Store ──────────────────────────────────────────────
class LabelRefurbishmentStore:
    """
    Maintains an EMA of model soft predictions for every training sample.
    Used by bootstrapping to get stable pseudo-labels.
    """
    def __init__(self, n_samples: int, num_classes: int, alpha: float = 0.9):
        self.num_classes  = num_classes
        self.alpha        = alpha
        self.ema_preds    = torch.ones(n_samples, num_classes).float() / num_classes
        self.update_count = torch.zeros(n_samples).long()

    def update(self, indices: torch.Tensor, soft_preds: torch.Tensor):
        idx = indices.cpu()
        p   = soft_preds.detach().cpu().float()
        self.ema_preds[idx] = self.alpha * self.ema_preds[idx] + (1.0 - self.alpha) * p
        self.update_count[idx] += 1

    def get_ema(self, indices: torch.Tensor, device) -> torch.Tensor:
        return self.ema_preds[indices.cpu()].to(device)


# ── 5. Noise Rate Estimator ────────────────────────────────────────────────────
class NoiseRateEstimator:
    """Runs GMM each epoch and records estimated noise rates."""
    def __init__(self):
        self.history: List[float] = []
        self.gmm = GaussianMixtureNoiseSeparator()

    def estimate(self, losses: np.ndarray, threshold: float = 0.5) -> Tuple[Dict, np.ndarray, np.ndarray]:
        p_clean, is_clean, est_nr = self.gmm.fit_predict(losses)
        self.history.append(est_nr)
        stats = self.gmm.get_stats()
        stats.update({"estimated_noise_rate": est_nr,
                      "num_clean": int(is_clean.sum()),
                      "num_noisy": int((~is_clean).sum())})
        return stats, p_clean, is_clean
