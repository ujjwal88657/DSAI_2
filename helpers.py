"""utils/helpers.py — General utilities."""
import os, json, random, hashlib
import numpy as np
import torch
from typing import Dict


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda"  and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps"   and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EMAModel:
    """Exponential Moving Average of model weights for stable inference."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self._orig  = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply_shadow(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self._orig[n] = p.data.clone(); p.data.copy_(self.shadow[n])

    def restore(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if n in self._orig: p.data.copy_(self._orig[n])
        self._orig.clear()


def gradient_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def save_json(obj: Dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
