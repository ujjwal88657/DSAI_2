"""
evaluation/metrics.py
Inference + full metric suite for binary hate-speech detection.

Outputs:
  - Accuracy, Precision, Recall, F1 (macro + weighted)
  - Per-class breakdown
  - Per-language breakdown (English / Hindi / Hinglish)
  - Confusion matrix
  - Optional embedding extraction for visualisation
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(
    model,
    loader,
    device: torch.device,
    return_embeddings: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
    """
    Returns: preds, probs, true_labels, embeddings (or None), languages (or None)
    """
    model.eval()
    preds, probs, labels, embs, langs = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Inference", ncols=80, leave=False):
            ids = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            tt  = batch["token_type_ids"].to(device)
            lbl = batch["label"]

            out = model(ids, msk, tt, return_embeddings=return_embeddings)
            p   = F.softmax(out["logits"], dim=1)

            preds.append(p.argmax(1).cpu().numpy())
            probs.append(p.cpu().numpy())
            labels.append(lbl.numpy())
            if return_embeddings and "embeddings" in out:
                embs.append(out["embeddings"].cpu().numpy())
            if "language" in batch:
                langs.extend(batch["language"] if isinstance(batch["language"], list) else
                             [x for x in batch["language"]])

    return (
        np.concatenate(preds),
        np.concatenate(probs),
        np.concatenate(labels),
        np.concatenate(embs) if embs else None,
        langs if langs else None,
    )


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(
    preds:  np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    acc      = accuracy_score(labels, preds)
    prec_mac = precision_score(labels, preds, average="macro",    zero_division=0)
    rec_mac  = recall_score(   labels, preds, average="macro",    zero_division=0)
    f1_mac   = f1_score(       labels, preds, average="macro",    zero_division=0)
    f1_w     = f1_score(       labels, preds, average="weighted", zero_division=0)
    f1_per   = f1_score(       labels, preds, average=None,       zero_division=0)
    prec_per = precision_score(labels, preds, average=None,       zero_division=0)
    rec_per  = recall_score(   labels, preds, average=None,       zero_division=0)
    cm       = confusion_matrix(labels, preds)
    report   = classification_report(labels, preds, target_names=class_names, zero_division=0)

    return {
        "accuracy":           float(acc),
        "precision_macro":    float(prec_mac),
        "recall_macro":       float(rec_mac),
        "f1_macro":           float(f1_mac),
        "f1_weighted":        float(f1_w),
        "per_class_f1":       f1_per.tolist(),
        "per_class_precision":prec_per.tolist(),
        "per_class_recall":   rec_per.tolist(),
        "confusion_matrix":   cm.tolist(),
        "classification_report": report,
    }


def compute_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    eps    = 1e-7
    clipped = np.clip(probs, eps, 1 - eps)
    return float(-np.log(clipped[np.arange(len(labels)), labels]).mean())


def compute_per_language_metrics(
    preds:     np.ndarray,
    labels:    np.ndarray,
    languages: List[str],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Breakdown accuracy and F1 per language group."""
    langs = np.array(languages)
    out   = {}
    for lang in np.unique(langs):
        mask = langs == lang
        if mask.sum() == 0:
            continue
        m = compute_metrics(preds[mask], labels[mask], class_names)
        out[lang] = {
            "n":         int(mask.sum()),
            "accuracy":  m["accuracy"],
            "f1_macro":  m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
        }
    return out


# ── High-level evaluator ────────────────────────────────────────────────────────
def evaluate_model(
    model,
    loader,
    device: torch.device,
    cfg,
    split:            str  = "test",
    verbose:          bool = True,
    return_predictions: bool = False,
) -> Dict:
    class_names = cfg.data.class_names

    preds, probs, labels, _, languages = predict(
        model, loader, device, return_embeddings=False
    )
    metrics        = compute_metrics(preds, labels, class_names)
    metrics["loss"] = compute_loss(probs, labels)

    # Per-language breakdown (if available)
    if languages:
        metrics["per_language"] = compute_per_language_metrics(
            preds, labels, languages, class_names
        )

    if verbose:
        print(f"\n  [{split.upper()} RESULTS]")
        print(f"  Accuracy:       {metrics['accuracy']:.4f}")
        print(f"  F1 Weighted:    {metrics['f1_weighted']:.4f}")
        print(f"  F1 Macro:       {metrics['f1_macro']:.4f}")
        print(f"  Precision:      {metrics['precision_macro']:.4f}")
        print(f"  Recall:         {metrics['recall_macro']:.4f}")
        print(f"\n{metrics['classification_report']}")
        if "per_language" in metrics:
            print("  Per-language breakdown:")
            for lang, m in metrics["per_language"].items():
                print(f"    {lang:12s}  n={m['n']:5,}  acc={m['accuracy']:.4f}  "
                      f"f1={m['f1_weighted']:.4f}")

    if return_predictions:
        metrics["preds"]  = preds
        metrics["probs"]  = probs
        metrics["labels"] = labels

    return metrics
