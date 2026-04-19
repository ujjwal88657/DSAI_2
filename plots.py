"""
visualization/plots.py
Six publication-quality plots:
  1. Training curves (loss, accuracy, F1, noise rate, rates)
  2. Confusion matrix (raw + normalized)
  3. Per-class performance bar chart
  4. Loss distribution + GMM overlay (DivideMix diagnostic)
  5. Embedding visualization (PCA + UMAP)
  6. Per-language performance breakdown
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Dict, List, Optional

# ── Palette ────────────────────────────────────────────────────────────────────
PAL = {
    "not_hate": "#2CA02C", "hate": "#D62728",
    "train":    "#1F4E79", "val": "#C00000",
    "model1":   "#2E75B6", "model2": "#ED7D31",
    "clean":    "#2E75B6", "noisy": "#D62728",
    "english":  "#5B9BD5", "hindi": "#ED7D31", "hinglish": "#70AD47",
}

def _save(fig, output_dir: str, name: str, dpi: int = 150) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Viz] {path}")
    return path

def _setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "#F8F9FA",
        "grid.color": "white", "grid.linewidth": 0.8,
    })

_setup()


# ── 1. Training curves ─────────────────────────────────────────────────────────
def plot_training_curves(history: Dict, output_dir: str, dpi: int = 150) -> str:
    epochs = list(range(1, len(history["train_loss_m1"]) + 1))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Dynamics — Noise-Robust Co-Teaching Pipeline", fontsize=14, y=1.01)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss_m1"], color=PAL["model1"], lw=2, label="Train M1")
    ax.plot(epochs, history["train_loss_m2"], color=PAL["model2"], lw=2, ls="--", label="Train M2")
    ax.plot(epochs, history["val_loss"],      color=PAL["val"],    lw=2, label="Val")
    ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.4)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["val_acc"], color=PAL["model1"], lw=2.5, marker="o", ms=4)
    ax.set_title("Val Accuracy"); ax.set_ylim([0, 1]); ax.grid(True, alpha=0.4)

    # F1
    ax = axes[0, 2]
    ax.plot(epochs, history["val_f1"], color=PAL["model2"], lw=2.5, marker="s", ms=4)
    ax.set_title("Val F1 Weighted"); ax.set_ylim([0, 1]); ax.grid(True, alpha=0.4)

    # Noise rate estimate
    ax = axes[1, 0]
    nr_vals = [v for v in history.get("estimated_noise_rate", []) if v is not None]
    if nr_vals:
        ax.plot(range(1, len(nr_vals)+1), nr_vals, color="#9467BD", lw=2)
        ax.axhline(y=0.30, color="red", ls=":", lw=1.5, label="True 30%")
        ax.set_title("GMM Noise Estimate"); ax.set_ylim([0, 0.65])
        ax.legend(); ax.grid(True, alpha=0.4)
    else:
        ax.set_title("GMM Noise Estimate"); ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

    # Keep / Forget rates
    ax = axes[1, 1]
    kr = history.get("keep_ratio",  [])
    fr = history.get("forget_rate", [])
    if kr: ax.plot(epochs, kr, color=PAL["train"], lw=2, label="Keep ratio")
    if fr: ax.plot(epochs, fr, color=PAL["val"],   lw=2, ls="--", label="Forget rate")
    ax.set_title("Selection Rates"); ax.set_ylim([0, 1.05])
    ax.legend(); ax.grid(True, alpha=0.4)

    # Epoch times
    ax = axes[1, 2]
    et = history.get("epoch_time", [])
    if et:
        ax.bar(epochs, et, color=PAL["model1"], alpha=0.7)
        ax.set_title("Epoch Duration (s)"); ax.grid(True, alpha=0.4, axis="y")

    fig.tight_layout()
    return _save(fig, output_dir, "training_curves", dpi)


# ── 2. Confusion matrix ────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          output_dir: str, title: str = "Confusion Matrix", dpi: int = 150) -> str:
    cm = np.array(cm)
    cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(title, fontsize=13)
    for ax, data, fmt, sub, cmap in zip(
        axes, [cm, cm_norm], ["d", ".2f"], ["Raw counts", "Row-normalised"], ["Blues", "YlOrRd"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="white", ax=ax, square=True, cbar_kws={"shrink": 0.8})
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(sub)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, output_dir, "confusion_matrix", dpi)


# ── 3. Per-class performance ───────────────────────────────────────────────────
def plot_per_class_performance(metrics: Dict, class_names: List[str],
                               output_dir: str, dpi: int = 150) -> str:
    x     = np.arange(len(class_names))
    width = 0.25
    prec  = metrics.get("per_class_precision", [0]*len(class_names))
    rec   = metrics.get("per_class_recall",    [0]*len(class_names))
    f1    = metrics.get("per_class_f1",        [0]*len(class_names))

    fig, ax = plt.subplots(figsize=(9, 6))
    for offset, vals, label, color in [
        (-width, prec, "Precision", "#2E75B6"),
        (0,      rec,  "Recall",    "#ED7D31"),
        (width,  f1,   "F1",        "#70AD47"),
    ]:
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylim([0, 1.15]); ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.4, axis="y")
    acc = metrics.get("accuracy", 0); f1w = metrics.get("f1_weighted", 0)
    ax.text(0.98, 0.98, f"Acc: {acc:.3f}\nF1w: {f1w:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="#E8F4FD", ec="#2E75B6"))
    fig.tight_layout()
    return _save(fig, output_dir, "per_class_performance", dpi)


# ── 4. Loss distribution ───────────────────────────────────────────────────────
def plot_loss_distribution(
    losses: np.ndarray, is_noisy: Optional[np.ndarray] = None,
    output_dir: str = "./visualizations",
    epoch: Optional[int] = None, gmm_params: Optional[Dict] = None, dpi: int = 150,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    suffix = f" (epoch {epoch})" if epoch is not None else ""

    ax = axes[0]
    if is_noisy is not None:
        cln = losses[is_noisy == 0]; nsy = losses[is_noisy == 1]
        ax.hist(cln, bins=60, alpha=0.65, color=PAL["clean"], label=f"Clean ({len(cln):,})", density=True)
        ax.hist(nsy, bins=60, alpha=0.65, color=PAL["noisy"], label=f"Noisy ({len(nsy):,})", density=True)
        ax.legend()
    else:
        ax.hist(losses, bins=80, color=PAL["model1"], alpha=0.8, density=True)

    if gmm_params:
        from scipy.stats import norm as scipy_norm
        x = np.linspace(losses.min(), losses.max(), 400)
        for (mu, sig, pi, col, lbl) in [
            (gmm_params.get("mu_clean", 0),  gmm_params.get("sigma_clean", 1),
             gmm_params.get("pi_clean", 0.5), PAL["clean"], "GMM clean"),
            (gmm_params.get("mu_noisy", 2),  gmm_params.get("sigma_noisy", 1),
             1 - gmm_params.get("pi_clean", 0.5), PAL["noisy"], "GMM noisy"),
        ]:
            ax.plot(x, pi * scipy_norm.pdf(x, mu, sig), "--", color=col, lw=2, label=lbl)
        ax.legend(fontsize=9)

    ax.set_title(f"Per-Sample Loss Distribution{suffix}")
    ax.set_xlabel("Loss"); ax.set_ylabel("Density"); ax.grid(True, alpha=0.4)

    ax2 = axes[1]
    if is_noisy is not None:
        parts = ax2.violinplot([losses[is_noisy == 0], losses[is_noisy == 1]],
                               positions=[0, 1], showmeans=True, showmedians=True)
        for pc, col in zip(parts["bodies"], [PAL["clean"], PAL["noisy"]]):
            pc.set_facecolor(col); pc.set_alpha(0.7)
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Clean", "Noisy"])
    else:
        ax2.violinplot([losses], positions=[0], showmeans=True, showmedians=True)
        ax2.set_xticks([0]); ax2.set_xticklabels(["All"])
    ax2.set_title(f"Loss Violin{suffix}"); ax2.set_ylabel("Loss"); ax2.grid(True, alpha=0.4, axis="y")

    fig.suptitle("DivideMix — Loss Distribution Analysis", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save(fig, output_dir, f"loss_distribution", dpi)


# ── 5. Embeddings ──────────────────────────────────────────────────────────────
def plot_embeddings(
    embeddings: np.ndarray, labels: np.ndarray, class_names: List[str],
    output_dir: str, method: str = "both",
    sample_size: int = 600, dpi: int = 150,
    languages: Optional[np.ndarray] = None,
    umap_n_neighbors: int = 15, umap_min_dist: float = 0.1,
) -> str:
    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[idx]; labels = labels[idx]
        if languages is not None: languages = languages[idx]

    colors = [PAL.get(cn, f"C{i}") for i, cn in enumerate(class_names)]
    label_colors = [colors[l] for l in labels]

    n_plots = 2 if method == "both" else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))
    if n_plots == 1: axes = [axes]
    fig.suptitle("BERT Embedding Visualisation", fontsize=14)

    plot_idx = 0
    if method in ("pca", "both"):
        pca    = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)
        _scatter(axes[plot_idx], coords, labels, colors, class_names, languages)
        var = pca.explained_variance_ratio_
        axes[plot_idx].set_title(f"PCA ({var[0]:.1%}+{var[1]:.1%}={sum(var):.1%})")
        axes[plot_idx].set_xlabel("PC1"); axes[plot_idx].set_ylabel("PC2")
        plot_idx += 1

    if method in ("umap", "both"):
        ax = axes[plot_idx]
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist,
                                n_components=2, random_state=42)
            coords  = reducer.fit_transform(embeddings)
            _scatter(ax, coords, labels, colors, class_names, languages)
            ax.set_title("UMAP"); ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        except ImportError:
            ax.text(0.5, 0.5, "pip install umap-learn", ha="center", va="center",
                    transform=ax.transAxes, color="red")
            ax.set_title("UMAP (unavailable)")

    for ax in axes: ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, output_dir, "embeddings", dpi)


def _scatter(ax, coords, labels, colors, class_names, languages):
    for lbl in np.unique(labels):
        m = labels == lbl
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=colors[lbl] if lbl < len(colors) else f"C{lbl}",
                   label=class_names[lbl], alpha=0.7, s=20, edgecolors="none")
    ax.legend(fontsize=9, frameon=True)


# ── 6. Per-language breakdown ─────────────────────────────────────────────────
def plot_per_language(per_lang: Dict, output_dir: str, dpi: int = 150) -> str:
    if not per_lang:
        return ""
    langs = list(per_lang.keys())
    accs  = [per_lang[l]["accuracy"]   for l in langs]
    f1s   = [per_lang[l]["f1_weighted"] for l in langs]
    ns    = [per_lang[l]["n"]           for l in langs]

    x = np.arange(len(langs)); width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#2E75B6", alpha=0.85)
    b2 = ax.bar(x + width/2, f1s,  width, label="F1 Weighted", color="#70AD47", alpha=0.85)
    for bar, val in [(b, v) for blist, vlist in [(b1, accs), (b2, f1s)] for b, v in zip(blist, vlist)]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for i, (lang, n) in enumerate(zip(langs, ns)):
        ax.text(i, -0.09, f"n={n:,}", ha="center", fontsize=9, color="gray")
    ax.set_xticks(x); ax.set_xticklabels(langs, fontsize=12)
    ax.set_ylim([0, 1.15]); ax.set_ylabel("Score")
    ax.set_title("Per-Language Performance", fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.4, axis="y")
    fig.tight_layout()
    return _save(fig, output_dir, "per_language_performance", dpi)


# ── Master runner ──────────────────────────────────────────────────────────────
def run_all_visualizations(
    history:           Dict,
    test_metrics:      Dict,
    per_sample_losses: Optional[np.ndarray],
    is_noisy:          Optional[np.ndarray],
    embeddings:        Optional[np.ndarray],
    true_labels:       Optional[np.ndarray],
    languages:         Optional[np.ndarray],
    cfg,
) -> List[str]:
    od  = cfg.viz.output_dir
    dpi = cfg.viz.dpi
    cns = cfg.data.class_names
    saved = []
    print("\n[Visualization] Generating all plots …")

    saved.append(plot_training_curves(history, od, dpi))

    if "confusion_matrix" in test_metrics:
        saved.append(plot_confusion_matrix(
            np.array(test_metrics["confusion_matrix"]), cns, od,
            title=f"Confusion Matrix — Test  (Acc={test_metrics['accuracy']:.3f})", dpi=dpi,
        ))

    saved.append(plot_per_class_performance(test_metrics, cns, od, dpi))

    if per_sample_losses is not None:
        saved.append(plot_loss_distribution(
            per_sample_losses, is_noisy=is_noisy, output_dir=od, epoch="final", dpi=dpi,
        ))

    if embeddings is not None and true_labels is not None:
        saved.append(plot_embeddings(
            embeddings, true_labels, cns, od,
            method="both", sample_size=cfg.viz.embedding_sample_size,
            languages=languages, dpi=dpi,
            umap_n_neighbors=cfg.viz.umap_n_neighbors,
            umap_min_dist=cfg.viz.umap_min_dist,
        ))

    if "per_language" in test_metrics:
        saved.append(plot_per_language(test_metrics["per_language"], od, dpi))

    nr_vals = [v for v in history.get("estimated_noise_rate", []) if v is not None]
    if nr_vals:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(range(1, len(nr_vals)+1), nr_vals, color="#9467BD", lw=2.5, marker="o", ms=5)
        ax.axhline(y=cfg.data.noise_rate, color="red", lw=2, ls="--",
                   label=f"True noise ({cfg.data.noise_rate:.0%})")
        ax.fill_between(range(1, len(nr_vals)+1), nr_vals, cfg.data.noise_rate,
                        alpha=0.15, color="#9467BD")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Noise Rate")
        ax.set_title("GMM Noise Rate Estimate vs. True Rate", fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.4)
        fig.tight_layout()
        saved.append(_save(fig, od, "noise_rate_history", dpi))

    print(f"[Visualization] {len(saved)} plots saved → {od}")
    return saved
