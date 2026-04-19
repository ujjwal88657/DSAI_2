"""
demo.py
Fast demo on the REAL combined_hate_speech_dataset.csv (29 550 rows)
TF-IDF (sparse, densified per-batch) + MLP  — ~60–90s on CPU, no GPU needed.

Implements ALL noise-handling strategies:
  SCE · Co-Teaching · DivideMix GMM · Bootstrapping · Small-Loss
"""

import os, sys, json, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_PATH
from data.dataset import clean_text, apply_noise
from losses.robust_losses import SymmetricCrossEntropyLoss, BootstrappingLoss, compute_per_sample_loss
from training.noise_strategies import CoTeaching, GaussianMixtureNoiseSeparator, SmallLossTrick, NoiseRateEstimator
from evaluation.metrics import compute_metrics, compute_loss, compute_per_language_metrics
from visualization.plots import (plot_training_curves, plot_confusion_matrix,
    plot_per_class_performance, plot_loss_distribution, plot_embeddings, plot_per_language)
from utils.helpers import set_seed, save_json

# ── Config ───────────────────────────────────────────────────────────────────
class Cfg:
    dataset_path     = DATASET_PATH
    num_classes      = 2
    class_names      = ["not_hate", "hate"]
    noise_rate       = 0.30
    noise_type       = "asymmetric"
    noise_seed       = 42
    random_seed      = 42
    num_epochs       = 12
    batch_size       = 128       # small batches, dense per-batch only
    lr               = 3e-3
    forget_rate      = 0.20
    num_gradual      = 5
    sce_alpha        = 0.10
    sce_beta         = 1.00
    bootstrap_beta   = 0.80
    bootstrap_start  = 4
    p_threshold      = 0.50
    keep_ratio_init  = 1.0
    keep_ratio_final = 0.70
    small_loss_start = 4
    tfidf_features   = 3000      # keep sparse → only 2 MB
    viz_dir          = "./visualizations"
    log_dir          = "./logs"

CFG = Cfg()
os.makedirs(CFG.viz_dir, exist_ok=True)
os.makedirs(CFG.log_dir, exist_ok=True)
set_seed(CFG.random_seed)


# ── Sparse-aware Dataset (densify per-batch, never store full dense matrix) ──
class SparseDS(Dataset):
    def __init__(self, X_sp, labels, is_noisy=None, indices=None, languages=None):
        self.X      = X_sp                    # scipy CSR — stays sparse in RAM
        self.labels = np.array(labels, dtype=np.int64)
        self.noisy  = np.array(is_noisy  if is_noisy  is not None else [0]*len(labels), dtype=np.int64)
        self.idx    = np.array(indices   if indices   is not None else list(range(len(labels))), dtype=np.int64)
        self.langs  = languages if languages is not None else ["unknown"]*len(labels)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        # Densify one row at a time — tiny memory footprint
        row = torch.tensor(self.X[i].toarray().squeeze(0), dtype=torch.float32)
        return {
            "features": row,
            "label":    torch.tensor(self.labels[i], dtype=torch.long),
            "is_noisy": torch.tensor(self.noisy[i],  dtype=torch.long),
            "index":    torch.tensor(self.idx[i],    dtype=torch.long),
            "language": self.langs[i],
        }


# ── MLP with bottleneck embedding layer ───────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, inp, nc):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(inp, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(256, 64),  nn.GELU(),
        )
        self.head = nn.Linear(64, nc)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, features, return_embeddings=False):
        emb = self.enc(features)
        out = {"logits": self.head(emb)}
        if return_embeddings:
            out["embeddings"] = emb.detach()
        return out


# ── Data prep ─────────────────────────────────────────────────────────────────
def prepare():
    import pandas as pd
    print("\n[Data] Loading real dataset …")
    df = pd.read_csv(CFG.dataset_path).rename(columns={"hate_label": "label"})
    df["text"]    = df["text"].apply(clean_text)
    df            = df[df["text"].str.len() > 3].reset_index(drop=True)
    df["label"]   = df["label"].astype(int)
    if "language"    not in df.columns: df["language"]    = "unknown"
    if "text_length" not in df.columns: df["text_length"] = df["text"].str.len()
    df["original_label"] = df["label"]
    df["is_noisy"]       = 0

    print(f"  {len(df):,} rows  |  not_hate={(df['label']==0).sum():,}  hate={(df['label']==1).sum():,}")
    print(f"  Languages: {df['language'].value_counts().to_dict()}")

    tr_val, te = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=CFG.random_seed)
    tr, va     = train_test_split(tr_val, test_size=0.15/0.85, stratify=tr_val["label"], random_state=CFG.random_seed)

    tr = apply_noise(tr.copy(), CFG.noise_type, CFG.noise_rate, CFG.num_classes, CFG.noise_seed)
    for split in [va, te]:
        split["original_label"] = split["label"]
        split["is_noisy"]       = 0
    tr = tr.reset_index(drop=True); va = va.reset_index(drop=True); te = te.reset_index(drop=True)
    print(f"  Train={len(tr):,}  Val={len(va):,}  Test={len(te):,}  noise={tr['is_noisy'].mean():.2%}")

    print(f"[Data] TF-IDF max_features={CFG.tfidf_features} (sparse) …")
    vec   = TfidfVectorizer(max_features=CFG.tfidf_features, ngram_range=(1, 2), sublinear_tf=True)
    X_tr  = vec.fit_transform(tr["text"])
    X_va  = vec.transform(va["text"])
    X_te  = vec.transform(te["text"])
    print(f"  Feature dim={X_tr.shape[1]}  sparse train={X_tr.data.nbytes//1024//1024} MB")

    mk = lambda df_, Xsp: SparseDS(Xsp, df_["label"].tolist(),
            df_["is_noisy"].tolist(), df_.index.tolist(), df_["language"].tolist())
    return mk(tr,X_tr), mk(va,X_va), mk(te,X_te), X_tr.shape[1], tr, va, te


# ── Eval helpers ───────────────────────────────────────────────────────────────
@torch.no_grad()
def run_eval(model, loader, device, return_emb=False):
    model.eval()
    preds, probs, labels, embs, langs = [], [], [], [], []
    for b in loader:
        out = model(b["features"].to(device), return_embeddings=return_emb)
        p   = F.softmax(out["logits"], 1)
        preds.extend(p.argmax(1).cpu().numpy())
        probs.extend(p.cpu().numpy()); labels.extend(b["label"].numpy())
        if return_emb and "embeddings" in out: embs.append(out["embeddings"].cpu().numpy())
        langs.extend(b["language"] if isinstance(b["language"], list) else list(b["language"]))
    return (np.array(preds), np.array(probs), np.array(labels),
            np.concatenate(embs) if embs else None, langs)

@torch.no_grad()
def get_losses(model, loader, device):
    model.eval()
    losses, noisy = [], []
    for b in loader:
        l = F.cross_entropy(model(b["features"].to(device))["logits"],
                            b["label"].to(device), reduction="none")
        losses.extend(l.cpu().numpy()); noisy.extend(b["is_noisy"].numpy())
    return np.array(losses), np.array(noisy)


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    tr_ds, va_ds, te_ds, inp_dim, tr_df, va_df, te_df = prepare()

    lkw   = dict(batch_size=CFG.batch_size, num_workers=0, pin_memory=False)
    ld1   = DataLoader(tr_ds, shuffle=True,  drop_last=True, **lkw)
    ld2   = DataLoader(tr_ds, shuffle=True,  drop_last=True, **lkw)
    va_ld = DataLoader(va_ds, shuffle=False, **lkw)
    te_ld = DataLoader(te_ds, shuffle=False, **lkw)
    tr_seq= DataLoader(tr_ds, shuffle=False, **lkw)

    model1 = MLP(inp_dim, CFG.num_classes).to(device)
    model2 = MLP(inp_dim, CFG.num_classes).to(device)
    print(f"[Model] MLP — input={inp_dim}  512→256→64→{CFG.num_classes}")

    sce_fn  = SymmetricCrossEntropyLoss(CFG.num_classes, CFG.sce_alpha, CFG.sce_beta, "none")
    boot_fn = BootstrappingLoss(CFG.num_classes, CFG.bootstrap_beta, "none")
    cot     = CoTeaching(CFG.forget_rate, CFG.num_gradual, CFG.num_epochs)
    slt     = SmallLossTrick(CFG.keep_ratio_init, CFG.keep_ratio_final, CFG.small_loss_start, CFG.num_epochs)
    nre     = NoiseRateEstimator()

    opt1 = torch.optim.AdamW(model1.parameters(), lr=CFG.lr, weight_decay=1e-4)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=CFG.lr, weight_decay=1e-4)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, CFG.num_epochs)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, CFG.num_epochs)

    history = {k: [] for k in ["train_loss_m1","train_loss_m2","val_loss","val_acc","val_f1",
                                "estimated_noise_rate","keep_ratio","forget_rate","epoch_time"]}
    best_f1, best_state = 0.0, None

    print(f"\n[Training] {CFG.num_epochs} epochs | batch={CFG.batch_size} | "
          f"noise={CFG.noise_type}@{CFG.noise_rate:.0%}\n")

    for epoch in range(CFG.num_epochs):
        t0 = time.time()
        model1.train(); model2.train()
        sum1 = sum2 = n = 0
        use_boot = epoch >= CFG.bootstrap_start

        for b1, b2 in zip(ld1, ld2):
            f1=b1["features"].to(device); l1=b1["label"].to(device)
            f2=b2["features"].to(device); l2=b2["label"].to(device)

            with torch.no_grad():
                lo1=model1(f1)["logits"]; lo2=model2(f2)["logits"]
            ci1=cot.select(compute_per_sample_loss(lo1,l1), epoch)
            ci2=cot.select(compute_per_sample_loss(lo2,l2), epoch)
            fn = boot_fn if use_boot else sce_fn

            opt1.zero_grad()
            l_1 = fn(model1(f2[ci2])["logits"], l2[ci2]).mean()
            l_1.backward()
            nn.utils.clip_grad_norm_(model1.parameters(), 1.0); opt1.step()

            opt2.zero_grad()
            l_2 = fn(model2(f1[ci1])["logits"], l1[ci1]).mean()
            l_2.backward()
            nn.utils.clip_grad_norm_(model2.parameters(), 1.0); opt2.step()

            sum1 += l_1.item(); sum2 += l_2.item(); n += 1

        sched1.step(); sched2.step()

        ep_losses, ep_noisy = get_losses(model1, tr_seq, device)
        gmm_stats, _, _     = nre.estimate(ep_losses)
        est_nr               = gmm_stats["estimated_noise_rate"]

        pv,prv,lv,_,_ = run_eval(model1, va_ld, device)
        vm   = compute_metrics(pv, lv, CFG.class_names)
        vloss= compute_loss(prv, lv)
        vacc = vm["accuracy"]; vf1 = vm["f1_weighted"]
        kr = slt.get_keep_ratio(epoch); fr = cot.get_forget_rate(epoch); et = time.time()-t0

        for k,v in [("train_loss_m1",sum1/n),("train_loss_m2",sum2/n),
                    ("val_loss",vloss),("val_acc",vacc),("val_f1",vf1),
                    ("estimated_noise_rate",est_nr),("keep_ratio",kr),("forget_rate",fr),("epoch_time",et)]:
            history[k].append(v)

        print(f"Epoch {epoch+1:02d}/{CFG.num_epochs} | L1={sum1/n:.4f} L2={sum2/n:.4f} | "
              f"Val Acc={vacc:.4f} F1={vf1:.4f} | NoiseEst={est_nr:.2%} FR={fr:.2f} | t={et:.1f}s")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model1.state_dict().items()}
            print(f"  *** Best saved (F1={vf1:.4f}) ***")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    model1.load_state_dict(best_state)
    pt,prt,lt,embs_t,langs_t = run_eval(model1, te_ld, device, return_emb=True)
    test_m = compute_metrics(pt, lt, CFG.class_names)
    test_m["loss"] = compute_loss(prt, lt)
    if langs_t:
        test_m["per_language"] = compute_per_language_metrics(pt, lt, langs_t, CFG.class_names)

    print(f"\n{'='*60}\nFINAL TEST RESULTS\n{'='*60}")
    print(test_m["classification_report"])
    print(f"  Accuracy    : {test_m['accuracy']:.4f}")
    print(f"  F1 Macro    : {test_m['f1_macro']:.4f}")
    print(f"  F1 Weighted : {test_m['f1_weighted']:.4f}")
    if "per_language" in test_m:
        print("\n  Per-language breakdown:")
        for lang, m in test_m["per_language"].items():
            print(f"    {lang:12s}  n={m['n']:5,}  acc={m['accuracy']:.4f}  f1={m['f1_weighted']:.4f}")

    fin_losses, fin_noisy = get_losses(model1, tr_seq, device)
    fin_gmm = GaussianMixtureNoiseSeparator(CFG.p_threshold)
    _, _, fin_est = fin_gmm.fit_predict(fin_losses)
    print(f"\n  GMM noise estimate : {fin_est:.2%}  (true: {CFG.noise_rate:.2%})")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[Visualization] Generating plots …")
    save_json({k: v for k, v in test_m.items() if k != "classification_report"},
              os.path.join(CFG.log_dir, "test_results.json"))
    save_json(history, os.path.join(CFG.log_dir, "training_log.json"))

    plot_training_curves(history, CFG.viz_dir)
    plot_confusion_matrix(np.array(test_m["confusion_matrix"]), CFG.class_names, CFG.viz_dir,
                          title=f"Confusion Matrix — Test  Acc={test_m['accuracy']:.3f}")
    plot_per_class_performance(test_m, CFG.class_names, CFG.viz_dir)
    plot_loss_distribution(fin_losses, is_noisy=fin_noisy, output_dir=CFG.viz_dir,
                           epoch="final", gmm_params=fin_gmm.get_stats())
    if embs_t is not None:
        plot_embeddings(embs_t, lt, CFG.class_names, CFG.viz_dir,
                        method="both", sample_size=600,
                        languages=np.array(langs_t) if langs_t else None)
    if "per_language" in test_m:
        plot_per_language(test_m["per_language"], CFG.viz_dir)

    nr_vals = history["estimated_noise_rate"]
    if nr_vals:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(range(1,len(nr_vals)+1), nr_vals, color="#9467BD", lw=2.5, marker="o", ms=5)
        ax.axhline(y=CFG.noise_rate, color="red", lw=2, ls="--", label=f"True ({CFG.noise_rate:.0%})")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Noise Rate")
        ax.set_title("GMM Noise Rate Estimate vs. True Rate"); ax.legend(); ax.grid(True, alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(CFG.viz_dir,"noise_rate_history.png"), dpi=150, bbox_inches="tight")
        plt.close(fig); print(f"  [Viz] {CFG.viz_dir}/noise_rate_history.png")

    print(f"\n{'='*60}\nDEMO COMPLETE\n{'='*60}")
    print(f"  Dataset        : {len(tr_df)+len(va_df)+len(te_df):,} real rows  (English/Hindi/Hinglish)")
    print(f"  Best Val F1    : {best_f1:.4f}")
    print(f"  Test Accuracy  : {test_m['accuracy']:.4f}")
    print(f"  Test F1 (w)    : {test_m['f1_weighted']:.4f}")
    print(f"  GMM noise est  : {fin_est:.2%}  (true {CFG.noise_rate:.2%})")
    print(f"  Plots          : {CFG.viz_dir}/")
    print(f"  Logs           : {CFG.log_dir}/")

if __name__ == "__main__":
    train()
