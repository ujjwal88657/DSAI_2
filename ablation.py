"""
experiments/ablation.py
Ablation study: 8 method configs × 4 noise rates on the real dataset.
Uses TF-IDF + MLP (fast) to compare methods without GPU overhead.

Run:  python experiments/ablation.py
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import apply_noise, clean_text, inject_asymmetric_noise
from robust_losses import (
    CrossEntropyLoss, SymmetricCrossEntropyLoss,
    GeneralizedCrossEntropyLoss, MAELoss, BootstrappingLoss,
    compute_per_sample_loss,
)
from noise_strategies import CoTeaching
from metrics import compute_metrics, compute_loss
from helpers import set_seed


# ── Lightweight dataset & model ───────────────────────────────────────────────
class SimpleDS(Dataset):
    def __init__(self, X, labels, is_noisy=None):
        self.X      = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.noisy  = torch.tensor(is_noisy if is_noisy is not None else [0]*len(labels), dtype=torch.long)
        self.idx    = torch.arange(len(labels))
    def __len__(self):    return len(self.labels)
    def __getitem__(self, i):
        return {"features": self.X[i], "label": self.labels[i],
                "is_noisy": self.noisy[i], "index": self.idx[i]}


class MLP(nn.Module):
    def __init__(self, inp, nc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, nc),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x): return {"logits": self.net(x)}


def _run(X_tr, y_tr, noisy_mask, X_val, y_val, X_te, y_te,
         loss_type, co_teach, bootstrap, nc=2, epochs=10, bs=64, lr=3e-3,
         seed=42, device=torch.device("cpu")):
    set_seed(seed)
    tr_ds = SimpleDS(X_tr, y_tr, noisy_mask)
    ld1   = DataLoader(tr_ds, batch_size=bs, shuffle=True,  drop_last=True)
    ld2   = DataLoader(tr_ds, batch_size=bs, shuffle=True,  drop_last=True)
    val_l = DataLoader(SimpleDS(X_val, y_val), batch_size=256, shuffle=False)
    te_l  = DataLoader(SimpleDS(X_te,  y_te),  batch_size=256, shuffle=False)

    m1 = MLP(X_tr.shape[1], nc).to(device)
    m2 = MLP(X_tr.shape[1], nc).to(device)
    lmap = {"ce": CrossEntropyLoss(nc,"none"), "sce": SymmetricCrossEntropyLoss(nc,0.1,1.0,"none"),
            "gce": GeneralizedCrossEntropyLoss(nc,0.7,"none"), "mae": MAELoss(nc,"none")}
    lf   = lmap[loss_type]
    blf  = BootstrappingLoss(nc, 0.8, "none")
    cot  = CoTeaching(0.20, 5, epochs)
    o1   = torch.optim.AdamW(m1.parameters(), lr=lr, weight_decay=1e-4)
    o2   = torch.optim.AdamW(m2.parameters(), lr=lr, weight_decay=1e-4)
    s1   = torch.optim.lr_scheduler.CosineAnnealingLR(o1, epochs)
    s2   = torch.optim.lr_scheduler.CosineAnnealingLR(o2, epochs)
    boot_start = 5

    for ep in range(epochs):
        m1.train(); m2.train()
        for b1, b2 in zip(ld1, ld2):
            f1=b1["features"].to(device); l1=b1["label"].to(device)
            f2=b2["features"].to(device); l2=b2["label"].to(device)
            if co_teach:
                with torch.no_grad():
                    lo1=m1(f1)["logits"]; lo2=m2(f2)["logits"]
                ci1=cot.select(compute_per_sample_loss(lo1,l1),ep)
                ci2=cot.select(compute_per_sample_loss(lo2,l2),ep)
                fn = blf if ep>=boot_start and bootstrap else lf
                o1.zero_grad(); (fn(m1(f2[ci2])["logits"],l2[ci2]).mean()).backward()
                nn.utils.clip_grad_norm_(m1.parameters(),1.0); o1.step()
                o2.zero_grad(); (fn(m2(f1[ci1])["logits"],l1[ci1]).mean()).backward()
                nn.utils.clip_grad_norm_(m2.parameters(),1.0); o2.step()
            else:
                fn = blf if ep>=boot_start and bootstrap else lf
                o1.zero_grad(); (fn(m1(f1)["logits"],l1).mean()).backward()
                nn.utils.clip_grad_norm_(m1.parameters(),1.0); o1.step()
        s1.step(); s2.step()

    m1.eval()
    preds,probs,labels=[],[],[]
    with torch.no_grad():
        for b in te_l:
            out=m1(b["features"].to(device))
            p=F.softmax(out["logits"],1)
            preds.extend(p.argmax(1).cpu().numpy())
            probs.extend(p.cpu().numpy()); labels.extend(b["label"].numpy())
    return compute_metrics(np.array(preds), np.array(labels))


# ── Load real data ────────────────────────────────────────────────────────────
def _load_data(data_path: str):
    import pandas as pd
    df = pd.read_csv(data_path).rename(columns={"hate_label": "label"})
    from dataset import clean_text
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 3].reset_index(drop=True)
    df["label"]          = df["label"].astype(int)
    df["original_label"] = df["label"]
    df["is_noisy"]       = 0
    if "text_length" not in df.columns:
        df["text_length"] = df["text"].str.len()
    return df


def run_ablation(data_path: str, output_dir: str = "./ablation_results"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Ablation] Device: {device}")

    df = _load_data(data_path)
    tr_val, te = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
    tr, va     = train_test_split(tr_val, test_size=0.15/0.85, stratify=tr_val["label"], random_state=42)
    y_clean = tr["label"].values
    y_val   = va["label"].values
    y_test  = te["label"].values

    vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2), sublinear_tf=True)
    X_tr  = vec.fit_transform(tr["text"]).toarray()
    X_val = vec.transform(va["text"]).toarray()
    X_te  = vec.transform(te["text"]).toarray()
    print(f"[Ablation] TF-IDF shape: {X_tr.shape}")

    noise_rates = [0.0, 0.15, 0.30, 0.45]
    configs = [
        {"name": "CE Baseline",          "loss":"ce",  "co":False,"boot":False},
        {"name": "GCE",                  "loss":"gce", "co":False,"boot":False},
        {"name": "MAE",                  "loss":"mae", "co":False,"boot":False},
        {"name": "SCE",                  "loss":"sce", "co":False,"boot":False},
        {"name": "SCE+Bootstrap",        "loss":"sce", "co":False,"boot":True},
        {"name": "CE+CoTeaching",        "loss":"ce",  "co":True, "boot":False},
        {"name": "SCE+CoTeaching",       "loss":"sce", "co":True, "boot":False},
        {"name": "SCE+CoTeach+Boot",     "loss":"sce", "co":True, "boot":True},
    ]

    all_res, rows = {}, []
    print(f"\n[Ablation] {len(configs)} configs × {len(noise_rates)} noise rates\n")

    for nr in noise_rates:
        print(f"\n{'─'*60}\n  NOISE RATE = {nr:.0%}\n{'─'*60}")
        if nr > 0:
            from dataset import inject_asymmetric_noise
            y_noisy, mask = inject_asymmetric_noise(y_clean, nr, 2, seed=42)
        else:
            y_noisy, mask = y_clean.copy(), np.zeros(len(y_clean), dtype=bool)

        for cfg in configs:
            t0 = time.time()
            print(f"  {cfg['name']:<32}", end="", flush=True)
            try:
                m = _run(X_tr, y_noisy, mask.astype(int), X_val, y_val, X_te, y_test,
                         cfg["loss"], cfg["co"], cfg["boot"], epochs=10, device=device)
                acc, f1 = m["accuracy"], m["f1_weighted"]
                print(f"Acc={acc:.4f}  F1={f1:.4f}  ({time.time()-t0:.1f}s)")
                all_res[f"{cfg['name']}@nr{int(nr*100)}"] = {"config":cfg["name"],"noise_rate":nr,"accuracy":acc,"f1_weighted":f1}
                rows.append({"Method":cfg["name"],"Noise":f"{nr:.0%}","Acc":f"{acc:.4f}","F1":f"{f1:.4f}"})
            except Exception as e:
                print(f"ERROR: {e}")
                rows.append({"Method":cfg["name"],"Noise":f"{nr:.0%}","Acc":"ERR","F1":"ERR"})

    with open(os.path.join(output_dir,"ablation_results.json"),"w") as f:
        json.dump(all_res, f, indent=2)

    print(f"\n{'='*70}\nABLATION SUMMARY\n{'='*70}")
    print(f"{'Method':<32}{'Noise':>8}{'Acc':>10}{'F1':>10}")
    print("─"*70)
    prev_noise = None
    for r in rows:
        if r["Noise"] != prev_noise:
            if prev_noise: print("─"*70)
            prev_noise = r["Noise"]
        print(f"{r['Method']:<32}{r['Noise']:>8}{r['Acc']:>10}{r['F1']:>10}")
    print("="*70)

    # Plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = plt.cm.tab10(np.linspace(0,1,len(configs)))
        for metric, ax in [("accuracy","Accuracy"),("f1_weighted","F1 Weighted")]:
            for cfg, col in zip(configs, colors):
                ys = [all_res.get(f"{cfg['name']}@nr{int(nr*100)}",{}).get(metric) for nr in noise_rates]
                valid_x = [nr*100 for nr,y in zip(noise_rates,ys) if y is not None]
                valid_y = [y for y in ys if y is not None]
                if valid_y:
                    axes[["accuracy","f1_weighted"].index(metric)].plot(
                        valid_x, valid_y, marker="o", lw=2, label=cfg["name"], color=col)
        for i, (ax, ttl) in enumerate(zip(axes,["Accuracy","F1 Weighted"])):
            ax.set_xlabel("Noise Rate (%)"); ax.set_ylabel(ttl)
            ax.set_title(f"{ttl} vs Noise Rate"); ax.set_ylim([0.4,1.02])
            ax.set_xticks([nr*100 for nr in noise_rates])
            ax.grid(True, alpha=0.4); ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        path = os.path.join(output_dir,"ablation_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"\n[Ablation] Plot: {path}")
    except Exception as e:
        print(f"[Ablation] Plot error: {e}")

    return all_res


if __name__ == "__main__":
    from config import DATASET_PATH
    run_ablation(DATASET_PATH, output_dir="./ablation_results")
