"""
training/trainer.py
End-to-end training loop that combines:
  Co-Teaching    → two models select clean examples for each other
  Small-Loss     → per-batch noise filtering (annealed keep ratio)
  DivideMix GMM  → epoch-level loss analysis to estimate noise rate
  Bootstrapping  → gradual label correction via EMA predictions
  SCE / GCE / MAE / CE  → noise-robust loss functions
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from robust_losses import (
    build_loss, BootstrappingLoss, compute_per_sample_loss
)
from noise_strategies import (
    SmallLossTrick, CoTeaching,
    GaussianMixtureNoiseSeparator,
    LabelRefurbishmentStore, NoiseRateEstimator,
)
from metrics import evaluate_model


# ── Logger ─────────────────────────────────────────────────────────────────────
class TrainingLogger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.path    = os.path.join(log_dir, "training_log.json")
        self.history: Dict[str, List] = {
            "train_loss_m1": [], "train_loss_m2": [],
            "val_loss": [], "val_acc": [], "val_f1": [],
            "estimated_noise_rate": [], "keep_ratio": [],
            "forget_rate": [], "epoch_time": [],
        }

    def log(self, d: Dict):
        for k, v in d.items():
            if k in self.history:
                self.history[k].append(float(v) if v is not None else None)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.history, f, indent=2)


# ── Trainer ─────────────────────────────────────────────────────────────────────
class Trainer:
    """
    Full training pipeline.  Pass two BERTClassifier instances for Co-Teaching,
    or the same one twice with use_co_teaching=False to run single-model training.
    """

    def __init__(self, cfg, model1, model2, data_module, device: torch.device):
        self.cfg         = cfg
        self.tcfg        = cfg.training
        self.model1      = model1
        self.model2      = model2
        self.data_module = data_module
        self.device      = device

        # ── Loss functions ───────────────────────────────────────────────────
        self.loss_fn      = build_loss(cfg)
        self.boot_loss    = BootstrappingLoss(
            cfg.data.num_classes, self.tcfg.bootstrap_beta, "none"
        )

        # ── Noise strategies ─────────────────────────────────────────────────
        n_train            = len(data_module.train_df)
        self.small_loss    = SmallLossTrick(
            self.tcfg.keep_ratio_initial, self.tcfg.keep_ratio_final,
            self.tcfg.small_loss_start,   self.tcfg.num_epochs,
        )
        self.co_teach      = CoTeaching(
            self.tcfg.forget_rate, self.tcfg.num_gradual,
            self.tcfg.num_epochs,  self.tcfg.exponent,
        )
        self.gmm           = GaussianMixtureNoiseSeparator(self.tcfg.p_threshold)
        self.refurb        = LabelRefurbishmentStore(n_train, cfg.data.num_classes)
        self.noise_est     = NoiseRateEstimator()

        # ── Optimisers (built in setup, need steps_per_epoch) ─────────────────
        self.opt1 = self.opt2 = None
        self.sched1 = self.sched2 = None

        # ── State ─────────────────────────────────────────────────────────────
        self.logger         = TrainingLogger(self.tcfg.log_dir)
        self.best_val_f1    = 0.0
        self.best_epoch     = 0
        self._epoch_losses: List[float] = []   # per-sample losses accumulated in epoch

    # ── Optimisers / schedulers ────────────────────────────────────────────────
    def _build_optimiser(self, model):
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        params   = [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": self.tcfg.weight_decay},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        return AdamW(params, lr=self.tcfg.learning_rate)

    def _build_schedulers(self, steps_per_epoch: int):
        total = self.tcfg.num_epochs * steps_per_epoch
        def _s(opt):
            return torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=self.tcfg.learning_rate, total_steps=total,
                pct_start=self.tcfg.warmup_ratio, anneal_strategy="cos",
            )
        self.opt1 = self._build_optimiser(self.model1)
        self.sched1 = _s(self.opt1)
        if self.tcfg.use_co_teaching:
            self.opt2 = self._build_optimiser(self.model2)
            self.sched2 = _s(self.opt2)

    # ── One Co-Teaching step ──────────────────────────────────────────────────
    def _step(self, batch1: Dict, batch2: Dict, epoch: int) -> Tuple[float, float]:
        def _tensors(b):
            return (b["input_ids"].to(self.device),
                    b["attention_mask"].to(self.device),
                    b["token_type_ids"].to(self.device),
                    b["label"].to(self.device),
                    b["index"])

        ids1, msk1, tt1, lbl1, idx1 = _tensors(batch1)
        ids2, msk2, tt2, lbl2, idx2 = _tensors(batch2)

        # ── Selection pass (no grad) ─────────────────────────────────────────
        self.model1.eval(); self.model2.eval()
        with torch.no_grad():
            lo1 = self.model1(ids1, msk1, tt1)["logits"]
            lo2 = self.model2(ids2, msk2, tt2)["logits"]

        ps1 = compute_per_sample_loss(lo1, lbl1)
        ps2 = compute_per_sample_loss(lo2, lbl2)
        self._epoch_losses.extend(ps1.cpu().numpy().tolist())

        ci1 = self.co_teach.select(ps1, epoch)   # model1's clean indices from batch1
        ci2 = self.co_teach.select(ps2, epoch)   # model2's clean indices from batch2

        # ── Training pass ────────────────────────────────────────────────────
        self.model1.train(); self.model2.train()
        use_boot = (epoch >= self.tcfg.bootstrap_start_epoch
                    and self.tcfg.use_bootstrapping)
        loss_fn  = self.boot_loss if use_boot else self.loss_fn

        if not self.tcfg.use_co_teaching:
            self.opt1.zero_grad()
            out1 = self.model1(ids1, msk1, tt1)
            if use_boot:
                soft1 = F.softmax(out1["logits"], dim=1)
                self.refurb.update(idx1, soft1)
            l1 = loss_fn(out1["logits"], lbl1).mean()
            l1.backward()
            nn.utils.clip_grad_norm_(self.model1.parameters(), self.tcfg.gradient_clip)
            self.opt1.step()
            if self.sched1:
                self.sched1.step()
            return l1.item(), l1.item()

        # Model 1 learns from model2's clean selection (from batch2)
        self.opt1.zero_grad()
        out1  = self.model1(ids2[ci2], msk2[ci2], tt2[ci2])
        if use_boot:
            soft1 = F.softmax(out1["logits"], dim=1)
            self.refurb.update(idx2[ci2.cpu()], soft1)
        l1 = loss_fn(out1["logits"], lbl2[ci2]).mean()
        l1.backward()
        nn.utils.clip_grad_norm_(self.model1.parameters(), self.tcfg.gradient_clip)
        self.opt1.step()
        if self.sched1: self.sched1.step()

        # Model 2 learns from model1's clean selection (from batch1)
        self.opt2.zero_grad()
        out2  = self.model2(ids1[ci1], msk1[ci1], tt1[ci1])
        l2 = loss_fn(out2["logits"], lbl1[ci1]).mean()
        l2.backward()
        nn.utils.clip_grad_norm_(self.model2.parameters(), self.tcfg.gradient_clip)
        self.opt2.step()
        if self.sched2: self.sched2.step()

        return l1.item(), l2.item()

    # ── Main training loop ────────────────────────────────────────────────────
    def train(self) -> Dict:
        print("\n" + "=" * 65)
        print("TRAINING START")
        print("=" * 65)

        loader1, loader2 = self.data_module.get_paired_train_loaders()
        val_loader       = self.data_module.get_val_loader()
        steps            = min(len(loader1), len(loader2))
        self._build_schedulers(steps)

        for epoch in range(self.tcfg.num_epochs):
            t0 = time.time()
            self._epoch_losses.clear()

            # ── Training ──────────────────────────────────────────────────────
            sum1 = sum2 = n = 0
            pbar = tqdm(zip(loader1, loader2), total=steps,
                        desc=f"  Epoch {epoch+1:02d}/{self.tcfg.num_epochs}", ncols=88)
            for b1, b2 in pbar:
                l1, l2 = self._step(b1, b2, epoch)
                sum1 += l1;  sum2 += l2;  n += 1
                if n % self.tcfg.log_every_n_steps == 0:
                    pbar.set_postfix({"L1": f"{sum1/n:.4f}", "L2": f"{sum2/n:.4f}",
                                      "FR": f"{self.co_teach.get_forget_rate(epoch):.2f}"})

            avg1, avg2 = sum1 / max(n, 1), sum2 / max(n, 1)

            # ── GMM noise analysis ────────────────────────────────────────────
            gmm_stats = {}
            if self.tcfg.use_divide_mix and len(self._epoch_losses) > 20:
                arr      = np.array(self._epoch_losses)
                gmm_stats, _, _ = self.noise_est.estimate(arr)
                nr = gmm_stats.get("estimated_noise_rate", 0)
                print(f"  [GMM]  est_noise={nr:.2%}  "
                      f"clean={gmm_stats['num_clean']:,}  noisy={gmm_stats['num_noisy']:,}")

            # ── Bootstrapping beta annealing ──────────────────────────────────
            if self.tcfg.use_bootstrapping:
                start = self.tcfg.bootstrap_start_epoch
                if epoch >= start:
                    prog = (epoch - start) / max(1, self.tcfg.num_epochs - start)
                    self.boot_loss.update_beta(0.5 + 0.4 * prog)

            # ── Validation ───────────────────────────────────────────────────
            vm = evaluate_model(self.model1, val_loader, self.device,
                                self.cfg, split="val", verbose=False)
            val_loss = vm["loss"]
            val_acc  = vm["accuracy"]
            val_f1   = vm["f1_weighted"]

            # ── Log ──────────────────────────────────────────────────────────
            et = time.time() - t0
            self.logger.log({
                "train_loss_m1": avg1, "train_loss_m2": avg2,
                "val_loss": val_loss,  "val_acc": val_acc, "val_f1": val_f1,
                "estimated_noise_rate": gmm_stats.get("estimated_noise_rate"),
                "keep_ratio": self.small_loss.get_keep_ratio(epoch),
                "forget_rate": self.co_teach.get_forget_rate(epoch),
                "epoch_time": et,
            })

            print(f"  L1={avg1:.4f}  L2={avg2:.4f}  "
                  f"Val Acc={val_acc:.4f}  Val F1={val_f1:.4f}  t={et:.1f}s")

            # ── Checkpoint ───────────────────────────────────────────────────
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch  = epoch + 1
                self._save()
                print(f"  *** Best model saved  (F1={val_f1:.4f}) ***")

        self.logger.save()
        print(f"\n[Train] Done.  Best epoch={self.best_epoch}  "
              f"Best Val F1={self.best_val_f1:.4f}")
        return self.logger.history

    # ── Checkpoint helpers ────────────────────────────────────────────────────
    def _save(self):
        os.makedirs(self.cfg.model.checkpoint_dir, exist_ok=True)
        torch.save({
            "epoch":         self.best_epoch,
            "model1_state":  self.model1.state_dict(),
            "model2_state":  self.model2.state_dict(),
            "best_val_f1":   self.best_val_f1,
        }, os.path.join(self.cfg.model.checkpoint_dir, "best_model.pt"))

    def load_best(self):
        p = os.path.join(self.cfg.model.checkpoint_dir, "best_model.pt")
        if not os.path.exists(p):
            print("[Checkpoint] No saved model found.")
            return
        ckpt = torch.load(p, map_location=self.device)
        self.model1.load_state_dict(ckpt["model1_state"])
        self.model2.load_state_dict(ckpt["model2_state"])
        print(f"[Checkpoint] Loaded — epoch {ckpt['epoch']}  "
              f"Val F1={ckpt['best_val_f1']:.4f}")
