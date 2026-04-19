"""
main.py
Full end-to-end pipeline using bert-base-multilingual-cased on the
real combined_hate_speech_dataset.csv (29 550 rows).

Usage
-----
python main.py                                    # defaults
python main.py --epochs 10 --noise_rate 0.30      # custom
python main.py --fast                             # 3-epoch smoke test
python main.py --loss gce --no_co_teach           # ablate methods
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CFG
from data.dataset import DataModule
from models.classifier import build_dual_models
from training.trainer import Trainer
from training.noise_strategies import GaussianMixtureNoiseSeparator
from evaluation.metrics import predict, compute_metrics, compute_loss, evaluate_model
from visualization.plots import run_all_visualizations
from utils.helpers import set_seed, get_device, save_json


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Hate Speech Noisy-Label Pipeline")
    p.add_argument("--epochs",        type=int,   default=None)
    p.add_argument("--batch_size",    type=int,   default=None)
    p.add_argument("--noise_rate",    type=float, default=None)
    p.add_argument("--noise_type",    type=str,   default=None,
                   choices=["symmetric","asymmetric","instance"])
    p.add_argument("--loss",          type=str,   default=None,
                   choices=["ce","sce","gce","mae"])
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--no_co_teach",   action="store_true")
    p.add_argument("--no_divide_mix", action="store_true")
    p.add_argument("--no_bootstrap",  action="store_true")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--fast",          action="store_true",
                   help="3-epoch smoke test (tiny batch for CI)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = CFG
    if args.epochs:        cfg.training.num_epochs    = args.epochs
    if args.batch_size:    cfg.training.batch_size    = args.batch_size
    if args.noise_rate is not None: cfg.data.noise_rate = args.noise_rate
    if args.noise_type:    cfg.data.noise_type        = args.noise_type
    if args.loss:          cfg.training.loss_type     = args.loss
    if args.lr:            cfg.training.learning_rate = args.lr
    if args.no_co_teach:   cfg.training.use_co_teaching  = False
    if args.no_divide_mix: cfg.training.use_divide_mix   = False
    if args.no_bootstrap:  cfg.training.use_bootstrapping= False
    if args.fast:
        cfg.training.num_epochs = 3
        cfg.training.batch_size = 16
        print("[Fast mode] 3 epochs, batch=16")

    device = get_device(cfg.training.device)
    print(f"[Device] {device}")
    cfg.display()

    # ── 1. Data ────────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nSTEP 1: DATA\n" + "="*60)
    dm = DataModule(cfg)
    dm.setup()

    # ── 2. Models ──────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nSTEP 2: MODELS\n" + "="*60)
    model1, model2 = build_dual_models(cfg, device)

    # ── 3. Train ───────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nSTEP 3: TRAINING\n" + "="*60)
    print(f"  Loss={cfg.training.loss_type.upper()}  "
          f"CoTeach={cfg.training.use_co_teaching}  "
          f"DivideMix={cfg.training.use_divide_mix}  "
          f"Bootstrap={cfg.training.use_bootstrapping}  "
          f"Noise={cfg.data.noise_type}@{cfg.data.noise_rate:.0%}")

    trainer = Trainer(cfg, model1, model2, dm, device)
    history = trainer.train()

    # ── 4. Evaluate ────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nSTEP 4: EVALUATION\n" + "="*60)
    trainer.load_best()
    test_loader = dm.get_test_loader()

    preds, probs, labels, embs, languages = predict(
        model1, test_loader, device, return_embeddings=True
    )
    test_metrics          = compute_metrics(preds, labels, cfg.data.class_names)
    test_metrics["loss"]  = compute_loss(probs, labels)

    if languages:
        from evaluation.metrics import compute_per_language_metrics
        test_metrics["per_language"] = compute_per_language_metrics(
            preds, labels, languages, cfg.data.class_names
        )

    print(f"\n{'='*60}\nFINAL TEST RESULTS\n{'='*60}")
    print(test_metrics["classification_report"])
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  F1 Macro:    {test_metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    if "per_language" in test_metrics:
        print("\n  Per-language:")
        for lang, m in test_metrics["per_language"].items():
            print(f"    {lang:12s}  n={m['n']:5,}  acc={m['accuracy']:.4f}  f1={m['f1_weighted']:.4f}")

    save_json({k: v for k, v in test_metrics.items() if k != "classification_report"},
              os.path.join(cfg.training.log_dir, "test_results.json"))

    # ── 5. GMM on training losses ──────────────────────────────────────────────
    print("\n" + "="*60 + "\nSTEP 5: LOSS ANALYSIS\n" + "="*60)
    model1.eval()
    all_losses, all_noisy = [], []
    train_loader_seq = dm.get_train_loader(shuffle=False)
    with torch.no_grad():
        for batch in train_loader_seq:
            ids = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            tt  = batch["token_type_ids"].to(device)
            lbl = batch["label"].to(device)
            out = model1(ids, msk, tt)
            l   = torch.nn.functional.cross_entropy(out["logits"], lbl, reduction="none")
            all_losses.extend(l.cpu().numpy().tolist())
            all_noisy.extend(batch["is_noisy"].numpy().tolist())

    losses_arr = np.array(all_losses)
    noisy_arr  = np.array(all_noisy)
    gmm = GaussianMixtureNoiseSeparator(cfg.training.p_threshold)
    _, _, est_nr = gmm.fit_predict(losses_arr)
    print(f"  GMM estimated noise rate : {est_nr:.2%}")
    print(f"  True noise rate          : {cfg.data.noise_rate:.2%}")
    print(f"  Estimation error         : {abs(est_nr - cfg.data.noise_rate):.2%}")

    # ── 6. Visualisations ──────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nSTEP 6: VISUALISATIONS\n" + "="*60)
    run_all_visualizations(
        history        = history,
        test_metrics   = test_metrics,
        per_sample_losses = losses_arr,
        is_noisy       = noisy_arr,
        embeddings     = embs,
        true_labels    = labels,
        languages      = np.array(languages) if languages else None,
        cfg            = cfg,
    )

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"  Best Val F1  : {trainer.best_val_f1:.4f}  (epoch {trainer.best_epoch})")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 (w)  : {test_metrics['f1_weighted']:.4f}")
    print(f"  Plots        : {cfg.viz.output_dir}/")
    print(f"  Checkpoint   : {cfg.model.checkpoint_dir}/")
    print(f"  Logs         : {cfg.training.log_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
