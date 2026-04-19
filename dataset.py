"""
data/dataset.py
Loads combined_hate_speech_dataset.csv (29 550 rows, binary hate labels).
Handles:
  - cleaning & normalisation
  - noise injection  (symmetric / asymmetric / instance-dependent)
  - stratified split  (train 70 / val 15 / test 15)
  - HuggingFace tokenisation
  - PyTorch Dataset & DataLoader construction
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Light normalisation — keeps multilingual (Hindi / Hinglish / English) text intact.
    Only removes duplicate whitespace, leading/trailing spaces, and null bytes.
    We intentionally avoid aggressive stripping because the model (multilingual BERT)
    handles emoji, punctuation, and mixed script natively.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", " ")          # null bytes
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Noise injection
# ─────────────────────────────────────────────────────────────────────────────

def inject_symmetric_noise(
    labels: np.ndarray, noise_rate: float, num_classes: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Each label flipped to a uniformly random OTHER class with prob noise_rate."""
    rng = np.random.RandomState(seed)
    noisy = labels.copy()
    mask  = rng.rand(len(labels)) < noise_rate
    for i in np.where(mask)[0]:
        choices = [c for c in range(num_classes) if c != labels[i]]
        noisy[i] = rng.choice(choices)
    return noisy, mask


def inject_asymmetric_noise(
    labels: np.ndarray, noise_rate: float, num_classes: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Asymmetric (label-dependent) noise.
    Binary case (0=not_hate, 1=hate):
      - not_hate → hate  with prob noise_rate  (false positives — annotators over-flag)
      - hate → not_hate  with prob noise_rate/2 (false negatives — miss subtle hate)
    This matches real crowdsourced annotation patterns.
    """
    rng   = np.random.RandomState(seed)
    noisy = labels.copy()
    mask  = np.zeros(len(labels), dtype=bool)

    for i, lbl in enumerate(labels):
        if lbl == 0 and rng.rand() < noise_rate:          # not_hate mis-labeled as hate
            noisy[i] = 1;  mask[i] = True
        elif lbl == 1 and rng.rand() < noise_rate / 2.0:  # hate mis-labeled as not_hate
            noisy[i] = 0;  mask[i] = True

    return noisy, mask


def inject_instance_noise(
    labels: np.ndarray, text_lengths: np.ndarray,
    noise_rate: float, num_classes: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Instance-dependent noise: shorter texts are harder → higher flip probability.
    """
    rng = np.random.RandomState(seed)
    # Difficulty ∝ inverse text length (short texts are ambiguous)
    norm_len  = text_lengths / (text_lengths.max() + 1e-6)
    difficulty = 1.0 - norm_len                          # high difficulty = short text
    flip_prob  = np.clip(noise_rate * difficulty / difficulty.mean(), 0, 0.85)

    noisy = labels.copy()
    mask  = np.zeros(len(labels), dtype=bool)
    for i in np.where(rng.rand(len(labels)) < flip_prob)[0]:
        choices = [c for c in range(num_classes) if c != labels[i]]
        noisy[i] = rng.choice(choices)
        mask[i]  = True
    return noisy, mask


def apply_noise(
    df: pd.DataFrame,
    noise_type: str,
    noise_rate: float,
    num_classes: int,
    seed: int = 42,
) -> pd.DataFrame:
    labels = df["label"].values.astype(int)

    if noise_type == "symmetric":
        noisy, mask = inject_symmetric_noise(labels, noise_rate, num_classes, seed)
    elif noise_type == "asymmetric":
        noisy, mask = inject_asymmetric_noise(labels, noise_rate, num_classes, seed)
    elif noise_type == "instance":
        lengths = df["text_length"].values if "text_length" in df.columns \
                  else np.array([len(t) for t in df["text"]])
        noisy, mask = inject_instance_noise(labels, lengths, noise_rate, num_classes, seed)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    df = df.copy()
    df["original_label"] = labels
    df["label"]          = noisy
    df["is_noisy"]       = mask.astype(int)
    actual = mask.mean()
    print(f"  [Noise] type={noise_type}  requested={noise_rate:.2%}  actual={actual:.2%}  "
          f"flipped={mask.sum():,} / {len(labels):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HateSpeechDataset(Dataset):
    """
    Tokenises each text on the fly using a cached tokenizer.
    Stores per-sample index so noise-tracking strategies (Co-Teaching, GMM)
    can map back to the original dataset row.
    """

    def __init__(
        self,
        texts:          List[str],
        labels:         List[int],
        tokenizer,
        max_len:        int          = 128,
        indices:        Optional[List[int]] = None,
        original_labels: Optional[List[int]] = None,
        is_noisy:       Optional[List[int]] = None,
        languages:      Optional[List[str]] = None,
    ):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.indices   = indices        if indices         is not None else list(range(len(texts)))
        self.orig_lbl  = original_labels if original_labels is not None else labels
        self.is_noisy  = is_noisy       if is_noisy        is not None else [0] * len(texts)
        self.languages = languages      if languages        is not None else ["unknown"] * len(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids",
                torch.zeros(self.max_len, dtype=torch.long)
            ).squeeze(0),
            "label":          torch.tensor(self.labels[idx],    dtype=torch.long),
            "original_label": torch.tensor(self.orig_lbl[idx],  dtype=torch.long),
            "is_noisy":       torch.tensor(self.is_noisy[idx],  dtype=torch.long),
            "index":          torch.tensor(self.indices[idx],   dtype=torch.long),
            "text":           self.texts[idx],
            "language":       self.languages[idx],
        }

    def update_labels(self, new_labels: np.ndarray):
        """Used by label-correction methods to refurbish labels in-place."""
        self.labels = new_labels.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────

class DataModule:
    """
    One-stop shop for:
      load CSV → clean → split → (optionally) inject noise → tokenize → DataLoaders
    """

    def __init__(self, cfg):
        self.cfg   = cfg
        self.dcfg  = cfg.data
        self.tcfg  = cfg.training
        self.tokenizer  = None
        self.train_df   = None
        self.val_df     = None
        self.test_df    = None
        self._full_df   = None

    # ── Public setup ──────────────────────────────────────────────────────────
    def setup(self):
        print("\n[DataModule] Loading dataset …")
        df = self._load_and_clean()

        # ── Split BEFORE noise injection (test/val stay clean) ────────────
        train_val, test_df = train_test_split(
            df,
            test_size  = self.dcfg.test_ratio,
            stratify   = df["label"],
            random_state = self.dcfg.random_seed,
        )
        val_frac = self.dcfg.val_ratio / (1 - self.dcfg.test_ratio)
        train_df, val_df = train_test_split(
            train_val,
            test_size  = val_frac,
            stratify   = train_val["label"],
            random_state = self.dcfg.random_seed,
        )

        # ── Inject noise only into training data ──────────────────────────
        if self.dcfg.simulate_noise:
            train_df = apply_noise(
                train_df,
                self.dcfg.noise_type,
                self.dcfg.noise_rate,
                self.dcfg.num_classes,
                self.dcfg.noise_seed,
            )
        else:
            train_df = train_df.copy()
            train_df["original_label"] = train_df["label"]
            train_df["is_noisy"]       = 0

        # Val / Test keep clean labels
        val_df = val_df.copy()
        val_df["original_label"] = val_df["label"]
        val_df["is_noisy"]       = 0
        test_df = test_df.copy()
        test_df["original_label"] = test_df["label"]
        test_df["is_noisy"]       = 0

        self.train_df = train_df.reset_index(drop=True)
        self.val_df   = val_df.reset_index(drop=True)
        self.test_df  = test_df.reset_index(drop=True)

        self._print_stats()

        # ── Tokenizer ─────────────────────────────────────────────────────
        print(f"[DataModule] Loading tokenizer: {self.dcfg.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.dcfg.tokenizer_name)

    # ── Dataset builders ──────────────────────────────────────────────────────
    def _make_ds(self, df: pd.DataFrame) -> HateSpeechDataset:
        return HateSpeechDataset(
            texts           = df["text"].tolist(),
            labels          = df["label"].tolist(),
            tokenizer       = self.tokenizer,
            max_len         = self.dcfg.max_seq_len,
            indices         = df.index.tolist(),
            original_labels = df["original_label"].tolist(),
            is_noisy        = df["is_noisy"].tolist(),
            languages       = df["language"].tolist() if "language" in df.columns else None,
        )

    def get_train_dataset(self) -> HateSpeechDataset:  return self._make_ds(self.train_df)
    def get_val_dataset(self)   -> HateSpeechDataset:  return self._make_ds(self.val_df)
    def get_test_dataset(self)  -> HateSpeechDataset:  return self._make_ds(self.test_df)

    # ── DataLoader builders ───────────────────────────────────────────────────
    def _loader(self, ds: HateSpeechDataset, shuffle: bool, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            ds,
            batch_size  = self.tcfg.batch_size,
            shuffle     = shuffle,
            num_workers = 0,
            pin_memory  = False,
            drop_last   = drop_last,
        )

    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        return self._loader(self.get_train_dataset(), shuffle, drop_last=True)

    def get_val_loader(self) -> DataLoader:
        return self._loader(self.get_val_dataset(), shuffle=False)

    def get_test_loader(self) -> DataLoader:
        return self._loader(self.get_test_dataset(), shuffle=False)

    def get_paired_train_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Two independently-shuffled loaders used by Co-Teaching."""
        return (
            self._loader(self.get_train_dataset(), shuffle=True,  drop_last=True),
            self._loader(self.get_train_dataset(), shuffle=True,  drop_last=True),
        )

    # ── Internals ─────────────────────────────────────────────────────────────
    def _load_and_clean(self) -> pd.DataFrame:
        df = pd.read_csv(self.dcfg.dataset_path)

        # Rename columns to canonical names
        df = df.rename(columns={"hate_label": "label"})

        # Clean text
        df["text"] = df["text"].apply(clean_text)

        # Drop rows with empty text
        before = len(df)
        df = df[df["text"].str.len() > 3].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"  [Clean] Dropped {dropped} near-empty rows")

        # Ensure label is int
        df["label"] = df["label"].astype(int)

        # Add text_length if not present (used by instance noise)
        if "text_length" not in df.columns:
            df["text_length"] = df["text"].str.len()

        # Ensure language column exists
        if "language" not in df.columns:
            df["language"] = "unknown"

        # Add original_label and is_noisy placeholders (overwritten by apply_noise)
        df["original_label"] = df["label"]
        df["is_noisy"]       = 0

        print(f"  [Load] {len(df):,} rows  |  "
              f"label 0={( df['label']==0).sum():,}  label 1={(df['label']==1).sum():,}")
        return df

    def _print_stats(self):
        for name, df in [("Train", self.train_df), ("Val", self.val_df), ("Test", self.test_df)]:
            dist = df["label"].value_counts().sort_index().to_dict()
            print(f"  {name:5s}  rows={len(df):,}  labels={dist}")
        nr = self.train_df["is_noisy"].mean()
        print(f"  Train noise rate: {nr:.2%}")
        if "language" in self.train_df.columns:
            lang = self.train_df["language"].value_counts().to_dict()
            print(f"  Train language mix: {lang}")
