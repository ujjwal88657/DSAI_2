"""
config.py
Single source of truth for every hyperparameter and path.
Dataset: combined_hate_speech_dataset.csv  (29 550 rows, binary labels)
"""

import os
from dataclasses import dataclass, field
from typing import List


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
DATASET_PATH = os.path.join(DATA_DIR, "combined_hate_speech_dataset.csv")


@dataclass
class DataConfig:
    dataset_path: str   = DATASET_PATH
    processed_dir: str  = os.path.join(DATA_DIR, "processed")

    # Label mapping  (dataset uses 0=not-hate, 1=hate — kept as-is)
    num_classes: int        = 2
    class_names: List[str]  = field(default_factory=lambda: ["not_hate", "hate"])

    # Noise simulation on top of the real dataset
    simulate_noise: bool = True
    noise_type: str      = "asymmetric"   # symmetric | asymmetric | instance
    noise_rate: float    = 0.30           # 30 % label corruption
    noise_seed: int      = 42

    # Train / val / test split
    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    test_ratio:  float = 0.15
    random_seed: int   = 42

    # Tokenizer
    max_seq_len:    int = 128
    tokenizer_name: str = "bert-base-multilingual-cased"


@dataclass
class ModelConfig:
    model_name:   str       = "bert-base-multilingual-cased"
    hidden_size:  int       = 768
    num_classes:  int       = 2
    dropout_rate: float     = 0.30
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [256])
    checkpoint_dir: str     = os.path.join(PROJECT_ROOT, "checkpoints")


@dataclass
class TrainingConfig:
    num_epochs:     int   = 10
    batch_size:     int   = 32
    learning_rate:  float = 2e-5
    weight_decay:   float = 1e-4
    warmup_ratio:   float = 0.10
    gradient_clip:  float = 1.0
    device:         str   = "cuda"        # auto-detected in main.py

    # Co-Teaching
    use_co_teaching: bool  = True
    forget_rate:     float = 0.20
    num_gradual:     int   = 5
    exponent:        float = 1.0

    # DivideMix GMM
    use_divide_mix: bool  = True
    p_threshold:    float = 0.5

    # Label Bootstrapping
    use_bootstrapping:     bool  = True
    bootstrap_beta:        float = 0.80
    bootstrap_start_epoch: int   = 3

    # Small-loss trick
    use_small_loss:      bool  = True
    small_loss_start:    int   = 3
    keep_ratio_initial:  float = 1.0
    keep_ratio_final:    float = 0.70

    # Loss function  (sce recommended)
    loss_type:  str   = "sce"
    sce_alpha:  float = 0.10
    sce_beta:   float = 1.00
    gce_q:      float = 0.70

    # Logging
    log_every_n_steps: int = 20
    log_dir: str = os.path.join(PROJECT_ROOT, "logs")


@dataclass
class VisualizationConfig:
    output_dir:            str   = os.path.join(PROJECT_ROOT, "visualizations")
    dpi:                   int   = 150
    embedding_sample_size: int   = 600
    umap_n_neighbors:      int   = 15
    umap_min_dist:         float = 0.10


@dataclass
class Config:
    data:     DataConfig          = field(default_factory=DataConfig)
    model:    ModelConfig         = field(default_factory=ModelConfig)
    training: TrainingConfig      = field(default_factory=TrainingConfig)
    viz:      VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        self.model.num_classes = self.data.num_classes
        for d in [self.data.processed_dir, self.model.checkpoint_dir,
                  self.training.log_dir, self.viz.output_dir]:
            os.makedirs(d, exist_ok=True)

    def display(self):
        import json, dataclasses
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(json.dumps(dataclasses.asdict(self), indent=2))
        print("=" * 60)


CFG = Config()
