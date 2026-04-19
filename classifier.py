"""
models/classifier.py
BERT-based binary hate-speech classifier.

Architecture:
  bert-base-multilingual-cased
      → AttentionPooling  (learnable weighted mean over token states)
      → Dropout
      → Linear(768 → 256) → LayerNorm → GELU → Dropout
      → Linear(256 → num_classes)
      → Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple


# ── Attention pooling ─────────────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    """
    Learn a scalar attention score per token, softmax over sequence,
    return weighted sum. Better than [CLS] alone for code-mixed text
    where the important tokens can appear anywhere.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden: (B, L, H)   mask: (B, L)
        scores = self.attn(hidden).squeeze(-1)                   # (B, L)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)        # (B, L, 1)
        return (hidden * weights).sum(dim=1)                     # (B, H)


# ── Classification head ────────────────────────────────────────────────────────
class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int, dropout: float):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Full model ────────────────────────────────────────────────────────────────
class BERTClassifier(nn.Module):
    """
    Wraps bert-base-multilingual-cased for binary hate-speech detection.
    Accepts multilingual input (English / Hindi / Hinglish) natively.
    """

    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg.model
        print(f"  [Model] Loading {mcfg.model_name} …")
        bert_cfg = AutoConfig.from_pretrained(mcfg.model_name, output_hidden_states=True)
        self.bert    = AutoModel.from_pretrained(mcfg.model_name, config=bert_cfg)
        self.pooler  = AttentionPooling(mcfg.hidden_size)
        self.drop    = nn.Dropout(mcfg.dropout_rate)
        self.head    = ClassificationHead(
            mcfg.hidden_size, mcfg.classifier_hidden_dims,
            mcfg.num_classes, mcfg.dropout_rate,
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Convenience ───────────────────────────────────────────────────────────
    def freeze_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = True

    def count_params(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        out = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        pooled = self.pooler(out.last_hidden_state, attention_mask)  # (B, H)
        logits = self.head(self.drop(pooled))
        result = {"logits": logits}
        if return_embeddings:
            result["embeddings"] = pooled.detach()
        return result


# ── Factory helpers ────────────────────────────────────────────────────────────
def build_model(cfg, device: torch.device) -> BERTClassifier:
    model = BERTClassifier(cfg).to(device)
    p = model.count_params()
    print(f"  [Model] Total={p['total']:,}  Trainable={p['trainable']:,}")
    return model


def build_dual_models(cfg, device: torch.device) -> Tuple[BERTClassifier, BERTClassifier]:
    m1 = build_model(cfg, device)
    m2 = build_model(cfg, device)
    print("  [Model] Dual models built for Co-Teaching")
    return m1, m2
