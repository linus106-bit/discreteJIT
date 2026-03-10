"""Shared utilities for config, metrics, and reproducibility."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import yaml


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def compute_metrics(
    pred_ids: torch.Tensor,
    target_ids: torch.Tensor,
    clean_ids: torch.Tensor,
    corruption_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, float]:
    valid = attention_mask.bool()
    token_correct = (pred_ids == target_ids) & valid
    token_acc = token_correct.sum().item() / max(1, valid.sum().item())

    seq_correct = ((pred_ids == target_ids) | ~valid).all(dim=1)
    seq_acc = seq_correct.float().mean().item()

    corr_valid = (corruption_mask == 1) & valid
    uncorr_valid = (corruption_mask == 0) & valid

    corr_acc = ((pred_ids == clean_ids) & corr_valid).sum().item() / max(1, corr_valid.sum().item())
    uncorr_acc = ((pred_ids == clean_ids) & uncorr_valid).sum().item() / max(1, uncorr_valid.sum().item())

    return {
        "token_accuracy": token_acc,
        "exact_sequence_accuracy": seq_acc,
        "corrupted_position_accuracy": corr_acc,
        "uncorrupted_position_accuracy": uncorr_acc,
    }


def averages_to_dict(acc: Dict[str, AverageMeter]) -> Dict[str, float]:
    return {k: meter.avg for k, meter in acc.items()}
