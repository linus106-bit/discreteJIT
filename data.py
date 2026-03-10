"""Dataset and dataloader for structured denoising."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from corruption import corrupt_sequence
from generators import repeating_motif
from vocab import SymbolVocab


@dataclass
class DataConfig:
    num_samples: int
    min_length: int
    max_length: int
    fixed_p: float = 0.1


class StructuredDenoisingDataset(Dataset):
    def __init__(
        self,
        data_cfg: DataConfig,
        vocab: SymbolVocab,
        seed: int,
    ) -> None:
        self.cfg = data_cfg
        self.vocab = vocab
        self.seed = seed

    def __len__(self) -> int:
        return self.cfg.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + idx)
        length = rng.randint(self.cfg.min_length, self.cfg.max_length)
        clean = repeating_motif(length, self.vocab.base_vocab_size, rng)
        corrupted, corruption_mask = corrupt_sequence(clean, self.cfg.fixed_p, self.vocab.base_vocab_size, rng)

        inp = [self.vocab.bos_id] + self.vocab.encode_symbols(corrupted) + [self.vocab.eos_id]
        target = [self.vocab.bos_id] + self.vocab.encode_symbols(clean) + [self.vocab.eos_id]
        corruption_mask = [0] + corruption_mask + [0]

        return {
            "input_ids": torch.tensor(inp, dtype=torch.long),
            "target_ids": torch.tensor(target, dtype=torch.long),
            "clean_ids": torch.tensor(target, dtype=torch.long),
            "corruption_mask": torch.tensor(corruption_mask, dtype=torch.long),
            "length": torch.tensor(len(inp), dtype=torch.long),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(int(x["length"]) for x in batch)

    def pad_1d(t: torch.Tensor, value: int) -> torch.Tensor:
        if t.size(0) == max_len:
            return t
        out = torch.full((max_len,), value, dtype=t.dtype)
        out[: t.size(0)] = t
        return out

    return {
        "input_ids": torch.stack([pad_1d(x["input_ids"], pad_id) for x in batch]),
        "target_ids": torch.stack([pad_1d(x["target_ids"], pad_id) for x in batch]),
        "clean_ids": torch.stack([pad_1d(x["clean_ids"], pad_id) for x in batch]),
        "corruption_mask": torch.stack([pad_1d(x["corruption_mask"], 0) for x in batch]),
        "lengths": torch.tensor([int(x["length"]) for x in batch], dtype=torch.long),
    }
