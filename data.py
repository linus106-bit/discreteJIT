"""Dataset and dataloader for structured denoising."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from corruption import corrupt_sequence
from generators import SequenceGenerator
from vocab import SymbolVocab


@dataclass
class DataConfig:
    num_samples: int
    min_length: int
    max_length: int
    fixed_p: Optional[float] = 0.1
    sampled_p: bool = False
    sampled_p_min: float = 0.0
    sampled_p_max: float = 0.5
    use_noise_level_token: bool = False


class StructuredDenoisingDataset(Dataset):
    def __init__(
        self,
        data_cfg: DataConfig,
        vocab: SymbolVocab,
        sequence_generator: SequenceGenerator,
        seed: int,
        objective: str = "clean_prediction",
    ) -> None:
        self.cfg = data_cfg
        self.vocab = vocab
        self.sequence_generator = sequence_generator
        self.seed = seed
        self.objective = objective

    def __len__(self) -> int:
        return self.cfg.num_samples

    def _sample_p(self, rng: random.Random) -> float:
        if self.cfg.sampled_p:
            return rng.uniform(self.cfg.sampled_p_min, self.cfg.sampled_p_max)
        if self.cfg.fixed_p is None:
            raise ValueError("fixed_p must be set when sampled_p is False")
        return self.cfg.fixed_p

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(self.seed + idx)
        length = rng.randint(self.cfg.min_length, self.cfg.max_length)
        clean, generator_name = self.sequence_generator.sample(length, self.vocab.base_vocab_size, rng)
        p = self._sample_p(rng)
        corrupted, corruption_mask = corrupt_sequence(clean, p, self.vocab.base_vocab_size, rng)

        inp = self.vocab.encode_symbols(corrupted)
        target_clean = self.vocab.encode_symbols(clean)
        target = target_clean if self.objective == "clean_prediction" else corruption_mask

        if self.cfg.use_noise_level_token:
            noise_id = self.vocab.noise_token_id(p)
            inp = [noise_id] + inp
            target_clean = [self.vocab.pad_id] + target_clean
            target = [self.vocab.pad_id] + target if self.objective == "clean_prediction" else [0] + target
            corruption_mask = [0] + corruption_mask

        inp = [self.vocab.bos_id] + inp + [self.vocab.eos_id]
        if self.objective == "clean_prediction":
            target = [self.vocab.bos_id] + target + [self.vocab.eos_id]
        else:
            target = [0] + target + [0]
        target_clean = [self.vocab.bos_id] + target_clean + [self.vocab.eos_id]
        corruption_mask = [0] + corruption_mask + [0]

        return {
            "input_ids": torch.tensor(inp, dtype=torch.long),
            "target_ids": torch.tensor(target, dtype=torch.long),
            "clean_ids": torch.tensor(target_clean, dtype=torch.long),
            "corruption_mask": torch.tensor(corruption_mask, dtype=torch.long),
            "length": torch.tensor(len(inp), dtype=torch.long),
            "p": torch.tensor(p, dtype=torch.float),
            "generator": generator_name,
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
        "target_ids": torch.stack([pad_1d(x["target_ids"], pad_id if batch[0]["target_ids"].dtype == torch.long else 0) for x in batch]),
        "clean_ids": torch.stack([pad_1d(x["clean_ids"], pad_id) for x in batch]),
        "corruption_mask": torch.stack([pad_1d(x["corruption_mask"], 0) for x in batch]),
        "lengths": torch.tensor([int(x["length"]) for x in batch], dtype=torch.long),
        "p": torch.tensor([float(x["p"]) for x in batch], dtype=torch.float),
        "generator": [x["generator"] for x in batch],
    }
