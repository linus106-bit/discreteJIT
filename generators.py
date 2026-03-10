"""Structured synthetic sequence generator."""
from __future__ import annotations

import random
from typing import List


def repeating_motif(length: int, vocab_size: int, rng: random.Random) -> List[int]:
    motif_len = rng.randint(2, 6)
    motif = [rng.randrange(vocab_size) for _ in range(motif_len)]
    return [motif[i % motif_len] for i in range(length)]
