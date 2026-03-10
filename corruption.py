"""Uniform symbol corruption for discrete denoising."""
from __future__ import annotations

import random
from typing import List, Tuple


def _different_uniform(original: int, vocab_size: int, rng: random.Random) -> int:
    candidate = rng.randrange(vocab_size - 1)
    return candidate + 1 if candidate >= original else candidate


def corrupt_sequence(
    clean: List[int],
    p: float,
    vocab_size: int,
    rng: random.Random,
    ensure_different: bool = True,
) -> Tuple[List[int], List[int]]:
    corrupted: List[int] = []
    mask: List[int] = []
    for token in clean:
        if rng.random() < p:
            replacement = (
                _different_uniform(token, vocab_size, rng)
                if ensure_different
                else rng.randrange(vocab_size)
            )
            corrupted.append(replacement)
            mask.append(1)
        else:
            corrupted.append(token)
            mask.append(0)
    return corrupted, mask
