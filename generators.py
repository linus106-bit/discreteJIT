"""Structured synthetic sequence generators."""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence, Tuple


def repeating_motif(length: int, vocab_size: int, rng: random.Random) -> List[int]:
    motif_len = rng.randint(2, 6)
    motif = [rng.randrange(vocab_size) for _ in range(motif_len)]
    return [motif[i % motif_len] for i in range(length)]


def arithmetic_mod(length: int, vocab_size: int, rng: random.Random) -> List[int]:
    s = rng.randrange(vocab_size)
    d = rng.randrange(vocab_size)
    return [(s + t * d) % vocab_size for t in range(length)]


def block_constant(length: int, vocab_size: int, rng: random.Random) -> List[int]:
    n_blocks = rng.randint(2, min(8, length))
    boundaries = sorted(rng.sample(range(1, length), k=n_blocks - 1))
    boundaries = [0] + boundaries + [length]
    seq: List[int] = []
    for i in range(len(boundaries) - 1):
        block_len = boundaries[i + 1] - boundaries[i]
        symbol = rng.randrange(vocab_size)
        seq.extend([symbol] * block_len)
    return seq


GENERATOR_REGISTRY: Dict[str, Callable[[int, int, random.Random], List[int]]] = {
    "repeating_motif": repeating_motif,
    "arithmetic_mod": arithmetic_mod,
    "block_constant": block_constant,
}


class SequenceGenerator:
    """Sampler over one or multiple generators."""

    def __init__(self, names: Sequence[str], weights: Sequence[float] | None = None) -> None:
        if not names:
            raise ValueError("At least one generator is required")
        for name in names:
            if name not in GENERATOR_REGISTRY:
                raise ValueError(f"Unknown generator: {name}")
        self.names = list(names)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.names)

    def sample(self, length: int, vocab_size: int, rng: random.Random) -> Tuple[List[int], str]:
        name = rng.choices(self.names, weights=self.weights, k=1)[0]
        seq = GENERATOR_REGISTRY[name](length, vocab_size, rng)
        return seq, name
