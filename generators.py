"""Structured synthetic sequence generators."""
from __future__ import annotations

import random
from typing import List


def repeating_motif(length: int, vocab_size: int, rng: random.Random) -> List[int]:
    motif_len = rng.randint(2, 6)
    motif = [rng.randrange(vocab_size) for _ in range(motif_len)]
    return [motif[i % motif_len] for i in range(length)]


def mirrored_motif(length: int, vocab_size: int, rng: random.Random) -> List[int]:
    motif_len = rng.randint(2, 5)
    motif = [rng.randrange(vocab_size) for _ in range(motif_len)]
    pal = motif + motif[-2::-1]
    return [pal[i % len(pal)] for i in range(length)]


def arithmetic_walk(length: int, vocab_size: int, rng: random.Random) -> List[int]:
    start = rng.randrange(vocab_size)
    step = rng.randint(1, max(1, vocab_size - 1))
    return [(start + i * step) % vocab_size for i in range(length)]


def interleaved_motifs(length: int, vocab_size: int, rng: random.Random) -> List[int]:
    len_a = rng.randint(2, 4)
    len_b = rng.randint(2, 4)
    motif_a = [rng.randrange(vocab_size) for _ in range(len_a)]
    motif_b = [rng.randrange(vocab_size) for _ in range(len_b)]
    seq = []
    for i in range(length):
        if i % 2 == 0:
            seq.append(motif_a[(i // 2) % len_a])
        else:
            seq.append(motif_b[(i // 2) % len_b])
    return seq


PATTERN_GENERATORS = {
    "repeating_motif": repeating_motif,
    "mirrored_motif": mirrored_motif,
    "arithmetic_walk": arithmetic_walk,
    "interleaved_motifs": interleaved_motifs,
}


def generate_structured_sequence(
    length: int,
    vocab_size: int,
    rng: random.Random,
    pattern_types: List[str] | None,
) -> List[int]:
    available = pattern_types or ["repeating_motif"]
    pattern_name = rng.choice(available)
    if pattern_name not in PATTERN_GENERATORS:
        raise ValueError(f"Unknown pattern type: {pattern_name}")
    return PATTERN_GENERATORS[pattern_name](length, vocab_size, rng)
