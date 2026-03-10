"""Vocabulary utilities for symbolic denoising experiments."""
from dataclasses import dataclass
from typing import List


@dataclass
class SymbolVocab:
    """Small discrete vocabulary with special tokens."""

    base_vocab_size: int = 8

    def __post_init__(self) -> None:
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        self.symbol_tokens = [str(i) for i in range(self.base_vocab_size)]
        self.special_tokens: List[str] = [self.pad_token, self.bos_token, self.eos_token]
        self.itos = self.special_tokens + self.symbol_tokens
        self.stoi = {t: i for i, t in enumerate(self.itos)}

        self.pad_id = self.stoi[self.pad_token]
        self.bos_id = self.stoi[self.bos_token]
        self.eos_id = self.stoi[self.eos_token]

        self.symbol_start = len(self.special_tokens)
        self.symbol_end = self.symbol_start + self.base_vocab_size

    @property
    def size(self) -> int:
        return len(self.itos)

    def symbol_id(self, value: int) -> int:
        return self.symbol_start + int(value)

    def id_to_symbol(self, idx: int) -> int:
        return idx - self.symbol_start

    def encode_symbols(self, values: List[int]) -> List[int]:
        return [self.symbol_id(v) for v in values]
