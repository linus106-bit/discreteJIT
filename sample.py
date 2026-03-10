"""Sample qualitative denoising examples from a trained checkpoint."""
from __future__ import annotations

import argparse

import torch

from data import DataConfig, StructuredDenoisingDataset, collate_batch
from model import DenoisingTransformer
from utils import load_config, set_seed
from vocab import SymbolVocab


def to_symbols(ids: torch.Tensor, vocab: SymbolVocab):
    vals = []
    for idx in ids.tolist():
        if vocab.symbol_start <= idx < vocab.symbol_end:
            vals.append(vocab.id_to_symbol(idx))
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"].get("use_gpu", True) else "cpu")

    vocab = SymbolVocab(**cfg["vocab"])
    model = DenoisingTransformer(vocab_size=vocab.size, **cfg["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()

    val_data_cfg = {**cfg["data"]["val"], "num_samples": args.num_samples}
    ds = StructuredDenoisingDataset(
        DataConfig(**val_data_cfg),
        vocab,
        seed=cfg["seed"] + 9999,
    )
    batch = collate_batch([ds[i] for i in range(args.num_samples)], vocab.pad_id)
    inp = batch["input_ids"].to(device)
    attn = inp.ne(vocab.pad_id)
    with torch.no_grad():
        pred = model(inp, attn).argmax(dim=-1).cpu()

    for i in range(args.num_samples):
        length = int(batch["lengths"][i])
        row_ids = slice(0, length)
        c = to_symbols(batch["clean_ids"][i][row_ids], vocab)
        x = to_symbols(batch["input_ids"][i][row_ids], vocab)
        y = to_symbols(pred[i][row_ids], vocab)
        print(f"sample={i}")
        print(" corrupted:", x)
        print(" predicted:", y)
        print(" clean    :", c)


if __name__ == "__main__":
    main()
