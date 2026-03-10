"""Evaluation script for discrete JIT denoising."""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from data import DataConfig, StructuredDenoisingDataset, collate_batch
from model import DenoisingTransformer
from utils import AverageMeter, averages_to_dict, compute_metrics, load_config, set_seed
from vocab import SymbolVocab


def evaluate_model(model, loader, device, pad_id):
    model.eval()
    metric_meters = defaultdict(AverageMeter)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            clean_ids = batch["clean_ids"].to(device)
            corruption_mask = batch["corruption_mask"].to(device)
            attn = input_ids.ne(pad_id)

            logits = model(input_ids, attn)
            pred_ids = logits.argmax(dim=-1)
            metrics = compute_metrics(pred_ids, target_ids, clean_ids, corruption_mask, attn)
            bsz = input_ids.size(0)
            for k, v in metrics.items():
                metric_meters[k].update(v, bsz)

    return averages_to_dict(metric_meters)


def build_eval_loader(cfg, vocab, seed):
    ds = StructuredDenoisingDataset(
        DataConfig(**cfg["data"]["eval"]),
        vocab=vocab,
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=cfg["training"]["eval_batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, vocab.pad_id),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"].get("use_gpu", True) else "cpu")

    vocab = SymbolVocab(**cfg["vocab"])
    model = DenoisingTransformer(vocab_size=vocab.size, **cfg["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])

    loader = build_eval_loader(cfg, vocab, cfg["seed"] + 2000)
    out = evaluate_model(model, loader, device, vocab.pad_id)
    print(out)


if __name__ == "__main__":
    main()
