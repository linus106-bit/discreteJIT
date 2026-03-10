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


def _decode_ids(token_ids, vocab):
    return [vocab.itos[int(idx)] for idx in token_ids]


def _format_corruption(input_tokens, corruption_mask):
    formatted = []
    for token, is_corrupted in zip(input_tokens, corruption_mask):
        suffix = "*" if int(is_corrupted) == 1 else ""
        formatted.append(f"{token}{suffix}")
    return formatted


def evaluate_model(model, loader, device, pad_id, vocab, num_visualizations=0):
    model.eval()
    metric_meters = defaultdict(AverageMeter)
    shown = 0

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

            if shown < num_visualizations:
                lengths = batch["lengths"]
                for i in range(bsz):
                    if shown >= num_visualizations:
                        break
                    seq_len = int(lengths[i])
                    example_input = input_ids[i, :seq_len].detach().cpu()
                    example_pred = pred_ids[i, :seq_len].detach().cpu()
                    example_target = target_ids[i, :seq_len].detach().cpu()
                    example_corruption = corruption_mask[i, :seq_len].detach().cpu()

                    input_tokens = _decode_ids(example_input, vocab)
                    pred_tokens = _decode_ids(example_pred, vocab)
                    target_tokens = _decode_ids(example_target, vocab)
                    marked_input = _format_corruption(input_tokens, example_corruption)

                    print(f"\nExample {shown + 1}")
                    print("  Input :", " ".join(marked_input))
                    print("  Output:", " ".join(pred_tokens))
                    print("  Target:", " ".join(target_tokens))
                    shown += 1

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
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Number of input/output/target examples to print during evaluation.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"].get("use_gpu", True) else "cpu")

    vocab = SymbolVocab(**cfg["vocab"])
    model = DenoisingTransformer(vocab_size=vocab.size, **cfg["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])

    loader = build_eval_loader(cfg, vocab, cfg["seed"] + 2000)
    out = evaluate_model(
        model,
        loader,
        device,
        vocab.pad_id,
        vocab,
        num_visualizations=max(0, args.show_examples),
    )
    print(out)


if __name__ == "__main__":
    main()
