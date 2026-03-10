"""Train a toy discrete structured denoising Transformer."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import DataConfig, StructuredDenoisingDataset, collate_batch
from evaluate import evaluate_model
from generators import SequenceGenerator
from model import DenoisingTransformer
from utils import AverageMeter, ensure_dir, load_config, save_json, set_seed
from vocab import SymbolVocab


def build_loader(cfg, split, vocab, seed, objective):
    ds = StructuredDenoisingDataset(
        DataConfig(**cfg["data"][split]),
        vocab=vocab,
        sequence_generator=SequenceGenerator(
            cfg["data"]["generators"], cfg["data"].get("generator_weights")
        ),
        seed=seed,
        objective=objective,
    )
    return DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"] if split == "train" else cfg["training"]["eval_batch_size"],
        shuffle=(split == "train"),
        collate_fn=lambda b: collate_batch(b, vocab.pad_id),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/default")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["training"].get("use_gpu", True) else "cpu")
    vocab = SymbolVocab(**cfg["vocab"])

    objective = cfg["objective"]["name"]
    train_loader = build_loader(cfg, "train", vocab, cfg["seed"], objective)
    val_loader = build_loader(cfg, "val", vocab, cfg["seed"] + 1000, objective)

    model = DenoisingTransformer(vocab_size=vocab.size, **cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"].get("weight_decay", 0.01))

    history = []
    best_metric = -1.0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        loss_meter = AverageMeter()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attn = input_ids.ne(vocab.pad_id)

            logits = model(input_ids, attn)
            if objective == "clean_prediction":
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=vocab.pad_id,
                )
            elif objective == "corruption_mask_prediction":
                loss = F.cross_entropy(
                    logits[..., :2].contiguous().view(-1, 2),
                    target_ids.view(-1),
                    ignore_index=vocab.pad_id,
                )
            else:
                raise ValueError(f"Unknown objective {objective}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), input_ids.size(0))

        val_metrics = evaluate_model(model, val_loader, device, vocab.pad_id)
        token_acc = val_metrics["overall"]["token_accuracy"]
        row = {"epoch": epoch, "train_loss": loss_meter.avg, **val_metrics["overall"]}
        history.append(row)
        print(row)

        ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": val_metrics}
        torch.save(ckpt, out_dir / "last.pt")
        if token_acc > best_metric:
            best_metric = token_acc
            torch.save(ckpt, out_dir / "best.pt")

    save_json({"history": history, "best_token_accuracy": best_metric}, out_dir / "metrics.json")


if __name__ == "__main__":
    main()
