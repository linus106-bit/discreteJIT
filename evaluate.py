"""Evaluation script for discrete JIT denoising."""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from data import DataConfig, StructuredDenoisingDataset, collate_batch
from generators import SequenceGenerator
from model import DenoisingTransformer
from utils import AverageMeter, averages_to_dict, compute_metrics, load_config, set_seed
from vocab import SymbolVocab


def evaluate_model(model, loader, device, pad_id):
    model.eval()
    metric_meters = defaultdict(AverageMeter)
    length_meters = defaultdict(AverageMeter)
    gen_meters = defaultdict(AverageMeter)

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

            for i in range(bsz):
                l = int(batch["lengths"][i])
                g = batch["generator"][i]
                valid_i = attn[i]
                tok_acc_i = ((pred_ids[i] == target_ids[i]) & valid_i).sum().item() / max(
                    1, valid_i.sum().item()
                )
                length_meters[l].update(tok_acc_i)
                gen_meters[g].update(tok_acc_i)

    return {
        "overall": averages_to_dict(metric_meters),
        "length_generalization_accuracy": {str(k): v.avg for k, v in sorted(length_meters.items())},
        "generator_wise_accuracy": {k: v.avg for k, v in sorted(gen_meters.items())},
    }


def build_eval_loader(cfg, vocab, seed, lengths=None, fixed_p_override=None):
    eval_cfg = cfg["data"]["eval"]
    eval_cfg = dict(eval_cfg)
    if lengths is not None:
        eval_cfg["min_length"] = lengths
        eval_cfg["max_length"] = lengths
    if fixed_p_override is not None:
        eval_cfg["sampled_p"] = False
        eval_cfg["fixed_p"] = fixed_p_override
    ds = StructuredDenoisingDataset(
        DataConfig(**eval_cfg),
        vocab=vocab,
        sequence_generator=SequenceGenerator(
            cfg["data"]["generators"], cfg["data"].get("generator_weights")
        ),
        seed=seed,
        objective=cfg["objective"]["name"],
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

    p_sweep = cfg.get("evaluation", {}).get("p_sweep", [])
    if p_sweep:
        out["p_sweep"] = {}
        for i, p in enumerate(p_sweep):
            p_loader = build_eval_loader(cfg, vocab, cfg["seed"] + 5000 + i, fixed_p_override=float(p))
            p_out = evaluate_model(model, p_loader, device, vocab.pad_id)
            out["p_sweep"][str(p)] = p_out["overall"]["token_accuracy"]

    eval_lengths = cfg["evaluation"].get("lengths", [32, 64, 96, 128])
    for l in eval_lengths:
        l_loader = build_eval_loader(cfg, vocab, cfg["seed"] + 3000 + l, lengths=l)
        l_out = evaluate_model(model, l_loader, device, vocab.pad_id)
        out["length_generalization_accuracy"][str(l)] = l_out["overall"]["token_accuracy"]

    print(out)


if __name__ == "__main__":
    main()
