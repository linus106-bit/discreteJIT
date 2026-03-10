# discrete-jit-structured-denoising

A toy PyTorch research project for one focused experiment:

> **Can a plain Transformer directly map a uniformly corrupted symbolic sequence back to a clean structured sequence?**

This project studies a **discrete analogue of JiT** using small-vocabulary sequences and a direct denoising objective.

## Discrete JiT in this toy setting

In this repository, "Discrete JiT" means learning a direct map:

- **input**: corrupted discrete sequence
- **output**: clean discrete sequence

There is no pretrained model, no latent variable model, no VAE, and no complex tokenizer. The default experiment is intentionally narrow and end-to-end runnable.

## Why structured denoising?

Clean sequences are **not IID random**; they come from a simple synthetic manifold of structured patterns:

1. `repeating_motif`
2. `arithmetic_mod`
3. `block_constant`

Because data are structured, denoising is meaningful: the model can infer underlying symbolic regularities instead of memorizing random noise.

## Corruption model

Default corruption is **uniform per-token replacement**:

- each position is replaced with probability `p`
- replacement is sampled uniformly from symbol vocabulary
- by default replacement differs from original symbol

`p=0.0` means no corruption; `p=1.0` means all positions corrupted.

## Main objective

Primary task is **direct clean prediction**:

- train with cross-entropy from corrupted input to clean target

Optional secondary objective (`corruption_mask_prediction`) is supported for comparison only.

## Repository layout

- `vocab.py` – symbolic vocabulary + special token handling
- `generators.py` – structured clean sequence generators
- `corruption.py` – uniform discrete corruption
- `data.py` – dataset + batching
- `model.py` – plain Transformer encoder
- `train.py` – training + validation + checkpoints
- `evaluate.py` – metrics, p-sweep, length generalization eval
- `sample.py` – qualitative examples
- `utils.py` – config, seed, and metric utilities
- `configs/` – runnable experiment configs

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the default baseline

```bash
python train.py --config configs/motif_clean_prediction.yaml --output-dir outputs/motif_clean
python evaluate.py --config configs/motif_clean_prediction.yaml --checkpoint outputs/motif_clean/best.pt
python sample.py --config configs/motif_clean_prediction.yaml --checkpoint outputs/motif_clean/best.pt
```

## Provided experiments

### A) Motif clean prediction

- config: `configs/motif_clean_prediction.yaml`
- generator: `repeating_motif`
- objective: `clean_prediction`
- fixed corruption: `p=0.1`

### B) Motif p-sweep

- config: `configs/motif_p_sweep.yaml`
- evaluates at `p in {0.0,0.1,0.2,0.4,0.6,0.8,1.0}`

```bash
python train.py --config configs/motif_p_sweep.yaml --output-dir outputs/motif_psweep
python evaluate.py --config configs/motif_p_sweep.yaml --checkpoint outputs/motif_psweep/best.pt
```

### C) Mixed generators clean prediction

- config: `configs/mixed_generators_clean.yaml`
- generators: motif + arithmetic + block-constant
- sampled corruption `p ~ Uniform(0, 0.5)`

### D) Optional objective comparison

- config: `configs/optional_objective_compare.yaml`
- objective list includes `clean_prediction` and `corruption_mask_prediction`
- run two trainings by overriding `objective.name`

## Metrics

Evaluation reports:

- token accuracy
- exact sequence accuracy
- corrupted-position accuracy
- uncorrupted-position accuracy
- length generalization accuracy (e.g. 32/64/96/128)
- generator-wise accuracy

## Notes

- Defaults are CPU/GPU compatible (single GPU if available).
- Keep this repo narrow: one focused structured denoising study.
