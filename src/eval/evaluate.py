"""Evaluate a saved checkpoint on a split with the same validate() the training
loop uses, so Dice/HD95 here are computed identically to the logged val numbers
(HD95 via src/eval/metrics.py, full resolution, mm).

Usage:
    python -m src.eval.evaluate --config CFG --checkpoint results/checkpoints/.../best.pt --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.dataset import load_config
from src.training.checkpoint import load_checkpoint
from src.training.train import build_datasets, build_loss, build_model, resolve_device, validate


def evaluate(config: dict, checkpoint: str | Path, split: str = "val") -> dict:
    device = resolve_device(config)
    datasets = build_datasets(config)
    if split not in datasets:
        raise SystemExit(f"unknown split {split!r}; available: {list(datasets)}")
    model = build_model(config).to(device)
    load_checkpoint(checkpoint, model, map_location=device, restore_rng=False)
    return validate(model, datasets[split], build_loss(config), device, config)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()
    metrics = evaluate(load_config(args.config), args.checkpoint, args.split)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
