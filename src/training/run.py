"""Config-driven training for B1 (unet3d), B2 (dual_unet) and ACF on PET/CT.

    python -m src.training.run --config configs/b4_acf_cond.yaml
    python -m src.training.run --config configs/b4_acf_cond.yaml --resume
    python -m src.training.run --config configs/b4_acf_cond.yaml --override train.n_epochs=2

Every run directory (config run.dir) gets: config.yaml (the exact config, overrides applied),
env.json (git hash + dirty flag, python / torch / CUDA / MONAI versions, device, seed),
metrics.csv + metrics.jsonl (one row per epoch), checkpoints/ (epoch_NNNN.pt every epoch,
best.pt by validation Dice). Checkpoints carry model / optimizer / scaler / scheduler / RNG
state, so --resume continues an interrupted (Colab) session.

Batching: a DataLoader fetch of loader_batch_size patients x patch.num_samples crops is split
into micro-batches of train.micro_batch_size patches; gradients accumulate over
train.accum_steps micro-batches (effective batch = micro_batch_size * accum_steps).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import monai
import numpy as np
import torch
import yaml
from monai.data import DataLoader as MonaiDataLoader
from torch.amp import GradScaler, autocast

from src.data.conditioning import load_stats
from src.data.dataset import load_config
from src.data.dual_cache import build_dual_datasets
from src.data.seed import seed_everything
from src.eval.evaluate_cases import call_model, evaluate_dataset, summarize
from src.models.acf import build_acf
from src.models.dual_encoder import build_dual_encoder_unet
from src.models.unet3d import build_unet3d
from src.training.checkpoint import (
    load_latest_valid_checkpoint, prune_checkpoints, save_best_checkpoint, save_checkpoint,
)
from src.training.losses import dice_ce_loss, evidential_loss
from src.training.train import apply_overrides, get_git_info, resolve_device, warn_if_smoke_only

MODEL_BUILDERS = {"unet3d": build_unet3d, "dual_unet": build_dual_encoder_unet, "acf": build_acf}


def build_model(config: dict) -> torch.nn.Module:
    name = config["model"]["name"]
    if name not in MODEL_BUILDERS:
        raise ValueError(f"unknown model {name!r}; registered: {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](config)


def build_optimizer_and_scheduler(model, config):
    t = config["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))
    sched = None
    if config.get("scheduler", {}).get("name", "cosine") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["n_epochs"], eta_min=config.get("scheduler", {}).get("eta_min", 0.0))
    return opt, sched


def compute_loss(out, labels, config, epoch):
    """Softmax head: Dice+CE on logits. Evidential head: evidential CE + annealed KL (+ Dice)."""
    lc = config.get("loss", {})
    if config["model"].get("head", "softmax") == "evidential":
        return evidential_loss(out.float(), labels, epoch, lc.get("kl_anneal_epochs", 10),
                               lc.get("dice_weight", 1.0), lc.get("smooth", 1.0))
    return dice_ce_loss(out.float(), labels, lc.get("smooth", 1.0), lc.get("lambda_dice", 1.0), lc.get("lambda_ce", 1.0)), {}


def micro_batches(batch: dict, size: int):
    n = batch["ct"].shape[0]
    if n % size:
        raise ValueError(f"fetched {n} crops, not divisible by micro_batch_size={size}; adjust "
                         "train.loader_batch_size * patch.num_samples")
    for s in range(0, n, size):
        yield {k: (v[s:s + size] if torch.is_tensor(v) else v[s:s + size]) for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, scaler, device, config, epoch) -> dict:
    t = config["train"]
    accum, mb = t["accum_steps"], t["micro_batch_size"]
    use_amp = t.get("amp", True) and device.type == "cuda"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    tot, n, step, data_s, gpu_s = 0.0, 0, 0, 0.0, 0.0
    t_ep = time.perf_counter()
    it = iter(loader)
    while True:
        t0 = time.perf_counter()
        batch = next(it, None)
        data_s += time.perf_counter() - t0
        if batch is None:
            break
        t0 = time.perf_counter()
        for mbatch in micro_batches(batch, mb):
            d = {k: v.to(device, non_blocking=True) for k, v in mbatch.items() if torch.is_tensor(v)}
            with autocast(device_type=device.type, enabled=use_amp):
                out = call_model(model, d["ct"], d["pet"], d["cond_ct"], d["cond_pet"], d["ct_spacing"], d["pet_spacing"])
            loss, _ = compute_loss(out, d["label"], config, epoch)
            scaler.scale(loss / accum).backward()
            tot, n, step = tot + loss.item(), n + 1, step + 1
            if step % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_s += time.perf_counter() - t0
    if step % accum:  # leftover partial accumulation window
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return {"train_loss": tot / max(n, 1), "time_data_s": data_s, "time_gpu_s": gpu_s,
            "time_total_s": time.perf_counter() - t_ep,
            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}


def environment_info(config: dict, device) -> dict:
    g = get_git_info()
    return {"git_commit": g["commit"], "git_dirty": g["dirty"], "seed": config["seed"], "device": str(device),
            "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "monai": monai.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


def train(config: dict, resume: bool = False) -> Path:
    seed_everything(config["seed"])
    device = resolve_device(config)
    warn_if_smoke_only(config, device)
    run_dir = Path(config["run"]["dir"])
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    env = environment_info(config, device)
    (run_dir / "env.json").write_text(json.dumps(env, indent=1))
    print(f"seed {config['seed']}  device {device}  git {env['git_commit']} dirty={env['git_dirty']}")

    datasets = build_dual_datasets(config)
    t = config["train"]
    nw = (os.cpu_count() or 0) if t.get("num_workers", 0) == "auto" else int(t.get("num_workers", 0))
    loader = MonaiDataLoader(datasets["train"], batch_size=t["loader_batch_size"], shuffle=True, num_workers=nw,
                             pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    model = build_model(config).to(device)
    print(f"model {config['model']['name']}: {sum(p.numel() for p in model.parameters()):,} parameters")
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    scaler = GradScaler(device=device.type, enabled=t.get("amp", True) and device.type == "cuda")
    stats = load_stats(config["conditioning"]["stats_path"])

    start, best = 1, -1.0
    if resume:
        path, ckpt = load_latest_valid_checkpoint(ckpt_dir, model, optimizer, scaler, scheduler, map_location=device)
        if path is not None:
            start, best = ckpt["epoch"] + 1, ckpt.get("best_metric", -1.0)
            print(f"resumed from {path} at epoch {start}")
    keep = config.get("checkpoint", {}).get("keep_last_n", 3)
    csv_path = run_dir / "metrics.csv"
    git = env["git_commit"]

    for epoch in range(start, t["n_epochs"] + 1):
        datasets["train"].set_epoch(epoch)
        row = {"epoch": epoch, "git": git, **train_one_epoch(model, loader, optimizer, scaler, device, config, epoch)}
        row["lr"] = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        is_best = False
        if epoch % t.get("val_every", 1) == 0 or epoch == t["n_epochs"]:
            df = evaluate_dataset(model, datasets["val"], config, stats, device, max_cases=t.get("val_max_cases"), git_hash=git)
            s = summarize(df)
            row.update({f"val_{k}": v for k, v in s.items()})
            vd = s["dice"]
            is_best = not np.isnan(vd) and vd > best
            best = max(best, vd) if not np.isnan(vd) else best
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "git"})
        new = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        ep = ckpt_dir / f"epoch_{epoch:04d}.pt"
        save_checkpoint(ep, epoch, model, optimizer, scaler, best, config, scheduler=scheduler, extra={"git": env})
        if is_best:
            save_best_checkpoint(ckpt_dir, ep)
        prune_checkpoints(ckpt_dir, keep)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    cfg = apply_overrides(load_config(a.config), a.override)
    if a.device:
        cfg["train"]["device"] = a.device
    train(cfg, a.resume)


if __name__ == "__main__":
    main()
