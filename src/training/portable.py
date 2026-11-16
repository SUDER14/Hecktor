"""Portable, step-based training for the shared-PET runs (B1, B2, B2b, B3, B4, B5). CLI: scripts/train.py.

Why a second trainer next to src/training/run.py: run.py counts epochs, checkpoints once per epoch and cannot
resume inside an epoch. A free GPU session can die at any minute, so this one

* measures length in OPTIMISER steps (config `steps.max_steps`), cosine schedule stepped per optimiser step;
* checkpoints every `steps.ckpt_every_minutes` and at every validation: temp file, fsync, rename; keeps the
  last 2 plus best.pt; each holds model, optimiser, scheduler, scaler, RNG states, step, data position, best score;
* validates every `steps.val_every` steps on a fixed subset of validation cases and once on all validation
  cases at the end; best checkpoint = highest mean of label-1 and label-2 aggregated Dice on validation;
* never builds a dataset from the test ids (see `select_ids`);
* writes <out_root>/runs/<name>/manifest.json, train_log.csv, val_log.csv.

Data order is a pure function of (seed, pass): pass p visits rng([seed, p]).permutation(n_train); crops are a pure
function of (seed, pass, case) in the dataset. A checkpoint stores (pass, fetches done), so a resumed run sees the
same batches as an uninterrupted one. On CPU the resumed weights are bit-identical (tests/test_portable_train.py);
on a GPU cuDNN is not bit-reproducible, so "identical" is only claimed for the CPU test.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import monai
import numpy as np
import torch
import yaml
from monai.data import list_data_collate
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from src.data.conditioning import load_stats
from src.data.seed import seed_everything
from src.data.splits import load_split
from src.eval.evaluate_shared import call_model
from src.eval.infer import resample_to_grid, sliding_window
from src.training.checkpoint import (
    _atomic_torch_save,
    _capture_rng_state,
    _restore_rng_state,
)
from src.training.run import build_model, compute_loss, micro_batches
from src.training.train import apply_overrides, get_git_info

ROOT = Path(__file__).resolve().parents[2]
KEEP_LAST = 2
HOSTED_ENV = ("COLAB_RELEASE_TAG", "COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE")   # set on Colab / Kaggle machines


def sha256_file(path) -> str:
    """Raw bytes. For binary files only."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(path) -> str:
    """sha256 of a TEXT file after CRLF -> LF, the same rule as src/eval/locked_test.py. One commit is checked out
    with LF on Colab and with CRLF on Windows, so a hash of the raw bytes would call one split file two files.
    Used for the config, the split and the conditioning statistics."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _repo_relative(path) -> str:
    """Repo-relative POSIX path when the file is inside the repository, else only its name: no machine path is
    written into a checkpoint or a manifest."""
    p = Path(path).resolve()
    return p.relative_to(ROOT).as_posix() if ROOT in p.parents else p.name


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def select_ids(split: dict) -> tuple[list[str], list[str]]:
    """(train ids, validation ids). The test list is read only to prove it is disjoint; no dataset, header or
    file of a test case is ever opened by this module."""
    train, val, test = list(split["train"]), list(split["val"]), set(split.get("test", []))
    leak = (set(train) | set(val)) & test
    if leak:
        raise RuntimeError(f"split is corrupt: {len(leak)} test ids appear in train/val, e.g. {sorted(leak)[:3]}")
    return train, val


def val_subset(val_ids: list[str], n: int, seed: int) -> list[int]:
    """Indices of the fixed validation subset: the same cases at every validation and after every resume."""
    if n >= len(val_ids):
        return list(range(len(val_ids)))
    return sorted(int(i) for i in np.random.default_rng([seed, 20261116]).permutation(len(val_ids))[:n])


# ---------------------------------------------------------------------------------------------- synthetic data
class SyntheticCrops(torch.utils.data.Dataset):
    """Random tensors with the keys and shapes of AcfCropDataset. For checking the loop, never for results."""

    def __init__(self, n: int, config: dict):
        self.n, self.patch = n, tuple(config["patch"]["spatial_size"])
        self.num_samples, self.seed = config["patch"]["num_samples"], int(config["seed"])
        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        self._epoch[0] = int(epoch)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> list[dict]:
        rng = np.random.default_rng([self.seed, int(self._epoch[0]), idx])
        return [_synthetic_item(rng, self.patch, idx) for _ in range(self.num_samples)]


class SyntheticVolumes(torch.utils.data.Dataset):
    def __init__(self, n: int, config: dict, salt: int = 999):
        self.n, self.seed, self.salt = n, int(config["seed"]), salt
        self.shape = tuple(int(p) + 8 for p in config["patch"]["spatial_size"])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        item = _synthetic_item(np.random.default_rng([self.seed, self.salt, idx]), self.shape, idx)
        aff = np.diag([*item["spacing"].tolist(), 1.0])
        item["affine"] = item["native_affine"] = torch.tensor(aff, dtype=torch.float64)
        item["native_label"] = item["label"][0].to(torch.uint8)
        return item


def _synthetic_item(rng, shape, idx: int) -> dict:
    img = rng.random((2, *shape), dtype=np.float32)
    lab = np.zeros(shape, np.int64)
    for k in (1, 2):   # one small box per label, brighter in the image so there is something to learn
        s = [int(rng.integers(0, max(1, n - 6))) for n in shape]
        sl = tuple(slice(a, a + 6) for a in s)
        lab[sl] = k
        img[(slice(None), *sl)] += 0.5 * k
    sp = np.array([1.0, 1.0, float(rng.choice([1.5, 3.0, 5.0]))], np.float32)
    return {"patient_id": f"SYN-{idx:03d}", "centre": "SYN", "image": torch.from_numpy(img),
            "label": torch.from_numpy(lab)[None], "spacing": torch.from_numpy(sp),
            "cond": torch.from_numpy((sp - np.array([1.0, 1.0, 3.0], np.float32)) / np.array([0.2, 0.2, 1.0], np.float32))}


def build_datasets(config: dict, data_root: Path | None, synthetic: bool):
    """(train crops, validation volumes). Real data: only train and validation ids are ever turned into samples."""
    if synthetic:
        return SyntheticCrops(16, config), SyntheticVolumes(4, config)
    from src.data import crop_cache
    from src.data.acf_dataset import AcfCropDataset, AcfVolumeDataset
    train_ids, val_ids = select_ids(load_split(config["split"]["path"]))
    stats = load_stats(config["conditioning"]["stats_path"])

    def samples(ids):
        return [{"patient_id": i, "centre": crop_cache.read_header(data_root / f"{i}.acf")["centre"]} for i in ids]

    return (AcfCropDataset(samples(train_ids), data_root, config, stats),
            AcfVolumeDataset(samples(val_ids), data_root, config, stats))


# -------------------------------------------------------------------------------------------------- validation
@torch.no_grad()
def validate(model, val_ds, indices, config: dict, device, use_amp: bool) -> dict:
    """Aggregated Dice per label over `indices`, on the native grid (resampled models are mapped back first)."""
    model.eval()
    patch, k = tuple(config["patch"]["spatial_size"]), config["model"]["out_channels"]
    head = config["model"].get("head", "softmax")
    inf = config.get("infer", {})
    inter, size = np.zeros(k), np.zeros(k)
    t0 = time.perf_counter()
    for i in indices:
        item = val_ds[i]
        cond, sp = item["cond"][None].to(device), item["spacing"][None].to(device)

        def fwd(x, cond=cond, sp=sp):
            with autocast(device_type=device.type, enabled=use_amp):
                return call_model(model, x, cond.expand(x.shape[0], -1), sp.expand(x.shape[0], -1))

        p = sliding_window(fwd, item["image"].numpy(), patch, k, head, inf.get("overlap", 0.5),
                           inf.get("sw_batch_size", 1), device)["p"]
        label, aff, lab_aff = item["native_label"].numpy(), item["affine"].numpy(), item["native_affine"].numpy()
        if p.shape[1:] != label.shape or not np.allclose(aff, lab_aff):
            p = resample_to_grid(p, aff, lab_aff, label.shape, order=1)
        pred = p.argmax(0)
        for c in range(1, k):
            inter[c] += np.logical_and(pred == c, label == c).sum()
            size[c] += (pred == c).sum() + (label == c).sum()
    model.train()
    agg = {c: (2 * inter[c] / size[c] if size[c] > 0 else float("nan")) for c in range(1, k)}
    score = float(np.mean([agg[c] for c in range(1, k)]))   # NaN if any label is empty everywhere
    return {"n_cases": len(indices), **{f"agg_dice_{c}": float(agg[c]) for c in agg}, "score": score,
            "seconds": round(time.perf_counter() - t0, 1)}


# ------------------------------------------------------------------------------------------------- checkpoints
def step_checkpoints(ckpt_dir: Path) -> list[Path]:
    return sorted(ckpt_dir.glob("step_*.pt"))


def load_latest(ckpt_dir: Path):
    """Newest checkpoint that loads; a corrupt newest file (should not exist, saves are atomic) is skipped."""
    for p in reversed(step_checkpoints(ckpt_dir)):
        try:
            return p, torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001  a damaged file must not block resume from the previous one
            print(f"WARNING: cannot load {p.name} ({type(e).__name__}); trying the previous checkpoint")
    return None, None


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _truncate_csv(path: Path, max_step: int) -> None:
    """Drop rows written after the checkpoint we resume from, so a log never holds a step twice."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if int(r["step"]) <= max_step]
    if len(keep) != len(rows):
        tmp = path.with_name(f".{path.name}.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(keep)
        os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------- train
def prepare_config(config_path: Path, overrides: list[str]) -> dict:
    """The config that is stored in checkpoints: portable, with no path of this machine. Split and statistics
    stay repo-relative; the cache folder, the zip path and the old run folder are blanked (the cache comes from
    --data-root at run time, see `runtime_config`)."""
    cfg = apply_overrides(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")), overrides)
    for sec, key in (("split", "path"), ("conditioning", "stats_path")):
        if Path(cfg[sec][key]).is_absolute():
            raise SystemExit(f"{config_path}: {sec}.{key} must be a path relative to the repository, got {cfg[sec][key]}")
        cfg[sec][key] = Path(cfg[sec][key]).as_posix()
    cfg.setdefault("cache", {})["dir"] = None
    if "data_dir" in cfg:
        cfg["data_dir"] = None
    if isinstance(cfg.get("run"), dict) and "dir" in cfg["run"]:
        cfg["run"]["dir"] = None
    if "steps" not in cfg:
        raise SystemExit(f"{config_path} has no `steps:` block (max_steps, val_every, val_subset_cases, ckpt_every_minutes)")
    return cfg


def runtime_config(cfg: dict, data_root: Path | None) -> dict:
    """A copy for THIS machine (absolute split / statistics paths, the cache folder given now). Never saved."""
    rt = copy.deepcopy(cfg)
    for sec, key in (("split", "path"), ("conditioning", "stats_path")):
        rt[sec][key] = str(ROOT / cfg[sec][key])
    rt["cache"]["dir"] = None if data_root is None else str(data_root)
    return rt


def check_environment(synthetic: bool) -> dict:
    """Refuse a dirty or unknown git tree for real runs, and refuse real data on a hosted notebook machine unless
    the owner set ALLOW_REAL_DATA=1 (to be set only after written permission from the data provider)."""
    git = get_git_info()
    if not synthetic:
        if git["commit"] is None or git["dirty"]:
            raise SystemExit("refusing to train: git tree is dirty or not a git checkout. Commit first; a run must "
                             "trace back to one commit. (--synthetic is allowed on a dirty tree.)")
        hosted = [v for v in HOSTED_ENV if os.environ.get(v)]
        if hosted and os.environ.get("ALLOW_REAL_DATA") != "1":
            raise SystemExit(f"refusing real data on a hosted machine ({hosted[0]} is set) without ALLOW_REAL_DATA=1. "
                             "Use --synthetic, or set it only after written permission from the data provider.")
    return git


def train(config_path, data_root, out_root, resume: str = "auto", synthetic: bool = False, overrides=(),
          name: str | None = None, device: str | None = None, kill_after_step: int | None = None) -> Path:
    """Returns the run directory. `kill_after_step` exits hard (no checkpoint, no manifest update) right after that
    optimiser step; it exists only so the tests can simulate a dead session."""
    git = check_environment(synthetic)
    config_path = Path(config_path)
    cfg = prepare_config(config_path, list(overrides))
    if not synthetic:
        if data_root is None:
            raise SystemExit("--data-root (the folder holding the .acf cache) is required unless --synthetic")
        data_root = Path(data_root)
    rt = runtime_config(cfg, None if synthetic else data_root)
    st, t = cfg["steps"], cfg["train"]
    seed = int(cfg["seed"])
    seed_everything(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(t.get("amp", True)) and dev.type == "cuda"
    name = name or f"{config_path.stem}_seed{seed}" + ("_synthetic" if synthetic else "")
    run_dir = Path(out_root) / "runs" / name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets(rt, data_root, synthetic)
    val_ids = [f"SYN-{i:03d}" for i in range(len(val_ds))] if synthetic else [s["patient_id"] for s in val_ds.samples]
    subset = val_subset(val_ids, int(st["val_subset_cases"]), seed)
    lbs, mbs, accum = int(t["loader_batch_size"]), int(t["micro_batch_size"]), int(t["accum_steps"])
    per_fetch = lbs * int(cfg["patch"]["num_samples"]) // mbs
    if lbs * cfg["patch"]["num_samples"] % mbs or accum % per_fetch:
        raise SystemExit("loader_batch_size * patch.num_samples must be a multiple of micro_batch_size, and accum_steps a "
                         "multiple of the micro-batches per fetch, so that every optimiser step ends on a fetch boundary")
    n_fetch = len(train_ds) // lbs

    model = build_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))
    max_steps = int(st["max_steps"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=cfg.get("scheduler", {}).get("eta_min", 0.0))
    scaler = GradScaler(device=dev.type, enabled=use_amp)

    ident = {"config_sha256": sha256_text(config_path), "overrides": list(overrides), "synthetic": synthetic}
    s = {"step": 0, "pass": 1, "fetch": 0, "best_score": -1.0, "best_step": None, "elapsed_s": 0.0, "final_val": None}
    pending_rng = None
    have = step_checkpoints(ckpt_dir)
    if resume == "none" and have:
        raise SystemExit(f"{ckpt_dir} already holds checkpoints; use --resume auto or another --name")
    if resume != "none":
        path, ck = (load_latest(ckpt_dir) if resume == "auto" else (Path(resume), torch.load(resume, map_location="cpu", weights_only=False)))
        if ck is not None:
            if ck["ident"] != ident:
                raise SystemExit(f"checkpoint {path} was written with a different config/overrides: {ck['ident']} != {ident}")
            model.load_state_dict(ck["model_state"])
            opt.load_state_dict(ck["optimizer_state"])
            sched.load_state_dict(ck["scheduler_state"])
            scaler.load_state_dict(ck["scaler_state"])
            s, pending_rng = dict(ck["state"]), ck["rng_state"]
            for f in ("train_log.csv", "val_log.csv"):
                _truncate_csv(run_dir / f, s["step"])
            print(f"resumed from {path.name}: step {s['step']}, pass {s['pass']}, fetch {s['fetch']}, best {s['best_score']:.4f}")

    manifest = {
        "name": name, "synthetic": synthetic, "git_commit": git["commit"], "git_dirty": git["dirty"], **ident,
        "config": _repo_relative(config_path), "split_sha256": None if synthetic else sha256_text(rt["split"]["path"]),
        "conditioning_stats_sha256": None if synthetic else sha256_text(rt["conditioning"]["stats_path"]),
        "n_train": len(train_ds), "n_val": len(val_ds), "val_subset_ids": [val_ids[i] for i in subset],
        "test_set_loaded": False, "seed": seed, "max_steps": max_steps, "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                     "monai": monai.__version__, "numpy": np.__version__, "platform": platform.platform()},
        "n_params": sum(p.numel() for p in model.parameters()), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    t_session = time.monotonic()

    def elapsed() -> float:
        return s["elapsed_s"] + time.monotonic() - t_session

    def write_manifest(finished: bool = False) -> None:
        e = elapsed()
        _atomic_text(run_dir / "manifest.json", json.dumps({
            **manifest, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "finished": finished, "step": s["step"],
            "best_score": s["best_score"], "best_step": s["best_step"], "final_val": s["final_val"],
            "elapsed_hours": round(e / 3600, 4), "gpu_hours": round(e / 3600, 4) if dev.type == "cuda" else None}, indent=1))

    def save(is_best: bool) -> None:
        payload = {"model_state": model.state_dict(), "optimizer_state": opt.state_dict(), "scheduler_state": sched.state_dict(),
                   "scaler_state": scaler.state_dict(), "rng_state": _capture_rng_state(), "ident": ident, "config": cfg,
                   "state": {**s, "elapsed_s": elapsed()}, "step": s["step"], "best_score": s["best_score"]}
        _atomic_torch_save(payload, ckpt_dir / f"step_{s['step']:08d}.pt")
        if is_best:
            _atomic_torch_save(payload, ckpt_dir / "best.pt")
        for old in step_checkpoints(ckpt_dir)[:-KEEP_LAST]:
            old.unlink()
        write_manifest()

    write_manifest()
    model.train()
    opt.zero_grad(set_to_none=True)
    micro, window, last_save = 0, [], time.monotonic()
    nw = int(t.get("num_workers", 0))
    while s["step"] < max_steps:
        order = np.random.default_rng([seed, s["pass"]]).permutation(len(train_ds))[: n_fetch * lbs]
        train_ds.set_epoch(s["pass"])
        it = iter(DataLoader(Subset(train_ds, order[s["fetch"] * lbs:].tolist()), batch_size=lbs, shuffle=False,
                             num_workers=nw, collate_fn=list_data_collate, pin_memory=dev.type == "cuda"))
        if pending_rng is not None:   # after iter(): creating the iterator draws from the torch RNG in both runs
            _restore_rng_state(pending_rng)
            pending_rng = None
        for batch in it:
            s["fetch"] += 1
            stepped = False
            for mb in micro_batches(batch, mbs):
                d = {k: v.to(dev, non_blocking=True) for k, v in mb.items() if torch.is_tensor(v)}
                with autocast(device_type=dev.type, enabled=use_amp):
                    out = call_model(model, d["image"], d["cond"], d["spacing"])
                loss, _ = compute_loss(out, d["label"], cfg, epoch=s["pass"])
                scaler.scale(loss / accum).backward()
                window.append(float(loss.detach()))
                micro += 1
                if micro == accum:
                    lr = opt.param_groups[0]["lr"]
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    s["step"] += 1
                    _append_csv(run_dir / "train_log.csv", {"step": s["step"], "pass": s["pass"], "loss": float(np.mean(window)),
                                                            "lr": lr, "elapsed_s": round(elapsed(), 1)})
                    micro, window, stepped = 0, [], True
            if not stepped:
                continue
            if kill_after_step is not None and s["step"] >= kill_after_step:
                raise SystemExit(3)
            if s["step"] % int(st["val_every"]) == 0 or s["step"] >= max_steps:
                v = validate(model, val_ds, subset, cfg, dev, use_amp)
                is_best = np.isfinite(v["score"]) and v["score"] > s["best_score"]
                if is_best:
                    s["best_score"], s["best_step"] = v["score"], s["step"]
                _append_csv(run_dir / "val_log.csv", {"step": s["step"], "kind": "subset", **v, "is_best": bool(is_best)})
                print(f"step {s['step']}/{max_steps} val subset score {v['score']:.4f} best {s['best_score']:.4f}", flush=True)
                save(is_best)
                last_save = time.monotonic()
            elif time.monotonic() - last_save >= float(st["ckpt_every_minutes"]) * 60:
                save(False)
                last_save = time.monotonic()
            if s["step"] >= max_steps:
                break
        else:
            s["pass"], s["fetch"] = s["pass"] + 1, 0

    if s["final_val"] is None:   # all validation cases, once, with the weights the test script will load
        best = ckpt_dir / "best.pt"
        if best.exists():
            model.load_state_dict(torch.load(best, map_location=dev, weights_only=False)["model_state"])
        v = validate(model, val_ds, list(range(len(val_ds))), cfg, dev, use_amp)
        _append_csv(run_dir / "val_log.csv", {"step": s["best_step"] if best.exists() else s["step"], "kind": "full_best", **v, "is_best": True})
        s["final_val"] = v
    write_manifest(finished=True)
    return run_dir
