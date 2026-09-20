"""Resumable checkpointing.

Training runs on Colab Pro (see CLAUDE.md); Colab sessions die without
warning, so a training loop without resumable checkpoints is not finished.
A checkpoint must carry everything needed to resume byte-for-byte, not just
model weights: optimizer state (momentum/Adam moments), scaler state (AMP
loss-scale), scheduler state, the epoch counter, the best metric seen so far,
and RNG state for python/numpy/torch(+CUDA) -- without the RNG state, a
resumed run draws a different sequence of augmentations/dropout/shuffles than
an uninterrupted one would have, which is exactly the kind of silent
divergence CLAUDE.md's reproducibility requirement rules out.

checkpoint_dir is a plain directory path here -- on Colab that path should
point into a mounted Google Drive, but that's an environment/config concern
(set it in the training config), not something this module needs to know.

Checkpoints accumulate one file per epoch if nothing prunes them, which is
wasteful on Drive quota over 15-25 experiments. prune_checkpoints keeps only
the most recent N epoch checkpoints; the best-metric checkpoint is saved
separately (best.pt) so pruning never discards it.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def _capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Writes to a hidden temp file in the same directory, fsyncs, then renames
    over the target. On Google Drive's FUSE mount a session can die mid-write;
    a plain torch.save would leave a truncated epoch_NNNN.pt that looks like
    the newest checkpoint and breaks --resume. With the rename, the final name
    only ever points at a fully written file."""
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_checkpoint(path, epoch, model, optimizer, scaler, best_metric, config, scheduler=None, extra=None) -> None:
    """extra: caller-defined keys merged into the saved dict (e.g. a wandb run
    id, so a resumed run keeps logging to the same wandb run instead of
    fragmenting its history). Must not collide with the fixed keys above."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "rng_state": _capture_rng_state(),
        "config": config,
    }
    if extra:
        overlap = set(extra) & set(payload)
        if overlap:
            raise ValueError(f"extra keys collide with fixed checkpoint keys: {overlap}")
        payload.update(extra)
    _atomic_torch_save(payload, path)


def load_checkpoint(
    path, model, optimizer=None, scaler=None, scheduler=None, map_location="cpu", restore_rng=True
) -> dict:
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if restore_rng and ckpt.get("rng_state") is not None:
        _restore_rng_state(ckpt["rng_state"])
    return ckpt


def list_checkpoints(checkpoint_dir) -> list[Path]:
    """epoch_*.pt files in checkpoint_dir, oldest first. Empty if the directory
    doesn't exist. Hidden .tmp files from an interrupted save don't match."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def latest_checkpoint(checkpoint_dir) -> Path | None:
    """Returns the highest-epoch checkpoint file in checkpoint_dir, or None if
    the directory doesn't exist or has none -- callers use this to decide
    whether a --resume actually has anything to resume from. Does not check the
    file is readable; use load_latest_valid_checkpoint for that."""
    found = list_checkpoints(checkpoint_dir)
    return found[-1] if found else None


def load_latest_valid_checkpoint(
    checkpoint_dir, model, optimizer=None, scaler=None, scheduler=None, map_location="cpu"
):
    """Loads the newest checkpoint that can actually be read, falling back to
    older ones. Returns (path, ckpt) or (None, None) if nothing loadable.

    Drive can hand back a file that exists but is truncated or unreadable
    (a session killed mid-write from before atomic saves, or a sync that hasn't
    finished). Failing the whole resume on that would strand the run when
    keep_last_n older checkpoints are sitting right there."""
    for path in reversed(list_checkpoints(checkpoint_dir)):
        try:
            return path, load_checkpoint(path, model, optimizer, scaler, scheduler, map_location=map_location)
        except Exception as e:  # noqa: BLE001 -- any unreadable file means try the next one
            print(f"! could not load {path} ({type(e).__name__}: {e}); trying an older checkpoint")
    return None, None


def save_best_checkpoint(checkpoint_dir, epoch_path) -> Path:
    """Copies the checkpoint just written for this epoch to best.pt. Callers
    decide "best" (e.g. val_dice improved) before calling this -- this
    function just does the copy."""
    checkpoint_dir = Path(checkpoint_dir)
    best_path = checkpoint_dir / "best.pt"
    tmp = checkpoint_dir / ".best.pt.tmp"
    try:
        shutil.copyfile(epoch_path, tmp)  # copyfile, not copy2: no metadata copy, one fewer thing that can fail on a network mount
        os.replace(tmp, best_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return best_path


def prune_checkpoints(checkpoint_dir, keep_last_n: int) -> None:
    """Deletes epoch_*.pt files beyond the keep_last_n most recent. Never
    touches best.pt, which lives outside the epoch_*.pt naming pattern."""
    checkpoint_dir = Path(checkpoint_dir)
    if keep_last_n <= 0:
        return
    candidates = list_checkpoints(checkpoint_dir)
    for stale in candidates[:-keep_last_n]:
        stale.unlink()
