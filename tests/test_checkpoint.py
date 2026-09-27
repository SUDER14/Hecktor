"""Tests for src/training/checkpoint.py's save/resume mechanics.

The scenario CLAUDE.md's resumable-checkpointing requirement cares about:
Colab dies mid-run. TestResumeContinuesLossCurve trains 2 epochs, "kills" the
process (drops the in-memory model/optimizer/scaler/scheduler and rebuilds
fresh ones, keeping only the checkpoint file on disk -- exactly what survives
a Colab disconnect), resumes from that checkpoint, and confirms the loss
curve is continuous against an uninterrupted 4-epoch reference run of the
same seed.

Uses a tiny synthetic in-memory dataset (not real Task06_Lung) so this runs
in seconds and doesn't depend on the multi-GB download being present --
exactly the same reasoning tests/test_adapters.py uses for HecktorAdapter.
This test is about checkpoint.py's contract, not about segmentation quality,
so a handful of small random volumes through a tiny 2-level UNet is enough.
"""

from __future__ import annotations

import pytest
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from src.data.seed import seed_everything
from src.training.checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_best_checkpoint,
    save_checkpoint,
)
from src.training.train import build_loss, build_model, build_optimizer, build_scheduler, train_one_epoch

DEVICE = torch.device("cpu")


class _TinyVolumes(Dataset):
    """Fixed (non-reseeded-per-run) synthetic volumes -- the randomness this
    test cares about is the training loop's own RNG consumption (DataLoader
    shuffle order), not the data itself."""

    def __init__(self, n=6, shape=(1, 16, 16, 16)):
        g = torch.Generator().manual_seed(0)
        self.images = [torch.randn(shape, generator=g) for _ in range(n)]
        self.labels = [(torch.rand(shape, generator=g) > 0.7).long() for _ in range(n)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": self.labels[idx]}


def _make_config(ckpt_dir, seed=123, total_epochs=4):
    return {
        "seed": seed,
        "model": {
            "name": "baseline_unet", "in_channels": 1, "out_channels": 2,
            "channels": [4, 8], "strides": [2], "num_res_units": 1,
        },
        "loss": {"include_background": True, "lambda_dice": 1.0, "lambda_ce": 1.0},
        "train": {
            "device": "cpu", "amp": False, "lr": 1e-2, "weight_decay": 0.0,
            "n_epochs": total_epochs, "val_every": 1,
            "loader_batch_size": 2, "micro_batch_size": 2, "accum_steps": 1,
            "sw_batch_size": 1, "num_workers": 0,
        },
        "scheduler": {"name": "cosine", "t_max": total_epochs, "eta_min": 1e-5},
        "patch": {"spatial_size": [16, 16, 16]},
        "baseline": {"resample_spacing": [1.0, 1.0, 1.0]},
        "checkpoint": {"dir": str(ckpt_dir), "keep_last_n": 5},
    }


def _build_run(config):
    """Fresh model/optimizer/scaler/scheduler, as a new process would build
    them -- caller is responsible for seeding beforehand."""
    model = build_model(config).to(DEVICE)
    loss_fn = build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler(device="cpu", enabled=False)
    return model, loss_fn, optimizer, scaler, scheduler


def _run_epochs(dataset, config, model, loss_fn, optimizer, scaler, scheduler, start, end):
    loader = DataLoader(dataset, batch_size=config["train"]["loader_batch_size"], shuffle=True)
    losses = []
    for _ in range(start, end + 1):
        loss = train_one_epoch(model, loader, loss_fn, optimizer, scaler, DEVICE, config)
        if scheduler is not None:
            scheduler.step()
        losses.append(loss)
    return losses


class TestResumeContinuesLossCurve:
    def test_resumed_losses_match_uninterrupted_reference(self, tmp_path):
        dataset = _TinyVolumes()

        # Reference: 4 epochs, uninterrupted, one process.
        ref_config = _make_config(tmp_path / "ref_ckpt", total_epochs=4)
        seed_everything(ref_config["seed"])
        ref_model, ref_loss_fn, ref_opt, ref_scaler, ref_sched = _build_run(ref_config)
        ref_losses = _run_epochs(dataset, ref_config, ref_model, ref_loss_fn, ref_opt, ref_scaler, ref_sched, 1, 4)

        # Resumed: epochs 1-2, checkpoint, "kill" everything, rebuild fresh,
        # resume, epochs 3-4.
        config = _make_config(tmp_path / "resume_ckpt", total_epochs=4)
        seed_everything(config["seed"])
        model, loss_fn, optimizer, scaler, scheduler = _build_run(config)
        losses_1_2 = _run_epochs(dataset, config, model, loss_fn, optimizer, scaler, scheduler, 1, 2)

        ckpt_path = tmp_path / "resume_ckpt" / "epoch_0002.pt"
        save_checkpoint(ckpt_path, 2, model, optimizer, scaler, best_metric=-1.0, config=config, scheduler=scheduler)

        del model, loss_fn, optimizer, scaler, scheduler  # simulate the Colab process dying

        resumed_model, resumed_loss_fn, resumed_optimizer, resumed_scaler, resumed_scheduler = _build_run(config)
        ckpt = load_checkpoint(
            ckpt_path, resumed_model, resumed_optimizer, resumed_scaler, resumed_scheduler, map_location=DEVICE
        )
        assert ckpt["epoch"] == 2
        start_epoch = ckpt["epoch"] + 1
        assert start_epoch == 3

        losses_3_4 = _run_epochs(
            dataset, config, resumed_model, resumed_loss_fn, resumed_optimizer, resumed_scaler, resumed_scheduler,
            start_epoch, 4,
        )

        resumed_losses = losses_1_2 + losses_3_4
        for epoch_idx, (ref, resumed) in enumerate(zip(ref_losses, resumed_losses), start=1):
            assert resumed == pytest.approx(ref, rel=1e-4, abs=1e-6), (
                f"epoch {epoch_idx} loss diverged after resume: "
                f"reference={ref!r} resumed={resumed!r}"
            )

        for (name, ref_p), (_, resumed_p) in zip(
            ref_model.state_dict().items(), resumed_model.state_dict().items()
        ):
            assert torch.allclose(ref_p, resumed_p, rtol=1e-4, atol=1e-6), f"param {name} diverged after resume"


class TestCheckpointPruning:
    def test_keeps_last_n_and_never_touches_best(self, tmp_path):
        model, loss_fn, optimizer, scaler, scheduler = _build_run(_make_config(tmp_path))
        config = _make_config(tmp_path)

        paths = []
        for epoch in range(1, 6):
            path = tmp_path / f"epoch_{epoch:04d}.pt"
            save_checkpoint(path, epoch, model, optimizer, scaler, best_metric=0.5, config=config, scheduler=scheduler)
            paths.append(path)
        save_best_checkpoint(tmp_path, paths[1])  # pretend epoch 2 was best
        prune_checkpoints(tmp_path, keep_last_n=2)

        remaining = sorted(p.name for p in tmp_path.glob("epoch_*.pt"))
        assert remaining == ["epoch_0004.pt", "epoch_0005.pt"]
        assert (tmp_path / "best.pt").exists()
        assert latest_checkpoint(tmp_path).name == "epoch_0005.pt"

    def test_latest_checkpoint_none_when_dir_missing(self, tmp_path):
        assert latest_checkpoint(tmp_path / "does_not_exist") is None
