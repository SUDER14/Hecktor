"""Tiny CPU configs on the synthetic cohort, shared by scripts/smoke_test.py,
scripts/overfit_one_batch.py and tests. Pipeline checks only: nothing trained at this
scale is a result (configs carry smoke_only: true, which prints the banner)."""

from __future__ import annotations

from pathlib import Path

from src.data.synthetic import SynthConfig, write_synthetic_hecktor2026

SMOKE_KINDS = ("unet3d", "dual_unet", "acf_softmax", "acf_evidential")


def ensure_cohort(workdir: str | Path, n_per_centre: int = 2, fov: float = 48.0) -> Path:
    root = Path(workdir) / "hecktor2026_training"
    if not (root / "HECKTOR_2026_Training_Task.csv").exists():
        write_synthetic_hecktor2026(root, SynthConfig(fov_mm=(fov, fov, fov), n_per_centre=n_per_centre, seed=7))
    return root


def smoke_config(workdir: str | Path, kind: str, n_epochs: int = 2) -> dict:
    if kind not in SMOKE_KINDS:
        raise ValueError(f"kind must be one of {SMOKE_KINDS}")
    workdir = Path(workdir)
    native = kind.startswith("acf")
    ch, st = [8, 16, 32, 64], [2, 2, 2]
    if kind == "unet3d":
        model = {"name": "unet3d", "channels": ch, "strides": st, "out_channels": 2}
    elif kind == "dual_unet":
        model = {"name": "dual_unet", "channels": ch, "strides": st, "out_channels": 2, "pet_depth": 3}
    else:
        model = {"name": "acf", "channels": ch, "strides": st, "pet_channels": [8, 16], "pet_offset": 2,
                 "out_channels": 2, "head": "evidential" if kind == "acf_evidential" else "softmax",
                 "hyper_hidden": 16}
    tag = "native" if native else "resampled"
    return {
        "seed": 42, "smoke_only": True, "dataset": "hecktor2026", "data_dir": str(ensure_cohort(workdir)),
        "split": {"strategy": "group_by_centre", "test_centres": ["MDA"], "val_frac": 0.25,
                  "path": str(workdir / "split.json")},
        "conditioning": {"stats_path": str(workdir / "cond_stats.json")},
        "data": {"mode": tag},
        "cache": {"dir": str(workdir / f"cache_{tag}"), "dtype": "float32"},
        "patch": {"spatial_size": [16, 16, 16], "pos": 2, "neg": 1, "num_samples": 2, "extent_tol_voxels": 1.0},
        "intensity": {"ct_window": {"min": -1000, "max": 400}, "pet": {"suv_clip_max": 10.0}},
        "baseline": {"resample_spacing": [1.0, 1.0, 1.0]},
        "model": model,
        "loss": {"smooth": 1.0, "kl_anneal_epochs": 2, "dice_weight": 1.0},
        "train": {"n_epochs": n_epochs, "lr": 0.003, "weight_decay": 0.0, "micro_batch_size": 2, "accum_steps": 2,
                  "loader_batch_size": 1, "amp": False, "num_workers": 0, "val_every": 1, "device": "cpu",
                  "val_max_cases": 1},
        "scheduler": {"name": "cosine", "eta_min": 1.0e-5},
        "checkpoint": {"keep_last_n": 2},
        "infer": {"overlap": 0.0, "sw_batch_size": 4},
        "run": {"dir": str(workdir / f"run_{kind}")},
    }
