"""Resume robustness for the Colab/Drive workflow.

tests/test_checkpoint.py replays the epoch loop by hand. These tests drive the
real train() entry point: interrupt, resume, and compare against an
uninterrupted run -- including the case where the newest checkpoint on disk is
truncated (what a session killed mid-write to Drive used to leave behind).

Local paths only: there is no Google Drive mount in this environment, so
nothing here exercises real Drive latency or FUSE behaviour. Paths with spaces
and nested directories are used to catch plain path-handling bugs.
"""

from __future__ import annotations

import re

import pytest
import torch

import src.training.train as train_mod
from src.training.checkpoint import (
    list_checkpoints,
    load_latest_valid_checkpoint,
    save_checkpoint,
)
from tests.test_checkpoint import DEVICE, _build_run, _make_config, _TinyVolumes

LOSS_RE = re.compile(r"epoch (\d+)/\d+\s+train_loss=([0-9.]+)")


def _losses(text: str) -> dict[int, float]:
    return {int(e): float(v) for e, v in LOSS_RE.findall(text)}


@pytest.fixture
def tiny_datasets(monkeypatch):
    ds = _TinyVolumes()
    monkeypatch.setattr(train_mod, "build_datasets", lambda config: {"train": ds, "val": ds})


def _cfg(ckpt_dir, n_epochs):
    cfg = _make_config(ckpt_dir, total_epochs=4)
    cfg["train"]["n_epochs"] = n_epochs
    return cfg


def test_train_resume_matches_uninterrupted(tiny_datasets, tmp_path, capsys):
    train_mod.train(_cfg(tmp_path / "ref", 4), resume=False)
    ref = _losses(capsys.readouterr().out)
    assert sorted(ref) == [1, 2, 3, 4]

    drive_like = tmp_path / "My Drive" / "hecktor project" / "ckpts"
    train_mod.train(_cfg(drive_like, 2), resume=False)
    first = _losses(capsys.readouterr().out)
    train_mod.train(_cfg(drive_like, 4), resume=True)
    out = capsys.readouterr().out
    second = _losses(out)

    assert "resumed from" in out and sorted(second) == [3, 4]
    for e, v in {**first, **second}.items():
        assert v == pytest.approx(ref[e], abs=2e-4), f"epoch {e}: resumed {v} vs reference {ref[e]}"


def test_train_resume_survives_truncated_latest_checkpoint(tiny_datasets, tmp_path, capsys):
    train_mod.train(_cfg(tmp_path / "ref", 4), resume=False)
    ref = _losses(capsys.readouterr().out)

    ckpts = tmp_path / "ckpts"
    train_mod.train(_cfg(ckpts, 2), resume=False)
    capsys.readouterr()
    latest = ckpts / "epoch_0002.pt"
    latest.write_bytes(latest.read_bytes()[:100])  # killed mid-write

    train_mod.train(_cfg(ckpts, 4), resume=True)
    out = capsys.readouterr().out
    second = _losses(out)

    assert "could not load" in out and "resumed from" in out
    assert sorted(second) == [2, 3, 4]  # fell back to epoch 1 and redid epoch 2
    for e, v in second.items():
        assert v == pytest.approx(ref[e], abs=2e-4), f"epoch {e}: resumed {v} vs reference {ref[e]}"


def test_interrupted_save_does_not_clobber_or_leave_temp(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    model, _, optimizer, scaler, scheduler = _build_run(config)
    path = tmp_path / "epoch_0001.pt"
    save_checkpoint(path, 1, model, optimizer, scaler, -1.0, config, scheduler=scheduler)
    good = path.read_bytes()

    def boom(*a, **k):
        raise OSError("session died")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError):
        save_checkpoint(path, 2, model, optimizer, scaler, -1.0, config, scheduler=scheduler)

    assert path.read_bytes() == good
    assert [p.name for p in tmp_path.iterdir()] == ["epoch_0001.pt"]


def test_no_loadable_checkpoint_returns_none(tmp_path):
    config = _make_config(tmp_path)
    model, _, optimizer, scaler, scheduler = _build_run(config)
    (tmp_path / "epoch_0001.pt").write_bytes(b"garbage")
    assert load_latest_valid_checkpoint(tmp_path, model, optimizer, scaler, scheduler, map_location=DEVICE) == (None, None)
    assert [p.name for p in list_checkpoints(tmp_path)] == ["epoch_0001.pt"]


def test_apply_overrides_sets_value_and_rejects_typos():
    cfg = {"train": {"n_epochs": 100, "lr": 1e-4}, "checkpoint": {"dir": "a"}}
    train_mod.apply_overrides(cfg, ["train.n_epochs=2", "checkpoint.dir=/content/drive/My Drive/x"])
    assert cfg["train"]["n_epochs"] == 2 and cfg["checkpoint"]["dir"] == "/content/drive/My Drive/x"
    with pytest.raises(SystemExit):
        train_mod.apply_overrides(cfg, ["train.n_epoch=2"])
    with pytest.raises(SystemExit):
        train_mod.apply_overrides(cfg, ["nosuch.key=1"])


def test_get_git_info_shape():
    info = train_mod.get_git_info()
    assert set(info) == {"commit", "dirty"}
    assert info["commit"] is None or re.fullmatch(r"[0-9a-f]{40}", info["commit"])


def test_production_config_uses_adamw_1e4():
    cfg = train_mod.load_config("configs/train_baseline_decathlon_lung.yaml")
    model = train_mod.build_model(cfg)
    opt = train_mod.build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)
