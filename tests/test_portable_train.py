"""Portable step-based trainer (src/training/portable.py, scripts/train.py) on synthetic tensors, CPU only."""
import csv
import json
from pathlib import Path

import pytest
import torch

from src.data.splits import load_split
from src.training import portable

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "shared_b4_acf_cond.yaml"
SMALL = ["patch.spatial_size=[32, 32, 32]", "patch.min_extent_mm=[20.0, 20.0, 20.0]", "train.accum_steps=2", "train.num_workers=0", "steps.max_steps=8",
         "steps.val_every=3", "steps.val_subset_cases=2", "steps.ckpt_every_minutes=1000"]


def _losses(run_dir):
    with open(run_dir / "train_log.csv", newline="") as f:
        return [(int(r["step"]), r["loss"], r["lr"]) for r in csv.DictReader(f)]


def test_killed_and_resumed_run_equals_the_uninterrupted_run(tmp_path):
    straight = portable.train(CFG, None, tmp_path / "a", synthetic=True, overrides=SMALL, device="cpu")
    with pytest.raises(SystemExit) as e:   # dies after step 5; the last checkpoint is the validation at step 3
        portable.train(CFG, None, tmp_path / "b", synthetic=True, overrides=SMALL, device="cpu", kill_after_step=5)
    assert e.value.code == 3
    killed = tmp_path / "b" / "runs" / straight.name
    assert [p.name for p in portable.step_checkpoints(killed / "checkpoints")] == ["step_00000003.pt"]
    assert len(_losses(killed)) == 5 and not json.loads((killed / "manifest.json").read_text())["finished"]
    resumed = portable.train(CFG, None, tmp_path / "b", resume="auto", synthetic=True, overrides=SMALL, device="cpu")

    assert _losses(resumed) == _losses(straight) and [s for s, _, _ in _losses(resumed)] == list(range(1, 9))
    wa = torch.load(straight / "checkpoints" / "step_00000008.pt", weights_only=False)
    wb = torch.load(resumed / "checkpoints" / "step_00000008.pt", weights_only=False)
    assert all(torch.equal(wa["model_state"][k], wb["model_state"][k]) for k in wa["model_state"])
    assert wa["state"]["best_score"] == wb["state"]["best_score"] and wa["state"]["pass"] == wb["state"]["pass"]
    for key in ("optimizer_state", "scheduler_state", "scaler_state", "rng_state", "step", "best_score"):
        assert key in wb

    # last 2 step checkpoints plus best; logs and manifest are complete
    names = sorted(p.name for p in (resumed / "checkpoints").iterdir())
    assert names == ["best.pt", "step_00000006.pt", "step_00000008.pt"]
    m = json.loads((resumed / "manifest.json").read_text())
    assert m["finished"] and m["step"] == 8 and m["synthetic"] and m["test_set_loaded"] is False
    assert m["final_val"]["n_cases"] == 4 and len(m["val_subset_ids"]) == 2 and len(m["config_sha256"]) == 64
    with open(resumed / "val_log.csv", newline="") as f:
        kinds = [(r["step"], r["kind"]) for r in csv.DictReader(f)]
    assert [k for _, k in kinds] == ["subset", "subset", "subset", "full_best"] and [s for s, _ in kinds][:3] == ["3", "6", "8"]


def test_resume_refuses_a_checkpoint_from_other_settings(tmp_path):
    two = [o for o in SMALL if "max_steps" not in o and "val_every" not in o] + ["steps.max_steps=2", "steps.val_every=2"]
    portable.train(CFG, None, tmp_path, synthetic=True, overrides=two, device="cpu")
    with pytest.raises(SystemExit, match="different config"):
        portable.train(CFG, None, tmp_path, synthetic=True, overrides=SMALL, device="cpu")
    with pytest.raises(SystemExit, match="already holds checkpoints"):
        portable.train(CFG, None, tmp_path, resume="none", synthetic=True, overrides=two, device="cpu")


def test_real_run_refuses_dirty_tree_and_hosted_machine_without_permission(monkeypatch, tmp_path):
    monkeypatch.setattr(portable, "get_git_info", lambda: {"commit": "abc", "dirty": True})
    with pytest.raises(SystemExit, match="dirty"):
        portable.train(CFG, tmp_path, tmp_path, overrides=SMALL)
    monkeypatch.setattr(portable, "get_git_info", lambda: {"commit": None, "dirty": None})
    with pytest.raises(SystemExit, match="dirty or not a git checkout"):
        portable.train(CFG, tmp_path, tmp_path, overrides=SMALL)
    monkeypatch.setattr(portable, "get_git_info", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setenv("COLAB_RELEASE_TAG", "x")
    monkeypatch.delenv("ALLOW_REAL_DATA", raising=False)
    with pytest.raises(SystemExit, match="ALLOW_REAL_DATA"):
        portable.train(CFG, tmp_path, tmp_path, overrides=SMALL)


def test_only_train_and_validation_ids_are_selected():
    split = load_split(ROOT / "configs" / "splits" / "hecktor2026_seed42.json")
    train, val = portable.select_ids(split)
    assert len(train) == 592 and len(val) == 67 and not (set(train) | set(val)) & set(split["test"])
    with pytest.raises(RuntimeError, match="test ids appear"):
        portable.select_ids({"train": ["a"], "val": ["b"], "test": ["b"]})
    assert portable.val_subset(val, 20, 42) == portable.val_subset(val, 20, 42) and len(portable.val_subset(val, 20, 42)) == 20


def test_all_six_run_configs_share_one_sampling_ratio_and_step_settings():
    import yaml
    cfgs = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "configs").glob("shared_b*.yaml"))]
    assert len(cfgs) == 6
    for key in (("patch", "pos"), ("patch", "neg"), ("patch", "num_samples"), ("steps",), ("train", "accum_steps"),
                ("train", "micro_batch_size")):
        vals = []
        for c in cfgs:
            v = c
            for k in key:
                v = v[k]
            vals.append(json.dumps(v, sort_keys=True))
        assert len(set(vals)) == 1, key
