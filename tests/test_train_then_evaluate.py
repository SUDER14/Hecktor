"""Integration: scripts/train.py --synthetic, then scripts/evaluate_test.py --synthetic on its output folder, and
the run-folder keys the test script reads. Plus the sliding-window probability regression test."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eval import locked_test
from src.eval.infer import sliding_window
from src.training import portable

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["patch.spatial_size=[32, 32, 32]", "patch.min_extent_mm=[8.0, 8.0, 8.0]", "train.accum_steps=1",
         "train.num_workers=0", "steps.max_steps=2", "steps.val_every=2", "steps.val_subset_cases=2"]


def test_train_synthetic_then_evaluate_synthetic(tmp_path):
    run_dir = portable.train(ROOT / "configs" / "shared_b4_acf_cond.yaml", None, tmp_path / "out", synthetic=True,
                             overrides=SMALL, device="cpu")
    # the keys scripts/evaluate_test.py reads from a run folder written by scripts/train.py
    m = json.loads((run_dir / "manifest.json").read_text())
    for key in ("finished", "synthetic", "split_sha256", "conditioning_stats_sha256", "git_commit", "config_sha256",
                "best_step", "max_steps", "seed"):
        assert key in m, key
    assert m["finished"] is True
    ck = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    for key in ("model_state", "config", "step", "best_score"):
        assert key in ck, key
    _model, cfg, info = locked_test.load_best(run_dir, torch.device("cpu"))
    _no_machine_path(ck["config"])                      # nothing of the training machine is stored in best.pt
    assert str(tmp_path) not in json.dumps(ck["config"]) and str(tmp_path) not in json.dumps(m["config"])
    assert info["step"] == 2 and cfg["model"]["name"] == "acf"

    res = locked_test.evaluate(run_dir, None, tmp_path / "eval", synthetic=True, device="cpu")
    assert res.name == run_dir.name
    for f in ("per_case.csv", "sweep.csv", "wrong_spacing.csv", "calibration.json", "uncertainty.json", "summary.json",
              "manifest.json"):
        assert (res / f).exists(), f
    assert len(pd.read_csv(res / "per_case.csv")) >= 5


def _no_machine_path(obj, where=""):
    """No absolute path (drive letter, leading slash or backslash) anywhere in a stored config."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _no_machine_path(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _no_machine_path(v, where)
    elif isinstance(obj, str):
        assert not Path(obj).is_absolute() and ":" not in obj[:3] and "\\" not in obj, f"{where} = {obj!r}"


def test_trainer_text_hashes_do_not_depend_on_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.json", tmp_path / "crlf.json"
    lf.write_bytes(b'{\n "train": ["a"]\n}\n')
    crlf.write_bytes(b'{\r\n "train": ["a"]\r\n}\r\n')
    assert portable.sha256_text(lf) == portable.sha256_text(crlf) == locked_test.sha256_text(crlf)
    assert portable.sha256_file(lf) != portable.sha256_file(crlf)          # the raw-byte hash is what used to be recorded
    split = ROOT / "configs" / "splits" / "hecktor2026_seed42.json"
    stats = ROOT / "configs" / "splits" / "hecktor2026_seed42_conditioning_stats.json"
    assert portable.sha256_text(split) == locked_test.sha256_text(split)
    # a manifest written by the trainer passes the test script's check as it stands
    run = tmp_path / "r"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"finished": True, "synthetic": False, "split_sha256": portable.sha256_text(split),
                                                   "conditioning_stats_sha256": portable.sha256_text(stats)}))
    assert locked_test._check_run_manifest(run, split, stats)["finished"]


def test_stored_config_has_no_machine_path_and_the_runtime_copy_has(tmp_path):
    cfg = portable.prepare_config(ROOT / "configs" / "shared_b1_unet_resample.yaml", [])
    _no_machine_path(cfg)
    assert cfg["split"]["path"] == "configs/splits/hecktor2026_seed42.json" and cfg["cache"]["dir"] is None
    assert cfg.get("data_dir") is None and cfg["run"]["dir"] is None
    rt = portable.runtime_config(cfg, tmp_path)
    assert Path(rt["split"]["path"]).is_file() and rt["cache"]["dir"] == str(tmp_path) and cfg["cache"]["dir"] is None
    # the test script turns the stored config into a local one without complaint, and it points at this checkout
    loc = locked_test.localise_config(cfg, tmp_path)
    assert (ROOT / loc["split"]["path"]).is_file() and (ROOT / loc["conditioning"]["stats_path"]).is_file()
    assert portable._repo_relative(ROOT / "configs" / "shared_b1_unet_resample.yaml") == "configs/shared_b1_unet_resample.yaml"
    assert portable._repo_relative(tmp_path / "x.yaml") == "x.yaml"


def test_sliding_window_returns_the_model_probabilities_not_a_second_softmax():
    logits = torch.tensor([4.0, 0.0, -1.0])

    def fwd(x):
        return logits.view(1, 3, 1, 1, 1).expand(x.shape[0], 3, *x.shape[2:]).clone()

    r = sliding_window(fwd, np.zeros((2, 12, 8, 8), np.float32), (8, 8, 8), 3, "softmax", 0.5, 1, "cpu")
    want = torch.softmax(logits, 0).numpy()
    np.testing.assert_allclose(r["p"][:, 3, 3, 3], want, atol=1e-6)
    np.testing.assert_allclose(r["p"].sum(0), 1.0, atol=1e-6)
    ent = float(-(want * np.log(want)).sum() / np.log(3))
    np.testing.assert_allclose(r["u"][0, 3, 3, 3], ent, atol=1e-5)
