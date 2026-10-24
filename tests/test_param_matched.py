"""B2b (configs/b2b_dual_matched.yaml) must stay within 3% of ACF's parameter count, and differ from B2
only in channel widths. Counts are printed (run with -s)."""
from pathlib import Path

import yaml

from src.training.run import build_model

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ACF_PARAMS = 6_229_986  # B3/B4/B5, recorded in docs/FINAL_REPORT.md
TOL = 0.03


def _load(name):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


def _n(cfg):
    return sum(p.numel() for p in build_model(cfg).parameters())


def test_b2b_params_within_3pct_of_acf():
    acf, b2, b2b = _n(_load("b4_acf_cond")), _n(_load("b2_dual_resample")), _n(_load("b2b_dual_matched"))
    print(f"ACF {acf:,} | B2 {b2:,} | B2b {b2b:,} ({(b2b - acf) / acf:+.2%} vs ACF)")
    assert acf == ACF_PARAMS
    assert abs(b2b - acf) / acf <= TOL


def test_b2b_differs_from_b2_only_in_widths():
    b2, b2b = _load("b2_dual_resample"), _load("b2b_dual_matched")
    assert b2b["model"]["channels"] != b2["model"]["channels"]
    for cfg in (b2, b2b):
        cfg["model"].pop("channels")
        cfg["run"].pop("dir")
    assert b2 == b2b
