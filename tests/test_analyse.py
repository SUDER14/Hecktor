"""scripts/analyse.py on synthetic per_case tables with known answers. No real result is read."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import analyse as an

N = 10
IDS = [f"T{i:02d}" for i in range(N)]
BASE = 0.50 + 0.02 * np.arange(N)                       # B3
STEP = 0.001 * np.arange(N)                             # distinct magnitudes, so the exact Wilcoxon test applies
EXACT_ALL_POSITIVE = 2 / 2 ** N                         # two-sided exact p when all 10 differences are positive


def write_run(root: Path, name: str, mean_dice, synthetic=None, extra_case: bool = True) -> None:
    d = root / name
    d.mkdir(parents=True)
    df = pd.DataFrame({"patient_id": IDS, "centre": "CHUS", "mean_dice": mean_dice, "dice_1": mean_dice,
                       "dice_2": mean_dice, "hd95_mm_1": 10 - 5 * np.asarray(mean_dice), "label_present_1": True})
    if extra_case:   # a case with neither label in the truth: no per-case mean Dice in any model
        df = pd.concat([df, pd.DataFrame({"patient_id": ["T99"], "centre": ["CHUV"], "mean_dice": [np.nan], "dice_1": [np.nan],
                                          "dice_2": [np.nan], "hd95_mm_1": [np.nan], "label_present_1": [False]})], ignore_index=True)
    df.to_csv(d / "per_case.csv", index=False)
    if synthetic is not None:
        (d / "manifest.json").write_text(json.dumps({"synthetic": synthetic}))


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "test"
    b4 = BASE + 0.05 + STEP                                           # clear difference to B3, all positive
    write_run(r, "shared_b3_acf_zeroed_seed42", BASE)
    write_run(r, "shared_b4_acf_cond_seed42", b4 - 0.02)              # two seeds of B4 whose per-case mean is b4
    write_run(r, "shared_b4_acf_cond_seed43", b4 + 0.02)
    write_run(r, "shared_b2b_dual_matched_seed42", b4)                # no difference to B4 at all
    sign = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    write_run(r, "shared_b2_dual_resample_seed42", b4 - sign * (0.001 + STEP))   # mixed signs: W+ = 25, W- = 30
    write_run(r, "shared_b1_unet_resample_seed42", b4 - sign * (0.001 + STEP) - 0.10 - STEP)   # B2 - B1 all positive
    write_run(r, "nnunet_fold0", b4 + 0.01)
    return r                                                          # B5 deliberately absent


def run(root):
    runs = an.discover(root, [])
    tables, info = an.load_tables(root, runs)
    return an.analyse(tables), info, runs


def row(res, comparison, metric="mean_dice"):
    return res[(res["comparison"] == comparison) & (res["metric"] == metric)].iloc[0]


def test_runs_are_matched_to_models_and_seeds_are_averaged_per_case_first(root):
    res, info, runs = run(root)
    assert runs["B2b"] == ["shared_b2b_dual_matched_seed42"] and runs["B2"] == ["shared_b2_dual_resample_seed42"]
    assert len(runs["B4"]) == 2 and info["models"]["B4"]["n_seeds"] == 2 and "B5" not in runs
    r = row(res, "B4 vs B3")
    assert r["n_pairs"] == N                                # 10 cases, not 20 case-seed rows
    assert r["n_dropped_nan"] == 1                          # the case with no label in the truth, counted
    assert r["mean_diff"] == pytest.approx(0.05 + STEP.mean())   # the +-0.02 seed offsets cancel


def test_clear_difference_known_answers(root):
    res, _, _ = run(root)
    r = row(res, "B4 vs B3")
    assert r["role"] == "primary" and r["tested"] and r["status"] == "ok"
    assert r["median_diff"] == pytest.approx(0.05 + np.median(STEP))          # 0.0545
    assert 0.05 <= r["median_ci_low"] <= r["median_diff"] <= r["median_ci_high"] <= 0.05 + STEP.max() + 1e-12
    assert r["p_value"] == pytest.approx(EXACT_ALL_POSITIVE)                  # 2 / 1024
    assert r["rank_biserial"] == pytest.approx(1.0)
    assert r["p_holm"] == pytest.approx(5 * EXACT_ALL_POSITIVE) and r["significant"] and r["holm_m"] == 5


def test_no_difference_known_answers(root):
    res, _, _ = run(root)
    r = row(res, "B4 vs B2b")
    assert r["median_diff"] == 0.0 and r["mean_diff"] == pytest.approx(0.0) and r["p_value"] == 1.0
    assert r["p_holm"] == 1.0 and not r["significant"] and r["rank_biserial"] == 0.0
    mixed = row(res, "B4 vs B2")                              # W+ = 25, W- = 30 of 55: nowhere near significant
    assert mixed["rank_biserial"] == pytest.approx((25 - 30) / 55) and mixed["p_value"] > 0.5 and not mixed["significant"]


def test_holm_family_is_always_five_and_a_missing_model_stays_in_the_count(root):
    res, _, _ = run(root)
    fam = res[(res["metric"] == "mean_dice") & res["tested"]]
    assert list(fam["comparison"]) == ["B4 vs B3", "B4 vs B2b", "B4 vs B2", "B5 vs B4", "B2 vs B1"] and (fam["holm_m"] == 5).all()
    b5 = row(res, "B5 vs B4")
    assert b5["status"].startswith("not run") and np.isnan(b5["p_value"]) and np.isnan(b5["p_holm"]) and not b5["significant"]
    # hand-worked Holm with m = 5: the two smallest p are both 2/1024 -> 5p, then max(5p, 4p) = 5p
    assert row(res, "B2 vs B1")["p_value"] == pytest.approx(EXACT_ALL_POSITIVE)
    assert row(res, "B2 vs B1")["p_holm"] == pytest.approx(5 * EXACT_ALL_POSITIVE)
    assert row(res, "B4 vs B2")["p_holm"] == 1.0              # 3 x p > 1, capped


def test_descriptive_and_secondary_rows_carry_no_p_value(root):
    res, _, _ = run(root)
    nn = row(res, "nnU-Net vs B4")
    assert nn["role"] == "descriptive" and not nn["tested"] and np.isnan(nn["p_value"]) and np.isnan(nn["p_holm"])
    assert nn["median_diff"] == pytest.approx(0.01)
    sec = res[res["metric"] != "mean_dice"]
    assert sec["p_value"].isna().all() and sec["p_holm"].isna().all() and not sec["tested"].any()
    hd = row(res, "B4 vs B3", "hd95_mm_1")                    # hd95 = 10 - 5 * dice in the fixture
    assert hd["median_diff"] == pytest.approx(-5 * (0.05 + np.median(STEP)))
    assert row(res, "B4 vs B3", "surface_dice_2mm_1")["status"].startswith("not run: column")


def test_command_line_writes_csv_and_summary_and_refuses_bad_inputs(root, tmp_path, monkeypatch, capsys):
    out = tmp_path / "stats"
    monkeypatch.setattr(sys, "argv", ["analyse.py", "--results-root", str(root), "--out", str(out)])
    an.main()
    csv = pd.read_csv(out / "comparisons.csv")
    assert len(csv) == 6 * (1 + len(an.SECONDARY_METRICS)) and (out / "comparisons_inputs.json").exists()
    text = (out / "comparisons_summary.txt").read_text()
    assert "B4 vs B3 (primary)" in text and "significant at 0.05 after Holm" in text and "B5 vs B4 (secondary): not run" in text
    assert "nnU-Net vs B4 (descriptive)" in text and "SYNTHETIC" not in text and str(tmp_path) not in text

    # explicit runs only: nothing else is picked up
    monkeypatch.setattr(sys, "argv", ["analyse.py", "--results-root", str(root), "--out", str(out), "--run",
                                      "B4=shared_b4_acf_cond_seed42", "--run", "B3=shared_b3_acf_zeroed_seed42"])
    an.main()
    csv = pd.read_csv(out / "comparisons.csv")
    assert row(csv, "B4 vs B3")["mean_diff"] == pytest.approx(0.03 + STEP.mean()) and row(csv, "B2 vs B1")["status"].startswith("not run")

    with pytest.raises(SystemExit, match="unknown model"):
        an.discover(root, ["B9=x"])
    write_run(root, "shared_b5_acf_evidential_seed42", BASE, synthetic=True)
    write_run(root, "shared_b5_acf_evidential_seed43", BASE, synthetic=False)
    with pytest.raises(SystemExit, match="synthetic and real runs are mixed"):
        an.load_tables(root, an.discover(root, []))
    write_run(root, "shared_b3_acf_zeroed_seed43", BASE, extra_case=False)   # a seed that misses a case
    with pytest.raises(SystemExit, match="do not cover the same cases"):
        an.load_tables(root, {"B3": ["shared_b3_acf_zeroed_seed42", "shared_b3_acf_zeroed_seed43"]})
