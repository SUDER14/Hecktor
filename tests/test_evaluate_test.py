"""Locked test script (src/eval/locked_test.py, scripts/evaluate_test.py): gates, lock, metrics, synthetic end to end.
Nothing here reads a real case."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.eval import locked_test as lt
from src.metrics import calibration as cal
from src.metrics.segmentation import aggregated_dice, aggregated_dice_from_counts, hd95, surface_dice

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["patch.spatial_size=[32, 32, 32]", "patch.min_extent_mm=[8.0, 8.0, 8.0]", "infer.sw_batch_size=4"]


# ------------------------------------------------------------------------------------------------------ gates
def _repo(tmp_path: Path) -> Path:
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "t")
    git("config", "user.email", "t@example.invalid")
    (tmp_path / "docs").mkdir()
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("commit", "-q", "-m", "code")
    return tmp_path


def _commit(repo: Path, msg: str = "c") -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=repo, check=True, capture_output=True)


def test_plan_gate_needs_the_plan_committed_and_unmodified(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(SystemExit, match="not committed"):
        lt.check_plan_committed(repo)
    (repo / lt.PLAN).write_text("plan v1\n")
    with pytest.raises(SystemExit, match="not committed"):   # written but not committed
        lt.check_plan_committed(repo)
    _commit(repo)
    info = lt.check_plan_committed(repo)
    assert len(info["plan_sha256"]) == 64 and len(info["commit"]) == 40
    (repo / lt.PLAN).write_text("plan v1, edited after the fact\n")
    with pytest.raises(SystemExit, match="differs from the committed version"):
        lt.check_plan_committed(repo)
    _commit(repo)
    assert lt.check_plan_committed(repo)["plan_sha256"] != info["plan_sha256"]


def test_clean_tree_gate(tmp_path):
    repo = _repo(tmp_path)
    lt.check_clean_tree(repo)
    (repo / "untracked.txt").write_text("ignored on purpose\n")
    lt.check_clean_tree(repo)
    (repo / "code.py").write_text("x = 2\n")
    with pytest.raises(SystemExit, match="tracked files differ"):
        lt.check_clean_tree(repo)
    for gate in (lt.check_clean_tree, lt.check_plan_committed):   # not a git checkout at all
        with pytest.raises(SystemExit):
            gate(Path(tmp_path.anchor))


def test_out_root_inside_the_repo_and_hosted_machine_are_refused(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="inside the repo"):
        lt.check_out_root(ROOT / "results" / "x")
    assert lt.check_out_root(tmp_path) == tmp_path.resolve()
    monkeypatch.setenv("COLAB_RELEASE_TAG", "x")
    monkeypatch.delenv("ALLOW_REAL_DATA", raising=False)
    with pytest.raises(SystemExit, match="ALLOW_REAL_DATA"):
        lt.check_not_hosted()
    monkeypatch.setenv("ALLOW_REAL_DATA", "1")
    lt.check_not_hosted()


def test_real_run_stops_at_the_gates_before_anything_is_loaded(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a checkpoint was loaded before the gates passed")

    monkeypatch.setattr(lt, "load_best", boom)
    monkeypatch.setattr(lt, "check_plan_committed", lambda: (_ for _ in ()).throw(SystemExit("plan gate")))
    with pytest.raises(SystemExit, match="plan gate"):
        lt.evaluate(tmp_path / "run", tmp_path / "cache", tmp_path / "out")
    monkeypatch.setattr(lt, "check_plan_committed", lambda: {"commit": "c", "plan_git_blob": "b", "plan_sha256": "s"})
    monkeypatch.setattr(lt, "check_clean_tree", lambda: (_ for _ in ()).throw(SystemExit("dirty gate")))
    with pytest.raises(SystemExit, match="dirty gate"):
        lt.evaluate(tmp_path / "run", tmp_path / "cache", tmp_path / "out")
    with pytest.raises(SystemExit, match="synthetic only"):   # a real run cannot be given another config
        lt.evaluate(tmp_path / "run", tmp_path / "cache", tmp_path / "out", overrides=["seed=1"])


# ------------------------------------------------------------------------------------------------------- lock
def test_lock_allows_one_run_resume_of_an_interrupted_one_and_logged_reruns_only(tmp_path):
    root, res = tmp_path / "test", tmp_path / "test" / "runA"
    ident = {"run": "runA", "commit": "c1", "plan_sha256": "p1", "checkpoint_sha256": "k1", "synthetic": False}
    with pytest.raises(SystemExit, match="no earlier test run"):
        lt.acquire_lock(res, root, ident, "there is nothing to rerun yet")
    assert lt.acquire_lock(res, root, ident, None) == "fresh"
    (res / "per_case.csv").write_text("first\n")
    assert lt.acquire_lock(res, root, ident, None) == "resume"            # interrupted, same commit/plan/checkpoint
    with pytest.raises(SystemExit, match="evaluated once"):               # interrupted, but the code changed
        lt.acquire_lock(res, root, {**ident, "commit": "c2"}, None)
    lock = json.loads((res / "LOCK.json").read_text())
    (res / "LOCK.json").write_text(json.dumps({**lock, "state": "finished"}))
    with pytest.raises(SystemExit, match="evaluated once"):
        lt.acquire_lock(res, root, ident, None)
    with pytest.raises(SystemExit, match="real reason"):
        lt.acquire_lock(res, root, ident, "oops")
    assert not (root / "RERUN_LOG.txt").exists()
    assert lt.acquire_lock(res, root, {**ident, "commit": "c2"}, "HD95 bug fixed in commit c2") == "fresh"
    log = [json.loads(x) for x in (root / "RERUN_LOG.txt").read_text().splitlines()]
    assert len(log) == 1 and log[0]["reason"] == "HD95 bug fixed in commit c2" and log[0]["old_identity"]["commit"] == "c1"
    moved = Path(log[0]["earlier_outputs_moved_to"])
    assert (moved / "per_case.csv").read_text() == "first\n" and not (res / "per_case.csv").exists()   # kept, not deleted

    # the plan hash of the first test run is pinned for every later run, of any model
    other = {**ident, "run": "runB", "plan_sha256": "p2"}
    with pytest.raises(SystemExit, match="analysis plan changed"):
        lt.acquire_lock(root / "runB", root, other, None)
    assert lt.acquire_lock(root / "runB", root, other, "plan section 11 amended, see commit c3") == "fresh"
    assert json.loads((root / "PLAN_SHA256.json").read_text())["plan_sha256"] == "p2"
    assert len((root / "RERUN_LOG.txt").read_text().splitlines()) == 2


# ---------------------------------------------------------------------------------------------------- metrics
def test_nll_temperature_and_pooled_curves_on_known_answers():
    p = np.array([[[0.5, 0.9]], [[0.5, 0.1]]])            # (K=2, 1, 2)
    y = np.array([[0, 1]])
    assert cal.nll(p, y) == pytest.approx(-(np.log(0.5) + np.log(0.1)) / 2)
    assert cal.nll(p, y, roi=np.array([[True, False]])) == pytest.approx(-np.log(0.5))
    assert np.isfinite(cal.nll(np.array([[[1.0]], [[0.0]]]), np.array([[1]])))   # p_y = 0 is clamped, not infinite

    logits = torch.randn(3, 4, 5, 6, generator=torch.Generator().manual_seed(0))
    prob = torch.softmax(logits, 0).numpy()
    np.testing.assert_allclose(cal.apply_temperature(prob, 1.0), prob, atol=1e-6)
    np.testing.assert_allclose(cal.apply_temperature(prob, 2.5), torch.softmax(logits / 2.5, 0).numpy(), atol=1e-5)

    # labels drawn from softmax(logits); the model reports softmax(3 * logits), i.e. it is over-confident by 3
    g = torch.Generator().manual_seed(1)
    z = torch.randn(3, 40, 40, 40, generator=g)
    y = torch.distributions.Categorical(logits=z.permute(1, 2, 3, 0)).sample().numpy()
    over = torch.softmax(3.0 * z, 0).numpy()
    grid = lt.T_GRID
    half = cal.nll_sum_on_grid(over[:, :20], y[:20], grid) + cal.nll_sum_on_grid(over[:, 20:], y[20:], grid)
    np.testing.assert_allclose(half, cal.nll_sum_on_grid(over, y, grid), rtol=1e-5)    # sums add across cases
    t, edge = cal.fit_temperature(half, grid)
    assert not edge and abs(np.log(t / 3.0)) < 0.1
    assert cal.nll(cal.apply_temperature(over, t), y) < cal.nll(over, y)

    rng = np.random.default_rng(0)
    u = rng.random(20000)
    pred = np.zeros(20000, int)
    label = (rng.random(20000) < u).astype(int)          # wrong more often where u is high
    h = cal.u_histograms(u, pred, label, 1000)
    exact = cal.uncertainty_error_overlap(u, pred, label)["auroc"]
    assert cal.auroc_from_histograms(h["wrong"], h["right"]) == pytest.approx(exact, abs=2e-3)
    rc = cal.risk_coverage_from_histograms(h["wrong"], h["right"])
    assert rc["coverage"][-1] == pytest.approx(1.0) and rc["risk"][-1] == pytest.approx(label.mean())
    assert rc["risk"][len(rc["risk"]) // 4] < rc["risk"][-1] and 0 < rc["aurc"] < label.mean()
    assert np.isnan(cal.auroc_from_histograms(np.zeros(5), np.ones(5)))


def test_seg_row_matches_the_uncropped_metrics_on_an_anisotropic_volume():
    sp = (1.0, 1.0, 3.0)
    label = np.zeros((40, 40, 20), np.uint8)
    pred = np.zeros_like(label)
    label[5:15, 5:15, 2:8] = 1
    pred[7:17, 5:15, 3:9] = 1          # shifted by 2 voxels in x and 1 slice (3 mm) in z
    label[30:40, 30:40, 14:20] = 2     # touches the volume border
    pred[31:40, 30:40, 14:20] = 2
    row = lt.seg_row(pred, label, sp, k=3)
    for c in (1, 2):
        p, t = pred == c, label == c
        assert row[f"hd95_mm_{c}"] == pytest.approx(hd95(p, t, sp).value)           # the crop changes nothing
        assert row[f"surface_dice_2mm_{c}"] == pytest.approx(surface_dice(p, t, sp, 2.0))
        assert row[f"surface_dice_3mm_{c}"] == pytest.approx(surface_dice(p, t, sp, 3.0))
    assert row["dice_1"] == pytest.approx(2 * 8 * 10 * 5 / (2 * 10 * 10 * 6))
    assert row["mean_dice"] == pytest.approx((row["dice_1"] + row["dice_2"]) / 2) and row["n_labels_present"] == 2
    assert np.isnan(row["fp_volume_ml_1"])

    # label 2 absent in the truth: no Dice, the predicted volume is the false-positive volume in mL
    only1 = np.where(label == 1, 1, 0).astype(np.uint8)
    r = lt.seg_row(pred, only1, sp, k=3)
    assert np.isnan(r["dice_2"]) and np.isnan(r["hd95_mm_2"]) and not r["pred_empty_2"]
    assert r["fp_volume_ml_2"] == pytest.approx(9 * 10 * 6 * 3.0 / 1000.0) and r["mean_dice"] == pytest.approx(r["dice_1"])
    # truth present, prediction empty: Dice 0, flagged, and no distance is imputed
    r = lt.seg_row(np.zeros_like(pred), label, sp, k=3)
    assert r["dice_1"] == 0.0 and r["pred_empty_1"] and np.isnan(r["hd95_mm_1"]) and r["surface_dice_2mm_1"] == 0.0
    # neither label in the truth: the case has no per-case mean Dice
    assert np.isnan(lt.seg_row(pred, np.zeros_like(label), sp, k=3)["mean_dice"])

    rows = pd.DataFrame([row, lt.seg_row(pred, only1, sp, k=3)])
    agg = lt.aggregated_summary(rows, k=3, n_boot=200)
    for c in (1, 2):
        direct = aggregated_dice([pred == c, pred == c], [label == c, only1 == c])
        assert agg[f"label_{c}"]["aggregated_dice"] == pytest.approx(direct)
        assert agg[f"label_{c}"]["ci_low"] <= direct <= agg[f"label_{c}"]["ci_high"]
    assert aggregated_dice_from_counts([0], [0], [0]) != aggregated_dice_from_counts([0], [0], [0])   # NaN


def test_derangement_and_subgroup_helpers():
    for n in (2, 3, 10, 123):
        p = lt.derangement(n)
        assert sorted(p) == list(range(n)) and not (p == np.arange(n)).any()
        assert (p == lt.derangement(n)).all()
    assert [lt.z_bin(z) for z in (1.25, 2.0, 2.5, 3.27, 5.0)] == ["z<=2.0", "z<=2.0", "2.0<z<=3.0", "3.0<z<=3.5", "z>3.5"]
    assert lt.is_conditioned({"model": {"name": "acf"}}) and not lt.is_conditioned({"model": {"name": "acf", "conditioning_zeroed": True}})
    assert not lt.is_conditioned({"model": {"name": "dual_unet"}})


# ------------------------------------------------------------------------------------------------- end to end
def test_synthetic_b4_end_to_end_with_interrupt_resume_lock_and_rerun(tmp_path):
    cfg = ROOT / "configs" / "shared_b4_acf_cond.yaml"
    kw = dict(run_dir=None, data_root=None, out_root=tmp_path, synthetic=True, config=cfg, overrides=SMALL, device="cpu",
              n_synthetic=(4, 2))
    with pytest.raises(SystemExit) as e:          # the session dies after 2 of 4 test cases
        lt.evaluate(**kw, stop_after=2)
    assert e.value.code == 3
    res = tmp_path / "results_test_synthetic" / "shared_b4_acf_cond_synthetic"
    assert json.loads((res / "LOCK.json").read_text())["state"] == "started" and not (res / "per_case.csv").exists()
    done = sorted(p.name for p in (res / "cases" / "test").iterdir())
    assert done == ["SYNT-000.npz", "SYNT-001.npz"]
    stamp = (res / "cases" / "test" / "SYNT-000.npz").stat().st_mtime_ns

    assert lt.evaluate(**kw) == res               # continues; finished cases are not recomputed
    assert (res / "cases" / "test" / "SYNT-000.npz").stat().st_mtime_ns == stamp
    assert json.loads((res / "LOCK.json").read_text())["state"] == "finished"
    per_case = pd.read_csv(res / "per_case.csv")
    assert list(per_case["patient_id"]) == [f"SYNT-{i:03d}" for i in range(4)] and set(per_case["centre"]) == {"CHUS", "CHUV"}
    for col in ("dice_1", "dice_2", "hd95_mm_1", "surface_dice_2mm_1", "surface_dice_3mm_2", "fp_volume_ml_2", "mean_dice",
                "inter_1", "pred_vox_2", "label_vox_1", "mean_u_all", "z_bin", "model"):
        assert col in per_case.columns, col
    assert np.isnan(per_case["dice_2"][2]) and np.isfinite(per_case["fp_volume_ml_2"][2])   # case 2 has no label 2

    sweep = pd.read_csv(res / "sweep.csv")        # native z = 1.0, 2.0, 3.27, 5.0 mm
    assert sorted(sweep.loc[sweep.patient_id == "SYNT-000", "factor"]) == [3, 5, 7]
    assert sorted(sweep.loc[sweep.patient_id == "SYNT-001", "target_mm"]) == [3.0, 5.0, 7.0]   # factors 2, 2, 4
    assert sorted(sweep.loc[sweep.patient_id == "SYNT-002", "target_mm"]) == [5.0, 7.0]         # 3 mm: factor 1, left out
    assert sorted(sweep.loc[sweep.patient_id == "SYNT-003", "target_mm"]) == []                 # 5 mm native: all factor 1
    wrong = pd.read_csv(res / "wrong_spacing.csv")
    assert len(wrong) == 4 and (wrong["spacing_from"] != wrong["patient_id"]).all() and wrong["dz_over_1mm"].any()

    calib = json.loads((res / "calibration.json").read_text())
    assert 0.25 <= calib["temperature"]["temperature"] <= 4.0 and "validation" in calib["temperature"]["fitted_on"]
    for subset in ("val", "test"):
        for key in ("raw_all", "raw_roi", "T_all", "T_roi"):
            m = calib[subset][key]
            assert 0 <= m["ece"] <= 1 and 0 <= m["brier"] <= 2 and m["nll"] > 0
            assert (res / f"reliability_{subset}_{key}.csv").exists() and (res / f"reliability_{subset}_{key}.png").exists()
    assert calib["val"]["T_all"]["nll"] <= calib["val"]["raw_all"]["nll"] + 1e-6    # T minimises validation NLL
    unc = json.loads((res / "uncertainty.json").read_text())
    assert 0 <= unc["test"]["roi"]["auroc_pooled"] <= 1 and "mean_u_all_test_minus_val" in unc
    assert (res / "risk_coverage_test_roi.csv").exists() and len(pd.read_csv(res / "val_per_case.csv")) == 2
    summary = json.loads((res / "summary.json").read_text())
    assert summary["test"]["n_cases"] == 4 and "aggregated_dice" in summary["test"]["labels"]["label_1"]
    assert "all" in summary["wrong_spacing"] and len(summary["sweep"]) == 3
    man = json.loads((res / "manifest.json").read_text())
    assert man["synthetic"] and man["resumed"] and len(man["checkpoint_sha256"]) == 64 and man["n_test"] == 4
    preds = tmp_path / "test_predictions" / res.name
    assert (preds / "SYNT-003_pred.nii.gz").exists() and (preds / "SYNT-003_uncertainty.nii.gz").exists()
    assert not list(ROOT.glob("results/test/*synthetic*"))     # nothing was written into the repo

    with pytest.raises(SystemExit, match="evaluated once"):    # one shot
        lt.evaluate(**kw)
    assert lt.evaluate(**kw, rerun="checking the rerun path in a unit test") == res
    log = (tmp_path / "results_test_synthetic" / "RERUN_LOG.txt").read_text().splitlines()
    assert len(log) == 1 and "checking the rerun path" in log[0]
    assert len(list((tmp_path / "results_test_synthetic").glob("shared_b4_acf_cond_synthetic__superseded_*"))) == 1
    again = pd.read_csv(res / "per_case.csv")
    np.testing.assert_allclose(again["mean_dice"], per_case["mean_dice"], equal_nan=True)   # same weights, same numbers


@pytest.mark.parametrize("stem,val_rows,has_wrong", [("shared_b1_unet_resample", False, False),
                                                     ("shared_b3_acf_zeroed", False, False),
                                                     ("shared_b5_acf_evidential", True, True)])
def test_synthetic_other_heads_and_paths(tmp_path, stem, val_rows, has_wrong):
    res = lt.evaluate(None, None, tmp_path, synthetic=True, config=ROOT / "configs" / f"{stem}.yaml", overrides=SMALL,
                      device="cpu", n_synthetic=(2, 1), save_maps=False)
    assert len(pd.read_csv(res / "per_case.csv")) == 2 and (res / "sweep.csv").exists()
    assert (res / "val_per_case.csv").exists() == val_rows and (res / "wrong_spacing.csv").exists() == has_wrong
    calib = json.loads((res / "calibration.json").read_text())
    assert calib["temperature"] is None and "raw_all" in calib["test"] and "T_all" not in calib["test"]
    assert not (tmp_path / "test_predictions").exists()


def test_external_label_maps_go_through_the_same_metrics_and_lock(tmp_path):
    import nibabel as nib

    ds = lt.SyntheticVolumes([f"SYNT-{i:03d}" for i in range(3)], {"patch": {"spatial_size": [24, 24, 24]}}, seed=1)
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    for i in range(3):
        item = ds[i]
        lab = item["native_label"].numpy().copy()
        if i == 1:
            lab[lab == 1] = 0                    # this prediction misses label 1 completely
        nib.save(nib.Nifti1Image(lab, item["native_affine"].numpy()), str(pred_dir / f"{item['patient_id']}.nii.gz"))
    res = lt.evaluate_predictions(pred_dir, "nnunet_synthetic", None, tmp_path / "out", synthetic=True, n_synthetic=3)
    df = pd.read_csv(res / "per_case.csv")
    assert list(df["dice_1"]) == [1.0, 0.0, 1.0] and bool(df["pred_empty_1"][1]) and df["dice_2"][0] == 1.0
    assert df["hd95_mm_1"][0] == 0.0 and np.isnan(df["hd95_mm_1"][1])
    with pytest.raises(SystemExit, match="evaluated once"):
        lt.evaluate_predictions(pred_dir, "nnunet_synthetic", None, tmp_path / "out", synthetic=True, n_synthetic=3)
    (pred_dir / "SYNT-002.nii.gz").unlink()
    with pytest.raises(SystemExit, match="have no prediction"):
        lt.evaluate_predictions(pred_dir, "other", None, tmp_path / "out", synthetic=True, n_synthetic=3)
