"""Locked, one-shot evaluation on the held-out test centres. CLI: scripts/evaluate_test.py.

Implements docs/ANALYSIS_PLAN.md sections 3 (per-case metrics), 5 (subgroup columns), 6 (thickness sweep,
wrong-spacing check) and 7 (calibration, uncertainty) for ONE trained run. Comparisons between runs (section 4)
are a separate step over the per_case.csv files this script writes (src/eval/stats.py, compare_models).

Why it is locked: the test set may inform no decision. So the script

* refuses to start unless docs/ANALYSIS_PLAN.md is committed and unmodified, and the git tree is clean;
* loads only <run>/checkpoints/best.pt, the checkpoint chosen on validation; there is no option for another one;
* writes a lock when it starts, keyed on the run name and on the checkpoint's sha256 (renaming the run folder or
  deleting LOCK.json does not unlock it). An interrupted run continues (same commit, plan and checkpoint) because every
  case is saved as it finishes; a FINISHED run can be repeated only with --rerun "<reason>", which is appended to
  results/test/RERUN_LOG.txt and moves the earlier outputs aside instead of deleting them;
* records the plan's sha256 for the first test run of any model; a later run under a different plan needs --rerun.

Results (small tables) go to <results_root>/<run>/, predictions and uncertainty maps to --out-root, which must be
outside the repo. Nothing here uploads, copies or syncs data anywhere.

--synthetic runs every code path on random volumes and a freshly initialised model: no data, no cache, no git
gates, results under --out-root. Its numbers mean nothing; it exists to prove the script runs end to end.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import ndimage
from scipy import stats as sstats
from torch.amp import autocast

from src.data.conditioning import load_stats, make_cond
from src.eval.evaluate_shared import call_model, coarsen_z
from src.eval.infer import resample_to_grid, save_nifti, sliding_window
from src.eval.stats import N_BOOT, paired_comparison
from src.metrics.calibration import (
    apply_temperature, auroc_from_histograms, brier, fit_temperature, nll, nll_sum_on_grid, reliability_bins,
    risk_coverage_from_histograms, u_histograms, uncertainty_error_overlap,
)
from src.metrics.segmentation import aggregated_dice_from_counts, dice, hd95, surface_dice

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/ANALYSIS_PLAN.md"
SWEEP_MM = (3.0, 5.0, 7.0)                                       # plan section 6
T_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 41))      # temperatures tried on validation
N_BINS, U_BINS, ROI_DILATE = 15, 1000, 5                         # plan section 7
Z_BINS = ((2.0, "z<=2.0"), (3.0, "2.0<z<=3.0"), (3.5, "3.0<z<=3.5"), (np.inf, "z>3.5"))   # plan section 5
WRONG_SPACING_SEED = 20261116
HOSTED_ENV = ("COLAB_RELEASE_TAG", "COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE")
MIN_REASON_CHARS = 10


# ------------------------------------------------------------------------------------------------------ gates
def _git(*args: str, root: Path = ROOT, binary: bool = False):
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout if binary else out.stdout.decode("utf-8", "replace").strip()


def check_plan_committed(root: Path = ROOT) -> dict:
    """The plan must be in HEAD and the file on disk must be that committed version. Compared through git's own
    object hash, so a CRLF checkout of the same commit is not mistaken for an edit."""
    commit = _git("rev-parse", "HEAD", root=root)
    if commit is None:
        raise SystemExit("refusing to run: not a git checkout, so the analysis plan cannot be shown to be committed")
    head_blob = _git("rev-parse", f"HEAD:{PLAN}", root=root)
    if head_blob is None:
        raise SystemExit(f"refusing to run: {PLAN} is not committed in HEAD")
    if not (root / PLAN).exists() or _git("hash-object", PLAN, root=root) != head_blob:
        raise SystemExit(f"refusing to run: {PLAN} on disk differs from the committed version. Commit the change "
                         "(and list it in section 11 of the plan) before any test run.")
    content = _git("cat-file", "blob", head_blob, root=root, binary=True)
    return {"commit": commit, "plan_git_blob": head_blob, "plan_sha256": hashlib.sha256(content).hexdigest()}


def check_clean_tree(root: Path = ROOT) -> None:
    status = _git("status", "--porcelain", "--untracked-files=no", root=root)
    if status is None or status:
        raise SystemExit("refusing to run: tracked files differ from HEAD (or git failed). A test result must trace "
                         "back to one commit.\n" + (status or ""))


def check_not_hosted() -> None:
    hosted = [v for v in HOSTED_ENV if os.environ.get(v)]
    if hosted and os.environ.get("ALLOW_REAL_DATA") != "1":
        raise SystemExit(f"refusing real data on a hosted machine ({hosted[0]} is set) without ALLOW_REAL_DATA=1. "
                         "Set it only after written permission from the data provider.")


def check_out_root(out_root: Path, root: Path = ROOT) -> Path:
    out_root = Path(out_root).resolve()
    if out_root == root.resolve() or root.resolve() in out_root.parents:
        raise SystemExit(f"--out-root {out_root} is inside the repo; predictions and uncertainty maps must stay outside it")
    return out_root


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable), encoding="utf-8")
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


def acquire_lock(res_dir: Path, results_root: Path, identity: dict, rerun_reason: str | None) -> str:
    """Returns "fresh" or "resume". One lock per run name AND per checkpoint.

    nothing there                 -> fresh (and --rerun is refused: there is nothing to rerun)
    lock "started", same identity -> resume an interrupted run; cases already saved are kept
    anything else                 -> only with --rerun "<reason>": logged, earlier outputs moved aside
    "Anything else" includes: a finished or different run under this name; the same checkpoint already
    evaluated under ANOTHER name (a renamed or copied run folder must not buy a second look at the test set);
    a results folder whose LOCK.json is gone (its per-case files must not be silently reused); and a plan that
    changed since the first test run, whose hash is pinned in <results_root>/PLAN_SHA256.json."""
    results_root.mkdir(parents=True, exist_ok=True)
    if rerun_reason is not None and len(rerun_reason.strip()) < MIN_REASON_CHARS:
        raise SystemExit(f'--rerun needs a real reason (at least {MIN_REASON_CHARS} characters), e.g. --rerun "bug in HD95 fixed in commit abc123"')
    lock_path, pin_path = res_dir / "LOCK.json", results_root / "PLAN_SHA256.json"
    pin = json.loads(pin_path.read_text(encoding="utf-8")) if pin_path.exists() else None
    plan_changed = pin is not None and pin["plan_sha256"] != identity["plan_sha256"]
    old = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    orphan = old is None and res_dir.exists() and any(res_dir.iterdir())
    twins = []
    for other in sorted(results_root.glob("*/LOCK.json")):
        if other.parent == res_dir or "__superseded_" in other.parent.name:
            continue
        oid = json.loads(other.read_text(encoding="utf-8"))["identity"]
        if oid.get("checkpoint_sha256") == identity["checkpoint_sha256"] and oid.get("synthetic") == identity["synthetic"]:
            twins.append(other.parent.name)
    same_started = old is not None and old["state"] == "started" and old["identity"] == identity
    if same_started and not plan_changed and rerun_reason is None and (not twins or old.get("rerun_reason")):
        return "resume"
    why = []
    if old is not None:
        why.append(f"a test run of {res_dir.name} already exists (state {old['state']}, commit {old['identity'].get('commit')})")
    if orphan:
        why.append(f"{res_dir} holds results but no LOCK.json")
    if twins:
        why.append(f"this checkpoint has already been evaluated on the test set as {', '.join(twins)}")
    if plan_changed:
        why.append(f"the analysis plan changed since the first test run ({pin['plan_sha256'][:12]} -> {identity['plan_sha256'][:12]})")
    if not why:
        if rerun_reason is not None:
            raise SystemExit(f"--rerun was given but {res_dir.name} has no earlier test run; remove the flag")
    else:
        if rerun_reason is None:
            raise SystemExit("refusing to run: " + "; ".join(why) + '. The test set is evaluated once. If a second '
                             'run is justified, pass --rerun "<reason>"; it is logged in RERUN_LOG.txt.')
        archived = None
        if old is not None or orphan:
            archived = res_dir.with_name(f"{res_dir.name}__superseded_{time.strftime('%Y%m%dT%H%M%S')}")
            os.replace(res_dir, archived)
        with open(results_root / "RERUN_LOG.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps({"date": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": res_dir.name, "reason": rerun_reason.strip(),
                                "because": why, "old_identity": old["identity"] if old else None, "new_identity": identity,
                                "earlier_outputs_moved_to": str(archived) if archived else None}) + "\n")
        if plan_changed:
            pin = None
    res_dir.mkdir(parents=True, exist_ok=True)
    if pin is None:
        _atomic_json(pin_path, {"plan_sha256": identity["plan_sha256"], "first_run": res_dir.name,
                                "date": time.strftime("%Y-%m-%dT%H:%M:%S")})
    _atomic_json(lock_path, {"state": "started", "identity": identity, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "rerun_reason": rerun_reason})
    return "fresh"


# ------------------------------------------------------------------------------------------- metrics per case
def _crop_to_union(a: np.ndarray, b: np.ndarray, pad: int = 1):
    """Both masks cut to the bounding box of their union plus `pad` voxels. Surface distances between the two
    masks are unchanged (every surface voxel of both is inside the box, and a mask that touches the volume
    border still touches the box border there), but the distance transforms run on a far smaller array."""
    idx = np.argwhere(a | b)
    if idx.size == 0:
        return a, b
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, a.shape)
    sl = tuple(slice(int(s), int(e)) for s, e in zip(lo, hi))
    return a[sl], b[sl]


def seg_row(pred: np.ndarray, label: np.ndarray, spacing_mm, k: int) -> dict:
    """Plan section 3 for one case on the native grid. For each label c in 1..k-1:
    Dice when c is in the truth, else the predicted volume in mL (false-positive volume); HD95 in mm only when
    c is in both truth and prediction (truth present + empty prediction is flagged, no distance is imputed);
    surface Dice at 2 and 3 mm when c is in the truth. The three voxel counts feed the aggregated Dice."""
    sp = tuple(float(s) for s in spacing_mm)
    voxel_ml = float(np.prod(sp)) / 1000.0
    row, present_dice = {}, []
    for c in range(1, k):
        p, t = pred == c, label == c
        n_p, n_t = int(p.sum()), int(t.sum())
        row[f"inter_{c}"], row[f"pred_vox_{c}"], row[f"label_vox_{c}"] = int((p & t).sum()), n_p, n_t
        row[f"label_present_{c}"] = n_t > 0
        row[f"dice_{c}"] = dice(p, t)
        row[f"fp_volume_ml_{c}"] = n_p * voxel_ml if n_t == 0 else float("nan")
        row[f"pred_empty_{c}"] = n_t > 0 and n_p == 0
        pc, tc = _crop_to_union(p, t)
        row[f"hd95_mm_{c}"] = hd95(pc, tc, sp).value if (n_t > 0 and n_p > 0) else float("nan")
        row[f"surface_dice_2mm_{c}"] = surface_dice(pc, tc, sp, 2.0) if n_t > 0 else float("nan")
        row[f"surface_dice_3mm_{c}"] = surface_dice(pc, tc, sp, 3.0) if n_t > 0 else float("nan")
        if n_t > 0:
            present_dice.append(row[f"dice_{c}"])
    row["n_labels_present"] = len(present_dice)
    row["mean_dice"] = float(np.mean(present_dice)) if present_dice else float("nan")
    return row


def _calib(p: np.ndarray, label: np.ndarray, roi) -> dict:
    """Additive pieces of ECE, Brier and NLL for one case and one region, so that cases pool by voxel."""
    rb = reliability_bins(p, label, N_BINS, roi)
    n = float(rb["count"].sum())
    return {"count": rb["count"], "sum_conf": np.nan_to_num(rb["confidence"]) * rb["count"],
            "sum_acc": np.nan_to_num(rb["accuracy"]) * rb["count"], "n": np.array(n),
            "brier_sum": np.array(brier(p, label, roi) * n if n else 0.0),
            "nll_sum": np.array(nll(p, label, roi) * n if n else 0.0)}


def calibration_part(p, u, pred, label, temperature: float | None) -> tuple[dict, dict]:
    """(row fields, arrays to pool) for one case. Regions: all voxels, and the plan's region of interest (union of
    truth and prediction foreground dilated by 5 voxels). `temperature` adds the temperature-scaled variant."""
    fg = (pred > 0) | (label > 0)
    roi = ndimage.binary_dilation(fg, iterations=ROI_DILATE) if fg.any() else None
    arrays, variants = {}, {"raw": p}
    if temperature is not None:
        variants["T"] = apply_temperature(p, temperature)
    for vname, pv in variants.items():
        for rname, r in (("all", None), ("roi", roi)):
            if rname == "roi" and roi is None:
                continue
            for key, val in _calib(pv, label, r).items():
                arrays[f"{vname}|{rname}|{key}"] = val
    for rname, r in (("all", None), ("roi", roi)):
        if rname == "roi" and roi is None:
            continue
        h = u_histograms(u, pred, label, U_BINS, r)
        arrays[f"u|{rname}|wrong"], arrays[f"u|{rname}|right"] = h["wrong"], h["right"]
    unc = uncertainty_error_overlap(u, pred, label, roi)
    row = {"mean_u_all": float(u.mean()), "mean_u_roi": float(u[roi].mean()) if roi is not None else float("nan"),
           "unc_auroc_roi": unc["auroc"], "roi_voxels": int(roi.sum()) if roi is not None else 0}
    return row, arrays


# ------------------------------------------------------------------------------------------------- inference
def _infer(model, image: np.ndarray, cond, spacing, cfg: dict, device, use_amp: bool) -> dict:
    patch, k = tuple(cfg["patch"]["spatial_size"]), cfg["model"]["out_channels"]
    inf = cfg.get("infer", {})
    cc = torch.as_tensor(np.asarray(cond), dtype=torch.float32)[None].to(device)
    sp = torch.as_tensor(np.asarray(spacing), dtype=torch.float32)[None].to(device)

    def fwd(x):
        with autocast(device_type=device.type, enabled=use_amp):
            return call_model(model, x, cc.expand(x.shape[0], -1), sp.expand(x.shape[0], -1))

    return sliding_window(fwd, image, patch, k, cfg["model"].get("head", "softmax"), inf.get("overlap", 0.5),
                          inf.get("sw_batch_size", 1), device)


def _to_native(res: dict, aff, lab_aff, shape) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities and uncertainty on the native label grid (linear), class = argmax AFTER that (plan section 2)."""
    p, u = res["p"], res["u"]
    if p.shape[1:] != tuple(shape) or not np.allclose(aff, lab_aff):
        p = resample_to_grid(p, aff, lab_aff, shape, order=1)
        u = resample_to_grid(u, aff, lab_aff, shape, order=1)
    return p, u[0]


def derangement(n: int, seed: int = WRONG_SPACING_SEED) -> np.ndarray:
    """A seeded permutation in which no index keeps its place (Sattolo's algorithm: one single cycle)."""
    perm, rng = np.arange(n), np.random.default_rng(seed)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def is_conditioned(cfg: dict) -> bool:
    return cfg["model"]["name"] == "acf" and not cfg["model"].get("conditioning_zeroed", False)


def _save_case(path: Path, rows: dict, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, __rows__=np.array(json.dumps(rows, default=_jsonable)), **arrays)
    os.replace(tmp, path)


def _load_case(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["__rows__"])), {k: z[k] for k in z.files if k != "__rows__"}


class _Ctx:
    """What every case needs: model, config, datasets, where to write."""

    def __init__(self, model, cfg, stats, device, use_amp, res_dir: Path, pred_dir: Path | None):
        self.model, self.cfg, self.stats, self.device, self.use_amp = model, cfg, stats, device, use_amp
        self.res_dir, self.pred_dir = res_dir, pred_dir
        self.k = cfg["model"]["out_channels"]
        self.native_mode = cfg.get("data", {}).get("mode", "native") == "native"


def run_fit_case(ctx: _Ctx, ds, i: int) -> None:
    """Validation pass 1 (softmax B4 only): the NLL of each temperature on the grid, nothing else."""
    out = ctx.res_dir / "cases" / "val_fit" / f"{ds.samples[i]['patient_id']}.npz"
    if out.exists():
        return
    item = ds[i]
    p, _ = _to_native(_infer(ctx.model, item["image"].numpy(), item["cond"], item["spacing"], ctx.cfg, ctx.device, ctx.use_amp),
                      item["affine"].numpy(), item["native_affine"].numpy(), item["native_label"].shape)
    _save_case(out, {}, {"nll_grid": nll_sum_on_grid(p, item["native_label"].numpy(), T_GRID)})


def run_main_case(ctx: _Ctx, ds, i: int, subset: str, temperature: float | None, save_maps: bool) -> None:
    """One case, predicted as trained: section 3 metrics plus the calibration pieces."""
    s = ds.samples[i]
    out = ctx.res_dir / "cases" / subset / f"{s['patient_id']}.npz"
    if out.exists():
        return
    item = ds[i]
    label, lab_aff = item["native_label"].numpy(), item["native_affine"].numpy()
    t0 = time.perf_counter()
    res = _infer(ctx.model, item["image"].numpy(), item["cond"], item["spacing"], ctx.cfg, ctx.device, ctx.use_amp)
    p, u = _to_native(res, item["affine"].numpy(), lab_aff, label.shape)
    dt = time.perf_counter() - t0
    pred = p.argmax(0).astype(np.uint8)
    sp = [float(abs(x)) for x in np.diag(lab_aff)[:3]]
    crow, arrays = calibration_part(p, u, pred, label, temperature)
    row = {"patient_id": s["patient_id"], "centre": s["centre"], "subset": subset, "spacing_x": sp[0], "spacing_y": sp[1],
           "spacing_z": sp[2], **seg_row(pred, label, sp, ctx.k), **crow, "infer_s": round(dt, 2)}
    if save_maps and ctx.pred_dir is not None and subset == "test":
        save_nifti(pred, lab_aff, ctx.pred_dir / f"{s['patient_id']}_pred.nii.gz", np.uint8)
        save_nifti(u, lab_aff, ctx.pred_dir / f"{s['patient_id']}_uncertainty.nii.gz", np.float32)
    _save_case(out, {"main": row}, arrays)


def run_extra_case(ctx: _Ctx, ds, native_ds, i: int, swap_from: int | None) -> None:
    """Thickness sweep and wrong-spacing check for one TEST case (plan section 6). Scored on the native label."""
    s = ds.samples[i]
    out = ctx.res_dir / "cases" / "test_extra" / f"{s['patient_id']}.npz"
    if out.exists():
        return
    item = ds[i]
    nat = item if native_ds is ds else native_ds[i]
    label, lab_aff = item["native_label"].numpy(), item["native_affine"].numpy()
    sp_native = [float(abs(x)) for x in np.diag(lab_aff)[:3]]
    base = {"patient_id": s["patient_id"], "centre": s["centre"], "spacing_z": sp_native[2]}
    sweep, wrong = [], []
    nat_img, nat_aff = nat["image"].numpy(), nat["affine"].numpy()
    for target in SWEEP_MM:
        f = max(1, int(round(target / sp_native[2])))
        if f == 1:
            continue   # the case is already about this thick; it is left out of this level (plan section 6)
        thick, thick_aff = coarsen_z(nat_img, nat_aff, target)
        if ctx.native_mode:   # the model sees the thick grid; conditioned models are told its spacing
            image, aff = thick, thick_aff
            spacing = np.abs(np.diag(thick_aff)[:3]).astype(np.float32)
            cond = make_cond(spacing, ctx.stats)
        else:                 # resampled models: the thick volume is interpolated onto their fixed grid
            aff = item["affine"].numpy()
            image = resample_to_grid(thick, thick_aff, aff, item["image"].shape[1:], order=1)
            spacing, cond = item["spacing"].numpy(), item["cond"].numpy()
        p, _ = _to_native(_infer(ctx.model, image, cond, spacing, ctx.cfg, ctx.device, ctx.use_amp), aff, lab_aff, label.shape)
        sweep.append({**base, "target_mm": target, "factor": f, "achieved_mm": f * sp_native[2],
                      **seg_row(p.argmax(0).astype(np.uint8), label, sp_native, ctx.k)})
    if swap_from is not None:   # the image is untouched; only what the model is TOLD about the spacing is wrong
        o = ds.samples[swap_from]
        p, _ = _to_native(_infer(ctx.model, item["image"].numpy(), np.asarray(o["cond"]), np.asarray(o["spacing"], np.float32),
                                 ctx.cfg, ctx.device, ctx.use_amp), item["affine"].numpy(), lab_aff, label.shape)
        dz = abs(float(o["spacing"][2]) - sp_native[2])
        wrong.append({**base, "spacing_from": o["patient_id"], "told_spacing_x": float(o["spacing"][0]),
                      "told_spacing_z": float(o["spacing"][2]), "abs_dz_mm": dz, "dz_over_1mm": dz > 1.0,
                      **seg_row(p.argmax(0).astype(np.uint8), label, sp_native, ctx.k)})
    _save_case(out, {"sweep": sweep, "wrong": wrong}, {})


# -------------------------------------------------------------------------------------------------- summaries
def z_bin(z: float) -> str:
    return next(name for hi, name in Z_BINS if z <= hi)


def spacing_key(spacing, nd: int = 3) -> tuple[float, float]:
    """(in-plane, z) rounded to 3 decimals; the 77 / 46 split of the plan is the same at 2, 3 and 4 decimals."""
    return round(float(spacing[0]), nd), round(float(spacing[2]), nd)


def aggregated_summary(df: pd.DataFrame, k: int, seed: int = 0, n_boot: int = N_BOOT) -> dict:
    """Aggregated Dice per label over the cases of `df`, with a percentile bootstrap interval over cases."""
    out, rng = {}, np.random.default_rng(seed)
    n = len(df)
    idx = rng.integers(0, n, (n_boot, n)) if n else None
    for c in range(1, k):
        i, p, t = (df[f"{col}_{c}"].to_numpy(np.float64) for col in ("inter", "pred_vox", "label_vox"))
        entry = {"aggregated_dice": aggregated_dice_from_counts(i, p, t), "n_cases": n,
                 "n_label_present": int(df[f"label_present_{c}"].sum()), "n_pred_empty": int(df[f"pred_empty_{c}"].sum()),
                 "mean_dice_where_present": _nanmean(df[f"dice_{c}"]), "mean_hd95_mm": _nanmean(df[f"hd95_mm_{c}"]),
                 "n_hd95": int(df[f"hd95_mm_{c}"].notna().sum()),
                 "mean_surface_dice_2mm": _nanmean(df[f"surface_dice_2mm_{c}"]),
                 "mean_surface_dice_3mm": _nanmean(df[f"surface_dice_3mm_{c}"]),
                 "mean_fp_volume_ml_where_absent": _nanmean(df[f"fp_volume_ml_{c}"])}
        if n:
            den = p[idx].sum(1) + t[idx].sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = 2.0 * i[idx].sum(1) / den
            boot = boot[np.isfinite(boot)]
            if len(boot):
                entry["ci_low"], entry["ci_high"] = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        out[f"label_{c}"] = entry
    return out


def _nanmean(x) -> float:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def _describe(df: pd.DataFrame, by: str) -> list[dict]:
    rows = []
    for key, g in df.groupby(by, sort=True):
        v = g["mean_dice"].dropna()
        rows.append({by: key if not isinstance(key, np.generic) else key.item(), "n": len(g), "n_with_mean_dice": len(v),
                     "mean_dice_mean": float(v.mean()) if len(v) else float("nan"),
                     "mean_dice_median": float(v.median()) if len(v) else float("nan")})
    return rows


def pooled_calibration(parts: list[dict]) -> dict:
    """ECE (15 bins), Brier and NLL pooled by voxel over cases, per variant (raw, T) and region (all, roi)."""
    out, bins = {}, {}
    for vname in ("raw", "T"):
        for rname in ("all", "roi"):
            acc = {}
            for a in parts:
                if f"{vname}|{rname}|n" not in a:
                    continue
                for key in ("count", "sum_conf", "sum_acc", "n", "brier_sum", "nll_sum"):
                    acc[key] = acc.get(key, 0) + a[f"{vname}|{rname}|{key}"]
            if not acc or float(acc["n"]) == 0:
                continue
            n, ok = float(acc["n"]), acc["count"] > 0
            conf, accu = acc["sum_conf"][ok] / acc["count"][ok], acc["sum_acc"][ok] / acc["count"][ok]
            out[f"{vname}_{rname}"] = {"ece": float(np.sum(acc["count"][ok] / n * np.abs(accu - conf))),
                                       "brier": float(acc["brier_sum"]) / n, "nll": float(acc["nll_sum"]) / n, "n_voxels": n}
            edges = np.linspace(0.0, 1.0, N_BINS + 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                bins[f"{vname}_{rname}"] = pd.DataFrame({
                    "bin_low": edges[:-1], "bin_high": edges[1:], "count": acc["count"],
                    "confidence": acc["sum_conf"] / acc["count"], "accuracy": acc["sum_acc"] / acc["count"]})
    return {"metrics": out, "bins": bins}


def pooled_uncertainty(parts: list[dict]) -> dict:
    out = {}
    for rname in ("all", "roi"):
        hs = [(a[f"u|{rname}|wrong"], a[f"u|{rname}|right"]) for a in parts if f"u|{rname}|wrong" in a]
        if not hs:
            continue
        w, r = np.sum([h[0] for h in hs], axis=0), np.sum([h[1] for h in hs], axis=0)
        rc = risk_coverage_from_histograms(w, r)
        out[rname] = {"auroc_pooled": auroc_from_histograms(w, r), "aurc": rc["aurc"], "n_wrong": float(w.sum()),
                      "n_right": float(r.sum()), "curve": pd.DataFrame({"coverage": rc["coverage"], "risk": rc["risk"]})}
    return out


def _reliability_png(bins: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = bins["count"] > 0
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(bins["confidence"][ok], bins["accuracy"][ok], "o-", ms=3)
    ax.set(xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1), title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_outputs(res_dir: Path, name: str, k: int, test_ids: list[str], val_ids: list[str], train_keys: set | None,
                  temperature_info: dict | None) -> dict:
    """Everything in res_dir other than the lock and the manifest, rebuilt from the per-case files."""
    def load(subset, ids):
        rows, parts = [], []
        for pid in ids:
            f = res_dir / "cases" / subset / f"{pid}.npz"
            if f.exists():
                r, a = _load_case(f)
                rows.append(r)
                parts.append(a)
        return rows, parts

    summary: dict = {"run": name, "note": "descriptive summaries; tests between models are a separate step (plan section 4)"}
    t_rows, t_parts = load("test", test_ids)
    test = pd.DataFrame([r["main"] for r in t_rows])
    test.insert(0, "model", name)
    test["z_bin"] = [z_bin(z) for z in test["spacing_z"]]
    if train_keys is not None:
        test["unseen_spacing"] = [spacing_key((x, x, z)) not in train_keys for x, z in zip(test["spacing_x"], test["spacing_z"])]
    test.to_csv(res_dir / "per_case.csv", index=False)
    summary["test"] = {"n_cases": len(test), "n_without_any_label": int((test["n_labels_present"] == 0).sum()),
                       "mean_dice_mean": _nanmean(test["mean_dice"]), "labels": aggregated_summary(test, k),
                       "by_centre": _describe(test, "centre"), "by_z_bin": _describe(test, "z_bin")}
    if "unseen_spacing" in test:
        summary["test"]["by_unseen_spacing"] = _describe(test, "unseen_spacing")
    summary["subgroup_caveat"] = ("subgroups overlap heavily (all CHUS cases are in the unseen-spacing group); a difference "
                                  "cannot be attributed to spacing rather than centre")

    x_rows, _ = load("test_extra", test_ids)
    sweep = pd.DataFrame([r for x in x_rows for r in x["sweep"]])
    wrong = pd.DataFrame([r for x in x_rows for r in x["wrong"]])
    if len(sweep):
        sweep.insert(0, "model", name)
        sweep.to_csv(res_dir / "sweep.csv", index=False)
        summary["sweep"] = [{"target_mm": float(t), "n_cases": len(g), "mean_achieved_mm": float(g["achieved_mm"].mean()),
                             "mean_dice_mean": _nanmean(g["mean_dice"]),
                             **{f"mean_hd95_mm_{c}": _nanmean(g[f"hd95_mm_{c}"]) for c in range(1, k)}}
                            for t, g in sweep.groupby("target_mm")]
    if len(wrong):
        wrong.insert(0, "model", name)
        wrong.to_csv(res_dir / "wrong_spacing.csv", index=False)
        m = test.set_index("patient_id")["mean_dice"].reindex(wrong["patient_id"]).to_numpy()
        w = wrong["mean_dice"].to_numpy()
        far = wrong["dz_over_1mm"].to_numpy(bool)
        summary["wrong_spacing"] = {"definition": "per-case mean Dice, correct spacing minus shuffled spacing",
                                    "all": paired_comparison(m, w), "dz_over_1mm": paired_comparison(m[far], w[far]),
                                    "dz_up_to_1mm": paired_comparison(m[~far], w[~far])}

    v_rows, v_parts = load("val", val_ids)
    val = pd.DataFrame([r["main"] for r in v_rows])
    if len(val):
        val.insert(0, "model", name)
        val.to_csv(res_dir / "val_per_case.csv", index=False)
    calibration = {"temperature": temperature_info, "bins": N_BINS, "roi": f"union of truth and prediction foreground dilated by {ROI_DILATE} voxels"}
    uncertainty = {"u_bins": U_BINS, "mean_u_note": "plan section 7: the region-of-interest mean (mean_u_roi) is the one used; the all-voxel mean is given next to it"}
    for subset, parts, df in (("val", v_parts, val), ("test", t_parts, test)):
        if not parts:
            continue
        pc = pooled_calibration(parts)
        calibration[subset] = pc["metrics"]
        for key, b in pc["bins"].items():
            b.to_csv(res_dir / f"reliability_{subset}_{key}.csv", index=False)
            _reliability_png(b, res_dir / f"reliability_{subset}_{key}.png", f"{name} {subset} {key}")
        pu = pooled_uncertainty(parts)
        for rname, d in pu.items():
            d.pop("curve").to_csv(res_dir / f"risk_coverage_{subset}_{rname}.csv", index=False)
        uncertainty[subset] = {**pu, "mean_u_all_case_mean": _nanmean(df["mean_u_all"]), "mean_u_roi_case_mean": _nanmean(df["mean_u_roi"]),
                               "unc_auroc_roi_case_mean": _nanmean(df["unc_auroc_roi"])}
    if len(val) and len(test):
        rng = np.random.default_rng(0)
        for col in ("mean_u_all", "mean_u_roi"):
            a, b = test[col].dropna().to_numpy(), val[col].dropna().to_numpy()
            if len(a) and len(b):
                d = rng.choice(a, (N_BOOT, len(a))).mean(1) - rng.choice(b, (N_BOOT, len(b))).mean(1)
                uncertainty[f"{col}_test_minus_val"] = {"difference": float(a.mean() - b.mean()), "ci_low": float(np.percentile(d, 2.5)),
                                                        "ci_high": float(np.percentile(d, 97.5)), "n_test": len(a), "n_val": len(b)}
    for col in ("mean_u_all", "mean_u_roi"):
        ok = test[[col, "mean_dice"]].dropna()
        if len(ok) >= 3 and ok[col].nunique() > 1 and ok["mean_dice"].nunique() > 1:
            r = sstats.spearmanr(ok[col], 1.0 - ok["mean_dice"])
            uncertainty[f"spearman_{col}_vs_dice_loss"] = {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(ok)}
    _atomic_json(res_dir / "calibration.json", calibration)
    _atomic_json(res_dir / "uncertainty.json", uncertainty)
    _atomic_json(res_dir / "summary.json", summary)
    return summary


# ------------------------------------------------------------------------------------------------ data + model
class SyntheticVolumes(torch.utils.data.Dataset):
    """Random volumes with the keys of AcfVolumeDataset. For proving the script runs; never for a result."""

    def __init__(self, ids: list[str], cfg: dict, seed: int):
        self.shape = tuple(int(p) + 8 for p in cfg["patch"]["spatial_size"])
        self.seed = seed
        zs = (1.0, 2.0, 3.27, 5.0)
        self.samples = []
        for i, pid in enumerate(ids):
            sp = [1.0, 1.0, zs[i % len(zs)]]
            self.samples.append({"patient_id": pid, "centre": "CHUS" if i % 2 == 0 else "CHUV", "spacing": sp,
                                 "cond": make_cond(sp, SYNTHETIC_STATS)})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s, rng = self.samples[i], np.random.default_rng([self.seed, i])
        img = rng.random((2, *self.shape), dtype=np.float32)
        lab = np.zeros(self.shape, np.uint8)
        for c in (1, 2):
            if c == 2 and i % 3 == 2:
                continue   # some cases have no label 2, to exercise the false-positive-volume branch
            st = [int(rng.integers(0, n - 8)) for n in self.shape]
            lab[tuple(slice(a, a + 8) for a in st)] = c
        aff = np.diag([*s["spacing"], 1.0])
        return {"patient_id": s["patient_id"], "centre": s["centre"], "image": torch.from_numpy(img),
                "label": torch.from_numpy(lab.astype(np.int64))[None], "cond": torch.as_tensor(s["cond"]),
                "spacing": torch.tensor(s["spacing"], dtype=torch.float32), "affine": torch.tensor(aff, dtype=torch.float64),
                "native_affine": torch.tensor(aff, dtype=torch.float64), "native_label": torch.from_numpy(lab)}


SYNTHETIC_STATS = {"CT": {"spacing_x": {"mean": 1.0, "std": 0.2}, "spacing_y": {"mean": 1.0, "std": 0.2},
                          "slice_thickness": {"mean": 3.0, "std": 1.0}}}


def localise_config(cfg: dict, data_root: Path | None) -> dict:
    """A copy of the checkpoint's config with every machine path replaced. scripts/train.py stores the config in
    best.pt with the TRAINING machine's absolute paths; none of them may steer, or be recorded by, a test run
    made elsewhere. Split and statistics become the repo-relative files of this checkout with the recorded
    file name (the sha256 check against the run manifest proves they are the same files); the cache folder is
    the --data-root given now; the zip path and the old run folder are not used here and are blanked."""
    out = copy.deepcopy(cfg)
    for sec, key in (("split", "path"), ("conditioning", "stats_path")):
        rel = f"configs/splits/{PurePosixPath(str(cfg[sec][key]).replace(chr(92), '/')).name}"
        if not (ROOT / rel).exists():
            raise SystemExit(f"{cfg[sec][key]} (recorded in the checkpoint) is not in this checkout as {rel}")
        out[sec][key] = rel
    out.setdefault("cache", {})["dir"] = str(data_root) if data_root is not None else None
    if "data_dir" in out:
        out["data_dir"] = None
    if isinstance(out.get("run"), dict) and "dir" in out["run"]:
        out["run"]["dir"] = None
    return out


def load_best(run_dir: Path, device) -> tuple[torch.nn.Module, dict, dict]:
    """Only <run>/checkpoints/best.pt. There is deliberately no argument for any other checkpoint."""
    from src.training.run import build_model   # imported here: it pulls in the whole training stack

    best = Path(run_dir) / "checkpoints" / "best.pt"
    if not best.exists():
        raise SystemExit(f"{best} does not exist. The test script loads only the best validation checkpoint of a run.")
    ck = torch.load(best, map_location="cpu", weights_only=False)
    if "model_state" not in ck or "config" not in ck:
        raise SystemExit(f"{best} is not a checkpoint written by scripts/train.py (no model_state / config)")
    cfg = ck["config"]
    model = build_model(cfg)
    model.load_state_dict(ck["model_state"])
    model.eval().to(device)
    return model, cfg, {"checkpoint": "checkpoints/best.pt", "checkpoint_sha256": sha256_file(best), "step": ck.get("step"),
                        "best_validation_score": ck.get("best_score")}


def make_synthetic_run(config_path: Path, overrides: list[str], out_root: Path) -> Path:
    """A run folder with a freshly initialised best.pt, so --synthetic goes through the same loading path."""
    from src.training.run import build_model
    from src.training.train import apply_overrides

    cfg = apply_overrides(yaml.safe_load(Path(config_path).read_text(encoding="utf-8")), list(overrides))
    run_dir = out_root / "runs" / f"{Path(config_path).stem}_synthetic"
    best = run_dir / "checkpoints" / "best.pt"
    if not best.exists():
        best.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(int(cfg["seed"]))
        torch.save({"model_state": build_model(cfg).state_dict(), "config": cfg, "step": 0, "best_score": None}, best)
    return run_dir


def _check_run_manifest(run_dir: Path, split_path: Path, stats_path: Path) -> dict:
    mf = run_dir / "manifest.json"
    if not mf.exists():
        raise SystemExit(f"{mf} is missing; the run must come from scripts/train.py, which writes it")
    m = json.loads(mf.read_text(encoding="utf-8"))
    if not m.get("finished"):
        raise SystemExit(f"run {run_dir.name} is not finished (manifest finished=false); best.pt could still change")
    if m.get("synthetic"):
        raise SystemExit(f"run {run_dir.name} was trained with --synthetic; it cannot be scored on real test data")
    for key, path in (("split_sha256", split_path), ("conditioning_stats_sha256", stats_path)):
        if m.get(key) != sha256_file(path):
            raise SystemExit(f"{path.name} in this checkout is not the file the run was trained with ({key} differs)")
    return m


# ------------------------------------------------------------------------------------------------------- main
def evaluate(run_dir: Path | None, data_root: Path | None, out_root: Path, synthetic: bool = False, config: Path | None = None,
             overrides=(), rerun: str | None = None, device: str | None = None, save_maps: bool = True,
             results_root: Path | None = None, n_synthetic: tuple[int, int] = (5, 3), stop_after: int | None = None) -> Path:
    """Returns the results directory. `stop_after` exits hard after that many test cases; it exists only so the
    tests can simulate a dead session."""
    out_root = check_out_root(out_root)
    if synthetic:
        if config is None and run_dir is None:
            raise SystemExit("--synthetic needs --config (a model config; weights are freshly initialised) or --run-dir")
        git = {"commit": _git("rev-parse", "HEAD"), "plan_git_blob": None,
               "plan_sha256": sha256_file(ROOT / PLAN) if (ROOT / PLAN).exists() else None}
        run_dir = Path(run_dir) if run_dir is not None else make_synthetic_run(config, list(overrides), out_root)
        results_root = Path(results_root) if results_root is not None else out_root / "results_test_synthetic"
    else:
        if overrides or config is not None:
            raise SystemExit("--config and --set are for --synthetic only; a real test run uses the config stored in best.pt")
        if run_dir is None or data_root is None:
            raise SystemExit("--run-dir and --data-root are required unless --synthetic")
        check_not_hosted()
        git = check_plan_committed()
        check_clean_tree()
        run_dir, data_root = Path(run_dir), Path(data_root)
        results_root = Path(results_root) if results_root is not None else ROOT / "results" / "test"
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, ck = load_best(run_dir, dev)
    cfg = localise_config(cfg, data_root)   # no path of the training machine is used or written from here on
    name = run_dir.name
    k = cfg["model"]["out_channels"]
    use_amp = bool(cfg["train"].get("amp", True)) and dev.type == "cuda"

    train_keys, run_manifest = None, None
    if synthetic:
        stats = SYNTHETIC_STATS
        test_ds = SyntheticVolumes([f"SYNT-{i:03d}" for i in range(n_synthetic[0])], cfg, seed=1)
        val_ds = SyntheticVolumes([f"SYNV-{i:03d}" for i in range(n_synthetic[1])], cfg, seed=2)
        native_test = test_ds
        hashes = {"split_sha256": None, "conditioning_stats_sha256": None}
    else:
        from src.data import crop_cache
        from src.data.acf_dataset import AcfVolumeDataset
        from src.data.splits import load_split

        split_path, stats_path = ROOT / cfg["split"]["path"], ROOT / cfg["conditioning"]["stats_path"]
        run_manifest = _check_run_manifest(run_dir, split_path, stats_path)
        split, stats = load_split(split_path), load_stats(stats_path)
        leak = set(split["test"]) & (set(split["train"]) | set(split["val"]))
        if leak:
            raise SystemExit(f"split is corrupt: {len(leak)} test ids are also in train/val")
        heads = {i: crop_cache.read_header(data_root / f"{i}.acf") for s in ("train", "val", "test") for i in split[s]}
        train_keys = {spacing_key(heads[i]["spacing"]) for i in split["train"]}   # headers only, for the subgroup column

        def samples(ids):
            return [{"patient_id": i, "centre": heads[i]["centre"]} for i in ids]

        test_ds = AcfVolumeDataset(samples(split["test"]), data_root, cfg, stats)
        val_ds = AcfVolumeDataset(samples(split["val"]), data_root, cfg, stats)
        native_test = test_ds
        if cfg.get("data", {}).get("mode", "native") != "native":   # the sweep starts from the native volume
            ncfg = copy.deepcopy(cfg)
            ncfg["data"]["mode"] = "native"
            native_test = AcfVolumeDataset(samples(split["test"]), data_root, ncfg, stats)
        hashes = {"split_sha256": sha256_file(split_path), "conditioning_stats_sha256": sha256_file(stats_path)}

    identity = {"run": name, "commit": git["commit"], "plan_sha256": git["plan_sha256"],
                "checkpoint_sha256": ck["checkpoint_sha256"], "synthetic": synthetic}
    res_dir = results_root / name
    mode = acquire_lock(res_dir, results_root, identity, rerun)
    pred_dir = (out_root / "test_predictions" / name) if save_maps else None
    ctx = _Ctx(model, cfg, stats, dev, use_amp, res_dir, pred_dir)
    test_ids, val_ids = [s["patient_id"] for s in test_ds.samples], [s["patient_id"] for s in val_ds.samples]
    conditioned, evidential = is_conditioned(cfg), cfg["model"].get("head", "softmax") == "evidential"
    fit_t = conditioned and not evidential          # plan section 7: temperature scaling for B4 only
    val_pass = conditioned                          # validation calibration for B4 and B5
    print(f"{name}: {mode} run, device {dev}, {len(test_ids)} test cases, {len(val_ids)} validation cases, "
          f"validation pass {val_pass}, temperature fit {fit_t}, wrong-spacing check {conditioned}", flush=True)
    t_start = time.time()

    temperature_info, temperature = None, None
    if fit_t:   # validation only; the test set plays no part in choosing T
        for i in range(len(val_ds)):
            run_fit_case(ctx, val_ds, i)
        sums = np.sum([_load_case(res_dir / "cases" / "val_fit" / f"{pid}.npz")[1]["nll_grid"] for pid in val_ids], axis=0)
        temperature, at_edge = fit_temperature(sums, T_GRID)
        temperature_info = {"temperature": temperature, "at_grid_edge": at_edge, "fitted_on": f"{len(val_ids)} validation cases, all voxels",
                            "grid": [float(T_GRID[0]), float(T_GRID[-1]), len(T_GRID)],
                            "note": "applied to the log of the sliding-window-averaged probability (src/metrics/calibration.py)"}
        print(f"temperature {temperature:.4f} (grid edge: {at_edge})", flush=True)
    if val_pass:
        for i in range(len(val_ds)):
            run_main_case(ctx, val_ds, i, "val", temperature, save_maps=False)
    perm = derangement(len(test_ds)) if conditioned and len(test_ds) > 1 else None
    for i in range(len(test_ds)):
        run_main_case(ctx, test_ds, i, "test", temperature, save_maps)
        run_extra_case(ctx, test_ds, native_test, i, int(perm[i]) if perm is not None else None)
        if stop_after is not None and i + 1 >= stop_after:
            raise SystemExit(3)
        if (i + 1) % 10 == 0:
            print(f"test case {i + 1}/{len(test_ds)}  {time.time() - t_start:.0f} s", flush=True)

    write_outputs(res_dir, name, k, test_ids, val_ids, train_keys, temperature_info)
    _atomic_json(res_dir / "manifest.json", {
        **identity, **hashes, **ck, "plan": PLAN, "plan_git_blob": git["plan_git_blob"], "config": cfg,
        "run_manifest": {key: run_manifest.get(key) for key in ("git_commit", "config_sha256", "best_step", "max_steps", "seed")} if run_manifest else None,
        "n_test": len(test_ids), "n_val": len(val_ids), "test_ids": test_ids, "sweep_mm": list(SWEEP_MM),
        "wrong_spacing_seed": WRONG_SPACING_SEED if perm is not None else None, "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "amp": use_amp,
        "predictions_dir": f"test_predictions/{name}" if pred_dir else None, "rerun_reason": rerun, "resumed": mode == "resume",
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds_this_session": round(time.time() - t_start, 1),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "platform": platform.platform()}})
    lock = json.loads((res_dir / "LOCK.json").read_text(encoding="utf-8"))
    _atomic_json(res_dir / "LOCK.json", {**lock, "state": "finished", "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return res_dir


# ---------------------------------------------------------------------------------------- external predictions
def evaluate_predictions(pred_dir: Path, name: str, data_root: Path | None, out_root: Path, synthetic: bool = False,
                         rerun: str | None = None, results_root: Path | None = None, split_path: Path | None = None,
                         n_synthetic: int = 5, k: int = 3) -> Path:
    """nnU-Net (or any external) label maps through the SAME metric code and the same lock. One file
    <pred_dir>/<case>.nii.gz per test case, labels 0..k-1, on the cropped cache grid (the grid nnU-Net was given).
    No probabilities, so no calibration; no sweep and no wrong-spacing check."""
    import nibabel as nib

    out_root = check_out_root(out_root)
    pred_dir = Path(pred_dir)
    train_keys = None
    if synthetic:
        git = {"commit": _git("rev-parse", "HEAD"), "plan_git_blob": None, "plan_sha256": None}
        ds = SyntheticVolumes([f"SYNT-{i:03d}" for i in range(n_synthetic)], {"patch": {"spatial_size": [24, 24, 24]}}, seed=1)
        cases = [(s["patient_id"], s["centre"]) for s in ds.samples]

        def truth(i):
            item = ds[i]
            return item["native_label"].numpy(), item["native_affine"].numpy()

        results_root = Path(results_root) if results_root is not None else out_root / "results_test_synthetic"
    else:
        if data_root is None:
            raise SystemExit("--data-root is required unless --synthetic")
        check_not_hosted()
        git = check_plan_committed()
        check_clean_tree()
        from src.data import crop_cache
        from src.data.splits import load_split

        data_root = Path(data_root)
        split = load_split(split_path or ROOT / "configs" / "splits" / "hecktor2026_seed42.json")
        heads = {i: crop_cache.read_header(data_root / f"{i}.acf") for s in ("train", "test") for i in split[s]}
        train_keys = {spacing_key(heads[i]["spacing"]) for i in split["train"]}
        cases = [(i, heads[i]["centre"]) for i in split["test"]]

        def truth(i):
            h, arr = crop_cache.open_case(data_root / f"{cases[i][0]}.acf")
            return np.asarray(arr["label"]).astype(np.uint8), np.asarray(h["affine"], float)

        results_root = Path(results_root) if results_root is not None else ROOT / "results" / "test"
    missing = [pid for pid, _ in cases if not (pred_dir / f"{pid}.nii.gz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} test cases have no prediction in {pred_dir}, e.g. {missing[:3]}")
    digest = hashlib.sha256("".join(f"{pid}:{sha256_file(pred_dir / f'{pid}.nii.gz')}\n" for pid, _ in cases).encode()).hexdigest()
    identity = {"run": name, "commit": git["commit"], "plan_sha256": git["plan_sha256"], "checkpoint_sha256": digest, "synthetic": synthetic}
    res_dir = results_root / name
    mode = acquire_lock(res_dir, results_root, identity, rerun)
    for i, (pid, centre) in enumerate(cases):
        out = res_dir / "cases" / "test" / f"{pid}.npz"
        if out.exists():
            continue
        label, aff = truth(i)
        img = nib.as_closest_canonical(nib.load(pred_dir / f"{pid}.nii.gz"))
        pred = np.asarray(img.dataobj).astype(np.uint8)
        if pred.shape != label.shape or not np.allclose(img.affine, aff, atol=1e-3):
            raise SystemExit(f"{pid}: prediction grid {pred.shape} does not match the cache grid {label.shape} (shape or affine)")
        if pred.max() >= k:
            raise SystemExit(f"{pid}: prediction holds label {int(pred.max())}, expected 0..{k - 1}")
        sp = [float(abs(x)) for x in np.diag(aff)[:3]]
        _save_case(out, {"main": {"patient_id": pid, "centre": centre, "subset": "test", "spacing_x": sp[0], "spacing_y": sp[1],
                                  "spacing_z": sp[2], **seg_row(pred, label, sp, k), "mean_u_all": float("nan"),
                                  "mean_u_roi": float("nan"), "unc_auroc_roi": float("nan")}}, {})
    write_outputs(res_dir, name, k, [pid for pid, _ in cases], [], train_keys, None)
    _atomic_json(res_dir / "manifest.json", {**identity, "kind": "external predictions (label maps)", "pred_dir": pred_dir.name,
                                             "n_test": len(cases), "rerun_reason": rerun, "resumed": mode == "resume",
                                             "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    lock = json.loads((res_dir / "LOCK.json").read_text(encoding="utf-8"))
    _atomic_json(res_dir / "LOCK.json", {**lock, "state": "finished", "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return res_dir
