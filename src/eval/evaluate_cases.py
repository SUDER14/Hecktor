"""Per-case evaluation of any model (B1 / B2 / ACF) against the NATIVE label.

Every model is scored on the same grid and label: the native CT grid. B1/B2 predict on a
resampled grid, so their probability / uncertainty maps are resampled back (trilinear)
before argmax -- otherwise their numbers would live on a different grid from ACF's and
not be comparable. HD95 and Surface Dice use the native CT voxel spacing in mm.

`degrade_z_mm` simulates thick-slice acquisition for the anisotropy experiment: CT slices
are block-averaged along z to the target thickness (conditioning is recomputed from the
new spacing), prediction is resampled back and scored against the unchanged native label.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from src.data.conditioning import make_cond4
from src.eval.infer import resample_to_grid, save_nifti, sliding_window_dual
from src.metrics.calibration import ece, reliability_bins, uncertainty_error_overlap
from src.metrics.segmentation import dice, hd95, surface_dice


def call_model(model, ct, pet, cond_ct, cond_pet, ct_sp, pet_sp):
    """One place that knows each model family's forward signature."""
    name = type(model).__name__
    if name == "ACF":
        return model(ct, pet, cond_ct, cond_pet, ct_sp, pet_sp)
    if name == "DualEncoderUNet":
        return model(ct, pet, ct_sp, pet_sp)
    return model(ct)  # UNet3D: CT only


def coarsen_z(vol: np.ndarray, affine: np.ndarray, target_mm: float):
    """Block-average along z to ~target_mm thickness; returns (volume, affine)."""
    f = max(1, int(round(target_mm / abs(affine[2, 2]))))
    if f == 1:
        return vol, affine
    n = (vol.shape[2] // f) * f
    out = vol[:, :, :n].reshape(vol.shape[0], vol.shape[1], n // f, f).mean(3)
    aff = affine.copy()
    aff[2, 2] = affine[2, 2] * f
    aff[:3, 3] = affine[:3, 3] + np.array([0, 0, (f - 1) / 2 * affine[2, 2]])  # block centre
    return out, aff


def _native_label(sample: dict, cache_label: np.ndarray, cache_affine, native_mode: bool):
    if native_mode:
        return cache_label, np.asarray(cache_affine)
    img = nib.as_closest_canonical(nib.load(sample["label"]))
    return np.asarray(img.dataobj).astype(np.uint8), img.affine


def evaluate_dataset(model, dataset, config: dict, stats: dict, device="cpu", out_dir: str | Path | None = None,
                     save_maps: bool = False, degrade_z_mm: float | None = None, max_cases: int | None = None,
                     git_hash: str | None = None, tag: str = "") -> pd.DataFrame:
    model.eval().to(device)
    patch = tuple(config["patch"]["spatial_size"])
    head = getattr(model, "head_type", "softmax")
    k = config["model"].get("out_channels", 2)
    native_mode = config.get("data", {}).get("mode", "native") == "native"
    overlap = config.get("infer", {}).get("overlap", 0.5)
    sw_bs = config.get("infer", {}).get("sw_batch_size", 1)
    rows, pooled = [], None
    for idx in range(min(len(dataset), max_cases or len(dataset))):
        item, sample = dataset[idx], dataset.samples[idx]
        ct, pet = item["ct"][0].numpy(), item["pet"][0].numpy()
        ct_aff, pet_aff = item["ct_affine"].numpy(), item["pet_affine"].numpy()
        label, lab_aff = _native_label(sample, item["label"][0].numpy().astype(np.uint8), ct_aff, native_mode)
        cond_ct, ct_sp = item["cond_ct"], item["ct_spacing"]
        eff_sz = float(abs(ct_aff[2, 2]))
        if degrade_z_mm is not None:
            thick, thick_aff = coarsen_z(ct, ct_aff, degrade_z_mm)
            eff_sz = float(abs(thick_aff[2, 2]))
            if native_mode:  # ACF sees the thick-slice grid and is told its spacing via conditioning
                ct, ct_aff = thick, thick_aff
                ct_sp = torch.tensor(np.diag(ct_aff)[:3], dtype=torch.float32)
                cond_ct = torch.from_numpy(make_cond4(ct_sp.numpy(), "CT", stats))
            else:  # B1/B2 are fed a resampled grid: thick slices, interpolated back to it
                ct = resample_to_grid(thick, thick_aff, ct_aff, ct.shape, order=1)
        # the PET window is derived from the CT window's extent, so a thick-slice CT must use a
        # patch whose z extent still divides: pad handles ragged ends, divisibility is the user's patch
        cc, cp = cond_ct[None].to(device), item["cond_pet"][None].to(device)
        csp, psp = ct_sp[None].to(device), item["pet_spacing"][None].to(device)

        def fwd(c, p, cc=cc, cp=cp, csp=csp, psp=psp):
            b = c.shape[0]
            return call_model(model, c, p, cc.expand(b, -1), cp.expand(b, -1), csp.expand(b, -1), psp.expand(b, -1))

        t0 = time.perf_counter()
        res = sliding_window_dual(fwd, ct, pet, ct_aff, pet_aff, patch, k, head, overlap, sw_bs, device)
        dt = time.perf_counter() - t0
        p, u = res["p"], res["u"]
        same_grid = p.shape[1:] == label.shape and np.allclose(ct_aff, lab_aff)
        if not same_grid:  # resampled / degraded pipelines: back to the native label grid
            p = resample_to_grid(p, ct_aff, lab_aff, label.shape, order=1)
            u = resample_to_grid(u, ct_aff, lab_aff, label.shape, order=1)
        pred = p.argmax(0).astype(np.uint8)
        sp = tuple(float(abs(x)) for x in np.diag(lab_aff)[:3])
        tum, lab_b = pred == 1, label == 1
        h = hd95(tum, lab_b, sp)
        roi = ndimage.binary_dilation(tum | lab_b, iterations=5) if (tum | lab_b).any() else None
        unc = uncertainty_error_overlap(u[0], pred, label.astype(np.uint8), roi)
        rb = reliability_bins(p, label, 15, roi)
        pooled = _pool(pooled, rb)
        rows.append({
            "patient_id": sample["patient_id"], "centre": sample["centre"], "tag": tag,
            "degrade_z_mm": degrade_z_mm, "effective_sz_mm": eff_sz, "dice": dice(tum, lab_b), "hd95_mm": h.value, "hd95_pred_empty": h.pred_empty,
            "surface_dice_2mm": surface_dice(tum, lab_b, sp, 2.0), "surface_dice_3mm": surface_dice(tum, lab_b, sp, 3.0),
            "ece_all": ece(p, label), "ece_roi": ece(p, label, roi=roi) if roi is not None else float("nan"),
            "unc_auroc": unc["auroc"], "unc_overlap_dice": unc["overlap_dice"],
            "mean_u": float(u.mean()), "label_voxels": int(lab_b.sum()), "infer_s": dt, "git": git_hash})
        if save_maps and out_dir is not None:
            pid = sample["patient_id"]
            save_nifti(pred, lab_aff, Path(out_dir) / f"{pid}_pred.nii.gz", np.uint8)
            save_nifti(u[0], lab_aff, Path(out_dir) / f"{pid}_uncertainty.nii.gz", np.float32)
    df = pd.DataFrame(rows)
    df.attrs["reliability"] = pooled
    return df


def _pool(acc, rb):
    """Accumulate reliability bins across cases: counts add, confidence / accuracy are count-weighted."""
    cnt = rb["count"]
    cur = {"edges": rb["edges"], "count": cnt.copy(), "sum_conf": np.nan_to_num(rb["confidence"]) * cnt,
           "sum_acc": np.nan_to_num(rb["accuracy"]) * cnt}
    if acc is None:
        return cur
    for k in ("count", "sum_conf", "sum_acc"):
        acc[k] = acc[k] + cur[k]
    return acc


def pooled_ece(pooled: dict) -> float:
    n = pooled["count"].sum()
    ok = pooled["count"] > 0
    conf, acc = pooled["sum_conf"][ok] / pooled["count"][ok], pooled["sum_acc"][ok] / pooled["count"][ok]
    return float(np.sum(pooled["count"][ok] / n * np.abs(acc - conf))) if n else float("nan")


def summarize(df: pd.DataFrame) -> dict:
    """NaN-aware means plus how many cases were excluded (empty labels), per the metric convention."""
    out = {"n_cases": len(df), "n_empty_label": int(df["dice"].isna().sum())}
    for c in ("dice", "hd95_mm", "surface_dice_2mm", "surface_dice_3mm", "ece_all", "ece_roi", "unc_auroc"):
        v = df[c].dropna()
        out[c] = float(v.mean()) if len(v) else float("nan")
    out["n_hd95_pred_empty"] = int(df["hd95_pred_empty"].sum())
    return out
