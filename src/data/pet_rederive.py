"""Re-derive a coarse PET grid from the PET released on the CT grid ("pet_mode: rederived").

HECKTOR 2026 ships PET already resampled onto the CT grid (VERIFIED, all 782 cases), so the
acquisition-grid PET (typically a few mm per voxel) is NOT available. To keep the
dual-resolution ACF path alive, this module goes back DOWN to a coarse grid:

    1. anti-alias: Gaussian blur with sigma = (ratio - 1) / 2 CT voxels per axis (the
       scikit-image `downscale_local` convention), ratio = coarse spacing / CT spacing;
    2. sample the blurred volume at the coarse voxel centres with linear interpolation.

This is an APPROXIMATION of the acquisition grid. It cannot recover detail the released
(already interpolated) PET lacks, and its blur/sampling is not the scanner's PSF or
reconstruction grid. It must never be called "native PET" in code, configs or reports.

Axes where the CT is already coarser than the target keep the CT spacing (max(target, CT)):
going finer would invent detail.

Diagonal positive affines only (after reorientation to RAS), like the rest of the pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

DEFAULT_PET_SPACING = (4.0, 4.0, 4.0)


def coarse_spacing(ct_spacing, target=DEFAULT_PET_SPACING) -> np.ndarray:
    return np.maximum(np.asarray(target, float), np.asarray(ct_spacing, float))


def rederive_pet(pet: np.ndarray, ct_affine, target=DEFAULT_PET_SPACING):
    """pet (X,Y,Z) on the CT grid -> (coarse array, coarse 4x4 affine, info dict)."""
    aff = np.asarray(ct_affine, float)
    lin = aff[:3, :3]
    if np.abs(lin - np.diag(np.diag(lin))).max() > 1e-4 * np.abs(np.diag(lin)).max() or (np.diag(lin) <= 0).any():
        raise ValueError("rederive_pet expects a diagonal-positive (RAS) affine")
    sp = np.diag(lin).copy()
    cs = coarse_spacing(sp, target)
    ratio = cs / sp
    sigma = np.maximum(0.0, (ratio - 1.0) / 2.0)
    src = ndimage.gaussian_filter(pet.astype(np.float32), sigma=sigma, mode="nearest") if sigma.any() \
        else pet.astype(np.float32)
    n_out = np.maximum(1, np.floor(np.asarray(pet.shape) / ratio + 1e-6).astype(int))
    # coarse voxel j covers CT indices [j*ratio, (j+1)*ratio): centre at (j+0.5)*ratio - 0.5 in CT index units
    axes = [(np.arange(n) + 0.5) * r - 0.5 for n, r in zip(n_out, ratio)]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"))
    out = ndimage.map_coordinates(src, coords, order=1, mode="nearest").astype(np.float32)
    caff = np.eye(4)
    caff[range(3), range(3)] = cs
    caff[:3, 3] = aff[:3, 3] + (np.array([a[0] for a in axes])) * sp  # world position of coarse voxel 0's centre
    return out, caff, {"coarse_spacing": cs.tolist(), "sigma_ct_voxels": sigma.tolist(), "shape": out.shape}


def upsample_to_ct(coarse: np.ndarray, coarse_affine, ct_shape, ct_affine, chunk: int = 16) -> np.ndarray:
    """Linear resampling of the coarse PET back onto the CT grid (chunked along x for memory)."""
    ca, ta = np.asarray(coarse_affine, float), np.asarray(ct_affine, float)
    cs, co = np.diag(ca[:3, :3]), ca[:3, 3]
    ts, to = np.diag(ta[:3, :3]), ta[:3, 3]
    out = np.empty(tuple(ct_shape), np.float32)
    ay = (to[1] + np.arange(ct_shape[1]) * ts[1] - co[1]) / cs[1]
    az = (to[2] + np.arange(ct_shape[2]) * ts[2] - co[2]) / cs[2]
    for x0 in range(0, ct_shape[0], chunk):
        ax = (to[0] + np.arange(x0, min(x0 + chunk, ct_shape[0])) * ts[0] - co[0]) / cs[0]
        coords = np.stack(np.meshgrid(ax, ay, az, indexing="ij"))
        out[x0:x0 + len(ax)] = ndimage.map_coordinates(coarse, coords, order=1, mode="nearest")
    return out


def roundtrip_metrics(pet: np.ndarray, back: np.ndarray, label: np.ndarray | None = None, chunk: int = 16) -> dict:
    """How much the down/up round trip loses, against the released PET. All measured, none assumed.

    Streams over x-slabs with float64 running sums, so a 1.7e8-voxel volume never needs a float64 copy
    (8 GB machines): Pearson r from n, sum(a), sum(b), sum(a^2), sum(b^2), sum(ab)."""
    def acc():
        return {"n": 0, "sa": 0.0, "sb": 0.0, "saa": 0.0, "sbb": 0.0, "sab": 0.0, "see": 0.0, "emax": 0.0,
                "amax": -np.inf, "bmax": -np.inf}

    def add(t, a, b):
        if a.size == 0:
            return
        a, b = a.astype(np.float64), b.astype(np.float64)
        e = b - a
        t["n"] += a.size
        t["sa"] += a.sum(); t["sb"] += b.sum()
        t["saa"] += (a * a).sum(); t["sbb"] += (b * b).sum(); t["sab"] += (a * b).sum()
        t["see"] += (e * e).sum()
        t["emax"] = max(t["emax"], float(np.abs(e).max()))
        t["amax"], t["bmax"] = max(t["amax"], float(a.max())), max(t["bmax"], float(b.max()))

    def r(t):
        n = t["n"]
        va, vb = t["saa"] / n - (t["sa"] / n) ** 2, t["sbb"] / n - (t["sb"] / n) ** 2
        return float((t["sab"] / n - t["sa"] * t["sb"] / n ** 2) / np.sqrt(va * vb)) if va > 0 and vb > 0 else float("nan")

    whole, gtv = acc(), acc()
    for x0 in range(0, pet.shape[0], chunk):
        a, b = pet[x0:x0 + chunk], back[x0:x0 + chunk]
        add(whole, a.ravel(), b.ravel())
        if label is not None:
            m = label[x0:x0 + chunk] > 0
            add(gtv, a[m], b[m])
    out = {"pearson_r": r(whole), "rmse": float(np.sqrt(whole["see"] / whole["n"])), "max_abs_err": whole["emax"],
           "pet_max": whole["amax"], "max_abs_err_rel_pet_max": whole["emax"] / max(whole["amax"], 1e-12)}
    if label is not None and gtv["n"] > 0:
        out.update({"pearson_r_gtv": r(gtv) if gtv["n"] > 2 else float("nan"),
                    "rmse_gtv": float(np.sqrt(gtv["see"] / gtv["n"])),
                    "gtv_max_orig": gtv["amax"], "gtv_max_roundtrip": gtv["bmax"],
                    "gtv_max_rel_change": (gtv["bmax"] - gtv["amax"]) / max(gtv["amax"], 1e-12),
                    "gtv_mean_rel_change": (gtv["sb"] - gtv["sa"]) / max(gtv["sa"], 1e-12)})
    return out
