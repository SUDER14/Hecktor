"""Label-free head-and-neck crop box (same rule for train, val and test).

Why: the full-field-of-view cache is too large (see docs/DATA_REPORT.md). Most scans include
chest/abdomen that never contain a head-and-neck GTV, so the cache stores a fixed-physical-size
box instead. The rule uses the CT only -- never the labels. Labels are used only afterwards, to
VERIFY that no label voxel falls outside the box (scripts/crop_scan.py, scripts/crop_verify.py).
If a case fails verification the RULE parameters change for everyone; a per-case box is never used.

Rule (volumes already reoriented to RAS, so +z is superior and the affine is diagonal):
  1. body = CT > BODY_HU. A slice counts as body if its body area >= MIN_AREA_MM2 (rejects
     stray noise voxels and slice-edge artefacts).
  2. top-of-head = the most superior body slice.
  3. z window = the slab of `length_mm` going DOWN from the top-of-head slice.
  4. x/y window = a `width_mm` square centred on the centroid of the body voxels inside that slab.
  5. The box is clamped to the volume (shifted, not shrunk, when possible). A scan that is shorter
     than `length_mm` simply keeps everything it has.

Everything is in voxel indices of the RAS volume; sizes in mm are converted with that scan's spacing.
"""

from __future__ import annotations

import numpy as np

BODY_HU = -300.0
MIN_AREA_MM2 = 1500.0


def body_slice_stats(ct: np.ndarray, spacing, lengths_mm=(300.0,)) -> dict:
    """Per-scan numbers the rule needs. `ct` (X,Y,Z) in HU, RAS. One pass over z, slice by slice,
    so peak memory is the volume plus one boolean slice."""
    sx, sy, sz = (float(s) for s in spacing)
    nz = ct.shape[2]
    area = np.zeros(nz)
    sums = np.zeros((nz, 2))          # per-slice sum of x and y indices of body voxels
    cnt = np.zeros(nz)
    xs = np.arange(ct.shape[0], dtype=np.float64)[:, None]
    ys = np.arange(ct.shape[1], dtype=np.float64)[None, :]
    for k in range(nz):
        m = ct[:, :, k] > BODY_HU
        c = float(m.sum())
        cnt[k], area[k] = c, c * sx * sy
        if c:
            sums[k] = ((m * xs).sum(), (m * ys).sum())
    ok = np.flatnonzero(area >= MIN_AREA_MM2)
    if len(ok) == 0:
        raise ValueError("no slice with body area >= %.0f mm2; CT range wrong?" % MIN_AREA_MM2)
    top_k, bottom_k = int(ok.max()), int(ok.min())
    out = {"top_k": top_k, "bottom_k": bottom_k}
    for L in lengths_mm:
        k0 = max(0, top_k - int(round(L / sz)) + 1)
        n = cnt[k0:top_k + 1].sum()
        cx, cy = (sums[k0:top_k + 1].sum(0) / n) if n else (ct.shape[0] / 2, ct.shape[1] / 2)
        out[f"cx_{int(L)}"], out[f"cy_{int(L)}"] = float(cx), float(cy)
    return out


def box_from_stats(shape, spacing, top_k: int, cx: float, cy: float, length_mm: float, width_mm: float):
    """Index box ((x0,x1),(y0,y1),(z0,z1)), half-open, inside the volume."""
    sx, sy, sz = (float(s) for s in spacing)
    nz_box = int(round(length_mm / sz))
    z1 = min(shape[2], top_k + 1)
    z0 = max(0, z1 - nz_box)
    box = [(0, 0), (0, 0), (z0, z1)]
    for ax, (c, s) in enumerate(((cx, sx), (cy, sy))):
        n = int(round(width_mm / s))
        if n >= shape[ax]:
            box[ax] = (0, shape[ax])
            continue
        a = int(round(c - n / 2.0))
        a = min(max(a, 0), shape[ax] - n)   # shift, do not shrink
        box[ax] = (a, a + n)
    return tuple(box)


def head_neck_box(ct: np.ndarray, spacing, length_mm: float, width_mm: float):
    st = body_slice_stats(ct, spacing, (length_mm,))
    return box_from_stats(ct.shape, spacing, st["top_k"], st[f"cx_{int(length_mm)}"], st[f"cy_{int(length_mm)}"],
                          length_mm, width_mm)


def margin_mm(box, spacing, lab_bbox) -> float:
    """Smallest distance (mm) from a label bounding box to any box face; negative = outside the box.
    lab_bbox = ((x0,x1),(y0,y1),(z0,z1)) inclusive-exclusive voxel indices of the labelled voxels."""
    m = np.inf
    for (b0, b1), (l0, l1), s in zip(box, lab_bbox, spacing):
        m = min(m, (l0 - b0) * s, (b1 - l1) * s)
    return float(m)
