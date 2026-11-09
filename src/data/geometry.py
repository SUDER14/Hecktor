"""Physical-extent bookkeeping for patches that live on different voxel grids.

Numpy-only on purpose: both the model (src/models/dual_encoder.py) and the
loader import this, and neither should pull the other's dependencies in.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def patch_extent_mm(voxels: Sequence[int], spacing: Sequence[float]) -> np.ndarray:
    """Physical side lengths (mm) of a patch: voxel count x voxel size, per axis."""
    return np.asarray(voxels, dtype=float) * np.asarray(spacing, dtype=float)


def check_physical_extent(
    ct_voxels: Sequence[int],
    ct_spacing: Sequence[float],
    pet_voxels: Sequence[int],
    pet_spacing: Sequence[float],
    tol_voxels: float = 1.0,
    context: str = "",
) -> None:
    """Raise if the CT and PET patches do not cover the same physical extent.

    A fixed voxel-count rule cannot say whether 24^3 PET goes with 96^3 CT: that
    depends on the spacings (1 mm / 4 mm: yes; 1 mm / 2 mm: no). Comparing
    extents in mm does. The tolerance is `tol_voxels` coarser voxels per axis,
    because a window can only be placed on whole voxels of each grid, so a
    sub-voxel mismatch on the coarser grid is unavoidable, not a bug.
    """
    ct_ext = patch_extent_mm(ct_voxels, ct_spacing)
    pet_ext = patch_extent_mm(pet_voxels, pet_spacing)
    coarser = np.maximum(np.asarray(ct_spacing, dtype=float), np.asarray(pet_spacing, dtype=float))
    diff = np.abs(ct_ext - pet_ext)
    bad = diff > tol_voxels * coarser + 1e-6
    if bad.any():
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}CT and PET patches cover different physical extents "
            f"(axes {np.flatnonzero(bad).tolist()} exceed the {tol_voxels} coarser-voxel "
            f"tolerance). CT {tuple(int(v) for v in ct_voxels)} vox x {tuple(float(s) for s in ct_spacing)} mm "
            f"= {np.round(ct_ext, 2).tolist()} mm; PET {tuple(int(v) for v in pet_voxels)} vox x "
            f"{tuple(float(s) for s in pet_spacing)} mm = {np.round(pet_ext, 2).tolist()} mm; "
            f"allowed difference {np.round(tol_voxels * coarser, 2).tolist()} mm per axis."
        )


# ---------------------------------------------------------------------------
# Shared-grid ("shared" pet_mode): fixed-voxel patches at native spacing. CT, PET and label
# are on one grid, so there is no cross-modality extent; this only guards against a patch
# that is physically absurd for its scan. Bounds are a guard, not a tuned value.
# ---------------------------------------------------------------------------
DEFAULT_MIN_EXTENT_MM = (8.0, 8.0, 8.0)
DEFAULT_MAX_EXTENT_MM = (1200.0, 1200.0, 1500.0)


def check_patch_extent(voxels, spacing, min_mm=DEFAULT_MIN_EXTENT_MM, max_mm=DEFAULT_MAX_EXTENT_MM,
                       context: str = "") -> np.ndarray:
    """Raise if a voxel patch at `spacing` is physically implausible; return its extent (mm).

    Too small: cannot hold a tumour plus context. Too large: bigger than any anatomy in the
    cohort, usually a spacing in the wrong unit. Also rejects non-positive/non-finite spacing.
    """
    sp = np.asarray(spacing, dtype=float)
    prefix = f"{context}: " if context else ""
    if sp.shape != (3,) or not np.isfinite(sp).all() or (sp <= 0).any():
        raise ValueError(f"{prefix}spacing must be 3 positive finite numbers (mm), got {sp.tolist()}")
    ext = patch_extent_mm(voxels, sp)
    lo, hi = np.asarray(min_mm, dtype=float), np.asarray(max_mm, dtype=float)
    if (ext < lo).any() or (ext > hi).any():
        raise ValueError(
            f"{prefix}patch {tuple(int(v) for v in voxels)} vox x {tuple(float(s) for s in sp)} mm = "
            f"{np.round(ext, 1).tolist()} mm is outside the allowed physical range "
            f"[{lo.tolist()}, {hi.tolist()}] mm per axis.")
    return ext
