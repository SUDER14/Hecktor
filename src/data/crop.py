"""Fixed-size voxel crops on a single shared grid.

CT, PET and label share one voxel grid in HECKTOR 2026 (VERIFIED: identical shape,
spacing and affine for all 782 cases), so one set of voxel indices crops every
channel and no world-coordinate bookkeeping between modalities is needed. The patch
is fixed in VOXELS at each scan's native spacing, so its extent in mm varies with
the scan (see src/data/geometry.py) -- that variation is what the spacing
conditioning has to handle; nothing is resampled.

Oblique / non-RAS affines are rejected rather than approximated (the manifest found
none in the cohort after reorientation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    shape: tuple[int, int, int]
    origin: np.ndarray   # world mm of voxel (0,0,0)'s centre
    spacing: np.ndarray  # mm per voxel, positive

    def voxel_to_world(self, idx) -> np.ndarray:
        return self.origin + np.asarray(idx) * self.spacing


def grid_from_affine(shape, affine, name: str = "volume") -> Grid:
    a = np.asarray(affine, dtype=float)
    lin = a[:3, :3]
    off_diag = lin - np.diag(np.diag(lin))
    if np.abs(off_diag).max() > 1e-4 * max(np.abs(np.diag(lin)).max(), 1e-12) or (np.diag(lin) <= 0).any():
        raise ValueError(
            f"{name}: affine linear part {lin.round(4).tolist()} is not diagonal-positive; "
            "physical cropping expects Orientationd('RAS') on an axis-aligned volume. "
            "Oblique acquisitions would need real resampling, which this pipeline does not do."
        )
    return Grid(tuple(int(d) for d in shape[-3:]), a[:3, 3].copy(), np.diag(lin).copy())


def extract_padded(arr: np.ndarray, idx, n_vox) -> np.ndarray:
    """arr[..., idx:idx+n] where indices outside the volume read as 0.

    Padding keeps world positions: output voxel j sits at source voxel idx+j, so a
    pad voxel is exactly where the grid would continue. 0 is air for a CT scaled
    to [0, 1] with a -1000 HU floor, zero uptake for PET, and background for the
    label."""
    out = np.zeros(arr.shape[:-3] + tuple(int(v) for v in n_vox), dtype=arr.dtype)
    src, dst = [], []
    for i0, n, size in zip(idx, n_vox, arr.shape[-3:]):
        a, b = max(int(i0), 0), min(int(i0) + int(n), size)
        if b <= a:
            raise ValueError("padded window does not overlap the volume at all")
        src.append(slice(a, b))
        dst.append(slice(a - int(i0), b - int(i0)))
    out[(...,) + tuple(dst)] = arr[(...,) + tuple(src)]
    return out



def crop_start(centre_vox, size, shape, pad: bool = True) -> np.ndarray:
    """Start index of a `size` window centred (as nearly as possible) on `centre_vox`.

    pad=False clamps the window inside the volume (an error if the volume is smaller
    than the window); pad=True lets it hang over the border, where extract_padded
    reads zeros, but still keeps at least part of it inside the volume.
    """
    size, shape = np.asarray(size), np.asarray(shape)
    start = np.rint(np.asarray(centre_vox, float) - size / 2.0).astype(int)
    if pad:
        return np.clip(start, -(size - 1), shape - 1)
    if (size > shape).any():
        raise ValueError(f"window {tuple(size)} voxels exceeds volume shape {tuple(shape)}; use pad=True")
    return np.clip(start, 0, shape - size)


def crop_patch(arrays: dict[str, np.ndarray], centre_vox, size, pad: bool = True):
    """Same window from every array (each [C, X, Y, Z] or [X, Y, Z], one shared grid).

    Returns (patches, start_idx). All arrays must have the same spatial shape: a
    mismatch means they are not on one grid, which is an error rather than something
    to crop around.
    """
    shapes = {n: tuple(a.shape[-3:]) for n, a in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"arrays are not on one grid: {shapes}")
    shape = next(iter(shapes.values()))
    start = crop_start(centre_vox, size, shape, pad)
    if pad:
        return {n: extract_padded(a, start, size) for n, a in arrays.items()}, start
    sl = tuple(slice(int(i), int(i + v)) for i, v in zip(start, size))
    return {n: a[(...,) + sl] for n, a in arrays.items()}, start


def sample_centre_vox(label_fg: np.ndarray, shape, pos: float, neg: float, rng) -> np.ndarray:
    """Patch centre in voxels: a random foreground voxel with probability pos/(pos+neg),
    otherwise uniform over the volume. `label_fg` is an (N, 3) array of foreground indices."""
    if len(label_fg) and rng.random() < pos / (pos + neg):
        return np.asarray(label_fg[rng.integers(len(label_fg))], float)
    return rng.random(3) * (np.asarray(shape) - 1)
