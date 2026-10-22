"""Crop CT and PET patches in world (mm) coordinates, without resampling.

MONAI's RandCropByPosNegLabeld crops every key with the same voxel indices,
which is only right when all keys share one grid. CT and PET do not (the 4-5x
resolution asymmetry), so here the patch is chosen in mm and each modality gets
its own voxel window: size = extent / own spacing, start = the voxel whose
leading edge is nearest the shared world start.

Conventions (all after Orientationd "RAS", so the affine is diagonal and
positive): voxel i of an axis is centred at origin + i*spacing, so its leading
edge is origin + (i - 0.5)*spacing. A window of n voxels starting at index i0
covers [edge(i0), edge(i0) + n*spacing]. Snapping each modality's start to its
own grid leaves at most half a voxel of start error per modality.

Oblique / non-RAS affines are rejected rather than approximated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from monai.transforms import MapTransform, Randomizable

from src.data.geometry import check_physical_extent, patch_extent_mm


@dataclass(frozen=True)
class Grid:
    shape: tuple[int, int, int]
    origin: np.ndarray   # world mm of voxel (0,0,0)'s centre
    spacing: np.ndarray  # mm per voxel, positive

    def start_edge(self, idx) -> np.ndarray:
        """World mm of the leading edge of voxel `idx` (per axis)."""
        return self.origin + (np.asarray(idx) - 0.5) * self.spacing

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


def window_voxels(extent_mm, spacing) -> tuple[int, int, int]:
    return tuple(int(max(1, round(e / s))) for e, s in zip(extent_mm, spacing))


def feasible_start_edges(grid: Grid, n_vox, name: str = "volume") -> tuple[np.ndarray, np.ndarray]:
    """World range of window start edges that keep an n_vox window inside the volume."""
    n = np.asarray(n_vox)
    if (n > np.asarray(grid.shape)).any():
        raise ValueError(f"{name}: window {tuple(n_vox)} voxels exceeds volume shape {grid.shape}")
    return grid.start_edge(0), grid.start_edge(np.asarray(grid.shape) - n)


def snap_start_index(grid: Grid, start_edge_mm, n_vox) -> np.ndarray:
    idx = np.rint((np.asarray(start_edge_mm) - grid.start_edge(0)) / grid.spacing).astype(int)
    return np.clip(idx, 0, np.asarray(grid.shape) - np.asarray(n_vox))


def physical_crop(
    volumes: dict[str, tuple[np.ndarray, np.ndarray]],
    reference: str,
    reference_voxels,
    centre_mm,
    tol_voxels: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Crop every volume to the physical window of the `reference` patch.

    volumes: name -> (array [C, X, Y, Z] or [X, Y, Z], 4x4 affine). Anything that
    lives on the reference grid (the label) can be passed as its own entry with
    the reference affine.
    reference_voxels: patch size on the reference (CT) grid. Defines extent_mm.
    centre_mm: desired patch centre in world mm; clamped, jointly for all
    volumes, so every window stays inside every volume (clamping per volume
    would shift one window away from the others).

    Returns (patches, info) with info["voxels"/"start_idx"/"start_mm"/"spacing"]
    keyed by volume name.
    """
    grids = {n: grid_from_affine(a.shape, aff, n) for n, (a, aff) in volumes.items()}
    ref = grids[reference]
    extent = patch_extent_mm(reference_voxels, ref.spacing)
    voxels = {n: window_voxels(extent, g.spacing) for n, g in grids.items()}
    voxels[reference] = tuple(int(v) for v in reference_voxels)
    for n, g in grids.items():
        if n != reference:
            check_physical_extent(reference_voxels, ref.spacing, voxels[n], g.spacing,
                                  tol_voxels, context=f"{n} vs {reference}")

    lo = np.full(3, -np.inf)
    hi = np.full(3, np.inf)
    for n, g in grids.items():
        l, h = feasible_start_edges(g, voxels[n], n)
        lo, hi = np.maximum(lo, l), np.minimum(hi, h)
    if (lo > hi).any():
        raise ValueError(
            f"no common physical window of {np.round(extent, 1).tolist()} mm fits inside all "
            f"volumes (start-edge range lo={np.round(lo, 1).tolist()} hi={np.round(hi, 1).tolist()}); "
            "the fields of view overlap too little for this patch size.")
    start = np.clip(np.asarray(centre_mm, dtype=float) - extent / 2.0, lo, hi)

    patches, info = {}, {"voxels": {}, "start_idx": {}, "start_mm": {}, "spacing": {}}
    for n, (arr, _) in volumes.items():
        g = grids[n]
        idx = snap_start_index(g, start, voxels[n])
        sl = tuple(slice(int(i), int(i + v)) for i, v in zip(idx, voxels[n]))
        patches[n] = arr[(...,) + sl]
        info["voxels"][n], info["start_idx"][n] = voxels[n], idx
        info["start_mm"][n], info["spacing"][n] = g.start_edge(idx), g.spacing
    return patches, info


def sample_centre_mm(label: np.ndarray, label_affine, pos: float, neg: float,
                     rng: np.random.RandomState) -> np.ndarray:
    """Pick a patch centre in world mm: on a random foreground voxel with
    probability pos/(pos+neg), otherwise uniformly anywhere in the label's
    field of view (physical_crop then clamps it to a feasible window)."""
    grid = grid_from_affine(label.shape, label_affine, "label")
    fg = np.argwhere((label[0] if label.ndim == 4 else label) > 0)
    if len(fg) and rng.rand() < pos / (pos + neg):
        return grid.voxel_to_world(fg[rng.randint(len(fg))])
    return grid.voxel_to_world(rng.rand(3) * (np.asarray(grid.shape) - 1))


class RandPhysicalCropd(Randomizable, MapTransform):
    """MONAI dict transform: joint CT+PET(+label) crop in world coordinates.

    `spatial_size` is the CT patch in voxels (same meaning as in
    RandCropByPosNegLabeld); the PET patch size is derived from the physical
    extent. Returns a list of `num_samples` dicts, each with plain tensors
    ct/pet/label plus ct_spacing / pet_spacing / ct_start_mm / pet_start_mm,
    so the model can run its own physical-extent check on exactly what was cut.
    The label is assumed to sit on the CT grid (verified, not assumed silently).
    """

    def __init__(self, ct_key: str, pet_key: str, label_key: str, spatial_size, pos: float = 1.0,
                 neg: float = 1.0, num_samples: int = 1, tol_voxels: float = 1.0):
        MapTransform.__init__(self, [ct_key, pet_key, label_key])
        self.ct_key, self.pet_key, self.label_key = ct_key, pet_key, label_key
        self.spatial_size, self.pos, self.neg = tuple(spatial_size), pos, neg
        self.num_samples, self.tol_voxels = num_samples, tol_voxels

    def randomize(self, data=None) -> None:
        pass  # randomness is drawn from self.R inside __call__ (one centre per sample)

    def __call__(self, data):
        d = dict(data)
        arrs = {k: (np.asarray(d[k]), np.asarray(d[k].affine)) for k in
                (self.ct_key, self.pet_key, self.label_key)}
        if arrs[self.label_key][0].shape[-3:] != arrs[self.ct_key][0].shape[-3:]:
            raise ValueError("label and CT grids differ; the label is assumed to be on the CT grid")
        arrs[self.label_key] = (arrs[self.label_key][0], arrs[self.ct_key][1])
        out = []
        for _ in range(self.num_samples):
            centre = sample_centre_mm(arrs[self.label_key][0], arrs[self.ct_key][1],
                                      self.pos, self.neg, self.R)
            patches, info = physical_crop(arrs, self.ct_key, self.spatial_size, centre, self.tol_voxels)
            item = {k: v for k, v in d.items() if k not in arrs}
            for k, p in patches.items():
                item[k] = torch.as_tensor(np.ascontiguousarray(p))
            item["ct_spacing"] = torch.as_tensor(info["spacing"][self.ct_key], dtype=torch.float32)
            item["pet_spacing"] = torch.as_tensor(info["spacing"][self.pet_key], dtype=torch.float32)
            item["ct_start_mm"] = torch.as_tensor(info["start_mm"][self.ct_key], dtype=torch.float32)
            item["pet_start_mm"] = torch.as_tensor(info["start_mm"][self.pet_key], dtype=torch.float32)
            out.append(item)
        return out
