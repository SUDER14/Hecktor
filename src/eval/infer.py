"""Sliding-window inference at native spacing, and NIfTI output of masks / uncertainty maps.

Windows are laid out on the CT grid (voxel starts, `overlap` fraction between
neighbours, last window flush with the border). For each CT window the PET window
is the same *physical* box on the PET grid (physical_crop machinery), so PET and CT
stay world-aligned exactly as in training. Regions outside the volume read as 0
(extract_padded), and only the in-volume part of each prediction is accumulated.

Accumulation: softmax heads average probabilities; the evidential head averages
EVIDENCE and derives alpha / p / u afterwards (averaging a Dirichlet's parameters is
well defined, averaging u directly would not be).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.data.physical_crop import (
    extract_padded, grid_from_affine, snap_start_index, window_voxels,
)
from src.models.evidential import dirichlet_summary, softmax_summary


def window_starts(n: int, p: int, overlap: float) -> list[int]:
    if n <= p:
        return [0]
    step = max(1, int(round(p * (1.0 - overlap))))
    s = list(range(0, n - p + 1, step))
    if s[-1] != n - p:
        s.append(n - p)
    return s


@torch.no_grad()
def sliding_window_dual(forward_fn, ct: np.ndarray, pet: np.ndarray, ct_affine, pet_affine,
                        patch: tuple[int, int, int], num_classes: int, head: str = "softmax",
                        overlap: float = 0.5, batch_size: int = 1, device="cpu") -> dict[str, np.ndarray]:
    """forward_fn(ct_patch (b,1,*patch), pet_patch (b,1,*pet_patch)) -> (b,K,*patch).

    ct (X,Y,Z), pet (x,y,z) are numpy arrays on their own grids. Returns {"p": (K,X,Y,Z),
    "u": (1,X,Y,Z), "pred": (X,Y,Z) int}.
    """
    cg, pg = grid_from_affine(ct.shape, ct_affine, "ct"), grid_from_affine(pet.shape, pet_affine, "pet")
    extent = np.asarray(patch) * cg.spacing
    pet_n = window_voxels(extent, pg.spacing)
    acc = np.zeros((num_classes,) + ct.shape, np.float32)
    cnt = np.zeros(ct.shape, np.float32)
    starts = [(i, j, k) for i in window_starts(ct.shape[0], patch[0], overlap)
              for j in window_starts(ct.shape[1], patch[1], overlap)
              for k in window_starts(ct.shape[2], patch[2], overlap)]
    for b0 in range(0, len(starts), batch_size):
        chunk = starts[b0:b0 + batch_size]
        cts, pets = [], []
        for st in chunk:
            cts.append(extract_padded(ct, st, patch))
            pidx = snap_start_index(pg, cg.start_edge(st), pet_n, clip=False)
            pets.append(extract_padded(pet, pidx, pet_n))
        ct_t = torch.from_numpy(np.stack(cts)[:, None]).float().to(device)    # (b,1,*patch)
        pet_t = torch.from_numpy(np.stack(pets)[:, None]).float().to(device)  # (b,1,*pet_patch)
        out = forward_fn(ct_t, pet_t).float()                                  # (b,K,*patch)
        out = (torch.softmax(out, 1) if head == "softmax" else out).cpu().numpy()
        for st, o in zip(chunk, out):
            sl = tuple(slice(s, min(s + p, n)) for s, p, n in zip(st, patch, ct.shape))
            inner = tuple(slice(0, s.stop - s.start) for s in sl)
            acc[(slice(None),) + sl] += o[(slice(None),) + inner]
            cnt[sl] += 1
    acc /= np.maximum(cnt, 1)[None]
    t = torch.from_numpy(acc)[None]
    summ = dirichlet_summary(t) if head == "evidential" else softmax_summary(t)
    p, u = summ["p"][0].numpy(), summ["u"][0].numpy()
    return {"p": p, "u": u, "pred": p.argmax(0).astype(np.uint8)}


def save_nifti(array: np.ndarray, affine, path: str | Path, dtype=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = array if dtype is None else array.astype(dtype)
    nib.save(nib.Nifti1Image(arr, np.asarray(affine, float)), str(path))
    return path


def resample_to_grid(arr: np.ndarray, src_affine, dst_affine, dst_shape, order: int = 1) -> np.ndarray:
    """Resample (..., X, Y, Z) from one axis-aligned grid onto another (diagonal affines).

    Used ONLY to bring resampled-pipeline predictions (B1/B2) back onto the native CT
    grid, so every model is scored against the same native label.
    """
    from scipy import ndimage

    sg, dg = grid_from_affine(arr.shape, src_affine, "src"), grid_from_affine(dst_shape, dst_affine, "dst")
    axes = [(dg.origin[i] + np.arange(dst_shape[i]) * dg.spacing[i] - sg.origin[i]) / sg.spacing[i] for i in range(3)]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"))
    lead = arr.shape[:-3]
    flat = arr.reshape((-1,) + arr.shape[-3:])
    out = np.stack([ndimage.map_coordinates(a.astype(np.float32), coords, order=order, mode="nearest") for a in flat])
    return out.reshape(lead + tuple(dst_shape))
