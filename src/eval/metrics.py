"""Segmentation metrics computed on CPU with numpy/scipy.

HD95 is implemented here instead of calling monai.metrics.compute_hausdorff_distance
because that call runs torch.quantile over the full surface-distance tensor and
raises "quantile() input tensor is too large" on full-resolution volumes
(torch's limit is ~16M elements). Nothing here is downsampled, so values stay
comparable across runs and volumes.

Definition (same convention as MONAI's, so earlier numbers are comparable in
kind): surface voxels are mask & ~binary_erosion(mask) (6-connectivity; voxels
on the array border count as surface because the erosion border value is 0).
For each surface voxel of A the distance to the nearest surface voxel of B is
read off a Euclidean distance transform of B's surface computed with the voxel
spacing, so distances are in mm and anisotropy is handled exactly. HD95 is the
larger of the two directed 95th percentiles (np.percentile, linear
interpolation).

Edge cases, reported separately instead of being folded into one number:
  * empty reference label -> value NaN (HD95 is undefined; callers exclude it).
  * empty prediction with a non-empty label -> value EMPTY_PRED_PENALTY_MM_DOC
    below, and pred_empty=True. The penalty is the volume's physical diagonal,
    i.e. the largest distance any two voxels in the volume can be apart, so it
    is finite, deterministic, and never smaller than a real HD95 on that volume.
    Callers should report how many samples hit it.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
from scipy import ndimage

EMPTY_PRED_PENALTY_MM_DOC = "physical diagonal of the volume in mm: sqrt(sum((shape_i * spacing_i)^2))"


class HD95Result(NamedTuple):
    value: float        # mm; NaN if the label is empty
    pred_empty: bool    # True when the prediction is empty and the label is not (value is the penalty)


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Boolean array marking voxels of `mask` that touch the background."""
    mask = mask.astype(bool)
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def _directed_percentile(src_surface: np.ndarray, dst_surface: np.ndarray, spacing, q: float) -> float:
    dist_to_dst = ndimage.distance_transform_edt(~dst_surface, sampling=spacing)
    return float(np.percentile(dist_to_dst[src_surface], q))


def hd95(pred: np.ndarray, label: np.ndarray, spacing: Sequence[float], percentile: float = 95.0) -> HD95Result:
    """HD95 in mm between two binary 3D masks. See module docstring for definition and edge cases."""
    pred, label = np.asarray(pred).astype(bool), np.asarray(label).astype(bool)
    if pred.shape != label.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs label {label.shape}")
    if len(spacing) != pred.ndim:
        raise ValueError(f"spacing has {len(spacing)} entries for a {pred.ndim}-D mask")
    if not label.any():
        return HD95Result(float("nan"), False)
    if not pred.any():
        diag = float(np.sqrt(sum((n * s) ** 2 for n, s in zip(pred.shape, spacing))))
        return HD95Result(diag, True)
    ps, ls = surface_voxels(pred), surface_voxels(label)
    return HD95Result(
        max(_directed_percentile(ps, ls, spacing, percentile), _directed_percentile(ls, ps, spacing, percentile)),
        False,
    )
