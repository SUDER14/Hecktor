"""Segmentation metrics on binary numpy masks (CPU, scipy). HD95 lives in src/eval/metrics.py
and is re-exported here so callers have one import location.

Empty-mask convention (shared by every metric here, documented once):
  * reference EMPTY      -> NaN. The metric is undefined; callers average with nan-aware
                            functions and should REPORT how many cases were excluded.
  * reference non-empty, prediction EMPTY -> Dice 0.0, Surface Dice 0.0, HD95 = volume
    diagonal penalty (see src/eval/metrics.py), flagged with pred_empty=True.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from src.eval.metrics import HD95Result, hd95, surface_voxels  # noqa: F401  (re-export)


def dice(pred: np.ndarray, label: np.ndarray) -> float:
    """2|P ∩ L| / (|P| + |L|). NaN when the label is empty."""
    pred, label = np.asarray(pred).astype(bool), np.asarray(label).astype(bool)
    if pred.shape != label.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs label {label.shape}")
    if not label.any():
        return float("nan")
    return float(2.0 * (pred & label).sum() / (pred.sum() + label.sum()))


def surface_dice(pred: np.ndarray, label: np.ndarray, spacing: Sequence[float], tolerance_mm: float) -> float:
    """Fraction of surface (of both masks) lying within `tolerance_mm` of the other's surface.

    (|{a in S_P : d(a, S_L) <= t}| + |{b in S_L : d(b, S_P) <= t}|) / (|S_P| + |S_L|)
    with distances from a Euclidean distance transform that honours the voxel spacing.
    """
    pred, label = np.asarray(pred).astype(bool), np.asarray(label).astype(bool)
    if pred.shape != label.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs label {label.shape}")
    if not label.any():
        return float("nan")
    if not pred.any():
        return 0.0
    sp, sl = surface_voxels(pred), surface_voxels(label)
    d_to_l = ndimage.distance_transform_edt(~sl, sampling=spacing)  # distance of every voxel to label surface
    d_to_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    good = (d_to_l[sp] <= tolerance_mm).sum() + (d_to_p[sl] <= tolerance_mm).sum()
    return float(good / (sp.sum() + sl.sum()))


def aggregated_dice(preds: Sequence[np.ndarray], labels: Sequence[np.ndarray]) -> float:
    """HECKTOR-style aggregated Dice for one label over a set of cases:
    2 * sum_i |P_i ∩ L_i| / sum_i (|P_i| + |L_i|).

    Sums are taken over cases BEFORE the ratio, so a case where the label is absent does not produce an
    undefined term: its false-positive voxels simply add to the denominator. NaN only when every prediction
    and every label is empty.
    """
    if len(preds) != len(labels):
        raise ValueError(f"{len(preds)} predictions vs {len(labels)} labels")
    inter, pred_n, label_n = [], [], []
    for p, t in zip(preds, labels):
        p, t = np.asarray(p).astype(bool), np.asarray(t).astype(bool)
        if p.shape != t.shape:
            raise ValueError(f"shape mismatch: pred {p.shape} vs label {t.shape}")
        inter.append(int((p & t).sum()))
        pred_n.append(int(p.sum()))
        label_n.append(int(t.sum()))
    return aggregated_dice_from_counts(inter, pred_n, label_n)


def aggregated_dice_from_counts(inter, pred_n, label_n) -> float:
    """The same aggregated Dice from per-case voxel counts (|P and L|, |P|, |L|). The test script stores these
    three integers per case instead of 123 full-size masks, and the bootstrap over cases resamples them."""
    inter, total = int(np.sum(inter)), int(np.sum(pred_n)) + int(np.sum(label_n))
    return float(2.0 * inter / total) if total else float("nan")
