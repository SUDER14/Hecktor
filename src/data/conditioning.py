"""Per-volume conditioning vectors for the FiLM hypernetwork.

Reuses the exact header-reading routine from spacing_analysis.py -- the one
that was verified against real Decathlon NIfTI files (get_zooms() matches the
affine-derived spacing, axis 2 is through-plane) -- rather than re-deriving
spacing some other way here.

Normalisation statistics MUST be fit on the training split only and reused
unchanged at val/test (see CLAUDE.md). fit_conditioning_stats() is therefore
only ever called with the train sample list; attach_conditioning_vectors()
takes the already-fit stats dict as an argument for every split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metadata.spacing_analysis import COND_FEATURES, fit_norm_stats, read_volume_geometry

CONDITIONING_DIM = len(COND_FEATURES) + 2  # + is_CT, is_PT


def geometry_dataframe(samples: list[dict]) -> pd.DataFrame:
    """One row per (patient_id, modality) image, read straight from the header."""
    rows = []
    for s in samples:
        for modality, path in zip(s["modalities"], s["image"]):
            rec = read_volume_geometry(Path(path))
            if rec is None:
                raise ValueError(f"could not read header geometry for {path}")
            rec["patient_id"] = s["patient_id"]
            rec["centre"] = s["centre"]
            rec["modality"] = modality
            rows.append(rec)
    return pd.DataFrame(rows)


def fit_conditioning_stats(train_samples: list[dict]) -> dict:
    """Fit per-modality z-score stats on the training split only."""
    df = geometry_dataframe(train_samples)
    return fit_norm_stats(df)


def save_stats(stats: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def attach_conditioning_vectors(samples: list[dict], stats: dict) -> list[dict]:
    """Adds two keys to a copy of each sample dict:

        "conditioning": {modality: np.ndarray[float32] of shape (CONDITIONING_DIM,)}
            order: [spacing_x_norm, spacing_y_norm, slice_thickness_norm, is_CT, is_PT]
        "geometry": {modality: {"spacing_x", "spacing_y", "slice_thickness"}}  # raw, mm

    Raises rather than silently dropping if a sample's modality has no fitted
    stats -- that means val/test saw a modality the training split never did,
    which is a split-construction bug, not something to paper over.

    This check runs before any file is opened: it's a cheap dict lookup that
    should fail fast with a clear message, rather than getting masked by
    whatever unrelated I/O error comes out of reading a header we were about
    to reject anyway.
    """
    for s in samples:
        for modality in s["modalities"]:
            if modality not in stats:
                raise KeyError(
                    f"modality {modality!r} (patient {s['patient_id']}) has no "
                    "fitted conditioning stats. Stats must be fit on the "
                    "training split only -- check that this modality was "
                    "actually present in it."
                )

    df = geometry_dataframe(samples).set_index(["patient_id", "modality"])

    out = []
    for s in samples:
        s = dict(s)
        cond, geom = {}, {}
        for modality in s["modalities"]:
            row = df.loc[(s["patient_id"], modality)]
            m_stats = stats[modality]
            normed = [
                (row[feat] - m_stats[feat]["mean"]) / m_stats[feat]["std"]
                for feat in COND_FEATURES
            ]
            is_ct = 1.0 if modality == "CT" else 0.0
            is_pt = 1.0 if modality in ("PT", "PET") else 0.0
            cond[modality] = np.array(normed + [is_ct, is_pt], dtype=np.float32)
            geom[modality] = {feat: float(row[feat]) for feat in COND_FEATURES}
        s["conditioning"] = cond
        s["geometry"] = geom
        out.append(s)
    return out
