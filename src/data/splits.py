"""Train/val/test split construction, written to disk as JSON for reproducibility.

Two strategies, chosen per-dataset (see CLAUDE.md):
  - Decathlon: plain random split (no centre metadata exists to group by).
  - HECKTOR: group split by clinical centre, because the held-out-centre
    protocol -- can the model generalise to a centre's acquisition geometry
    it has never seen -- is the central experiment, not an afterthought.

A split is represented purely as patient_id lists, never re-derived from
directory order, so it survives filesystem/OS differences.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def random_split(samples: list[dict], val_frac: float, test_frac: float,
                  seed: int) -> dict[str, list[str]]:
    if val_frac + test_frac >= 1.0:
        raise ValueError(f"val_frac + test_frac must be < 1, got {val_frac + test_frac}")

    patient_ids = sorted(s["patient_id"] for s in samples)
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(patient_ids)

    n = len(shuffled)
    n_val = round(n * val_frac)
    n_test = round(n * test_frac)

    return {
        "val": sorted(shuffled[:n_val].tolist()),
        "test": sorted(shuffled[n_val:n_val + n_test].tolist()),
        "train": sorted(shuffled[n_val + n_test:].tolist()),
    }


def group_split_by_centre(samples: list[dict], test_centres: list[str],
                           val_frac: float, seed: int) -> dict[str, list[str]]:
    """Held-out-centre split: `test_centres` are entirely excluded from
    train/val (the held-out-centre protocol), and the remaining centres are
    further split into train/val by centre (GroupShuffleSplit), so a centre
    never straddles train and val either.
    """
    test_centres = set(test_centres)
    unknown = test_centres - {s["centre"] for s in samples}
    if unknown:
        raise ValueError(f"test_centres not present in samples: {unknown}")
    if "UNKNOWN" in {s["centre"] for s in samples}:
        raise ValueError(
            "found samples with centre='UNKNOWN' -- group_split_by_centre "
            "requires every sample to carry a real centre code (this is what "
            "the Decathlon adapter reports, since Decathlon filenames don't "
            "encode a centre -- use random_split for that dataset instead)."
        )

    test_ids = sorted(s["patient_id"] for s in samples if s["centre"] in test_centres)
    remaining = [s for s in samples if s["centre"] not in test_centres]
    if not remaining:
        raise ValueError("test_centres consumed every sample; nothing left for train/val")

    remaining_ids = np.array([s["patient_id"] for s in remaining])
    remaining_groups = np.array([s["centre"] for s in remaining])

    if val_frac <= 0:
        train_ids, val_ids = remaining_ids.tolist(), []
    else:
        n_groups = len(set(remaining_groups))
        if n_groups < 2:
            raise ValueError(
                f"only {n_groups} centre(s) left after holding out {test_centres}; "
                "cannot carve out a group-disjoint validation split from one centre"
            )
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        train_idx, val_idx = next(splitter.split(remaining_ids, groups=remaining_groups))
        train_ids = remaining_ids[train_idx].tolist()
        val_ids = remaining_ids[val_idx].tolist()

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": test_ids,
    }


def save_split(split: dict[str, list[str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: str | Path) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


def apply_split(samples: list[dict], split: dict[str, list[str]]) -> dict[str, list[dict]]:
    by_id = {s["patient_id"]: s for s in samples}
    out = {}
    for subset, ids in split.items():
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(f"split references patient_ids not in samples: {missing}")
        out[subset] = [by_id[i] for i in ids]
    return out
