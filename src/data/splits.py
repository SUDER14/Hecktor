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


TINY_CENTRES = ("HMR", "USZ")  # too few patients for a test set on their own


def check_test_centres(samples: list[dict], test_centres, tiny_centres=TINY_CENTRES) -> None:
    """A test set made only of tiny centres gives per-centre statistics too noisy to
    interpret, so refuse it; tiny centres may be added next to a larger one."""
    tc = {c.upper() for c in test_centres}
    if tc and tc <= {c.upper() for c in tiny_centres}:
        raise ValueError(f"test_centres {sorted(tc)} are all tiny centres {sorted(tiny_centres)}; "
                         "add at least one larger centre to the held-out set")


def group_split_by_centre(samples: list[dict], test_centres: list[str],
                           val_frac: float, seed: int) -> dict[str, list[str]]:
    """Held-out-centre split: `test_centres` are entirely excluded from
    train/val (the held-out-centre protocol), and the remaining centres are
    further split into train/val by centre (GroupShuffleSplit), so a centre
    never straddles train and val either.
    """
    check_test_centres(samples, test_centres)
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


def save_split(split: dict[str, list[str]], path: str | Path, meta: dict | None = None) -> None:
    """`meta` (e.g. seed, strategy, test centres) is stored under "_meta"; apply_split ignores it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({**split, **({"_meta": meta} if meta else {})}, f, indent=2)


def load_split(path: str | Path) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


def apply_split(samples: list[dict], split: dict[str, list[str]]) -> dict[str, list[dict]]:
    by_id = {s["patient_id"]: s for s in samples}
    out = {}
    for subset, ids in split.items():
        if subset.startswith("_"):  # metadata such as "_meta" (seed, strategy), not a subset
            continue
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(f"split references patient_ids not in samples: {missing}")
        out[subset] = [by_id[i] for i in ids]
    return out


def stratified_val_split(samples: list[dict], test_centres: list[str], val_frac: float,
                         seed: int) -> dict[str, list[str]]:
    """Held-out TEST centres (whole centres, never split), then a seeded random `val_frac` of the
    remaining cases as validation, stratified by centre (each centre contributes round(val_frac * n),
    at least one if the centre has 2+ cases). No patient appears in two subsets; validation shares
    centres with training by design (it tracks overfitting, the test centres measure generalisation)."""
    check_test_centres(samples, test_centres)
    tc = {c.upper() for c in test_centres}
    unknown = tc - {s["centre"].upper() for s in samples}
    if unknown:
        raise ValueError(f"test_centres not present in samples: {sorted(unknown)}")
    test_ids = sorted(s["patient_id"] for s in samples if s["centre"].upper() in tc)
    rng = np.random.RandomState(seed)
    train_ids, val_ids = [], []
    by_centre: dict[str, list[str]] = {}
    for s in samples:
        if s["centre"].upper() not in tc:
            by_centre.setdefault(s["centre"], []).append(s["patient_id"])
    for centre in sorted(by_centre):
        ids = sorted(by_centre[centre])
        n_val = int(round(val_frac * len(ids))) if val_frac > 0 else 0
        n_val = max(n_val, 1) if (val_frac > 0 and len(ids) >= 2) else n_val
        picked = set(rng.permutation(ids)[:n_val].tolist())
        val_ids += [i for i in ids if i in picked]
        train_ids += [i for i in ids if i not in picked]
    return {"train": sorted(train_ids), "val": sorted(val_ids), "test": test_ids}
