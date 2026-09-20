"""Wires adapters + splits + conditioning + transforms into MONAI datasets.

IMPORTANT LIMITATION -- read before pointing this at HECKTOR:
Each sample dict produced here is a single (patient, modality) pair, e.g. one
dict for a patient's CT and a separate dict for their PT. That is correct and
sufficient for Decathlon, which is single-modality by construction.

For HECKTOR's dual-encoder case, CT and PT must eventually be presented to
the model *together* per patient, cropped around the same physical location
-- but at native resolution their voxel grids differ (the ~4-5x PET/CT
resolution asymmetry CLAUDE.md calls out as an open problem), so MONAI's
RandCropByPosNegLabeld cannot be handed both image keys at once: it assumes
every key it crops shares one voxel grid/shape. Synchronising a physically-
aligned crop across two different native grids without resampling is an open
part of the ACF design, not something solved here. Do not wire a dual-input
training loop against this module's per-modality Dataset until that's
designed -- it will silently crop CT and PT at mismatched physical locations.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from monai.data import Dataset as MonaiDataset

from src.data.adapters import DecathlonAdapter, HecktorAdapter, SegmentationDataset
from src.data.conditioning import (
    attach_conditioning_vectors,
    fit_conditioning_stats,
    load_stats,
    save_stats,
)
from src.data.splits import (
    apply_split,
    group_split_by_centre,
    load_split,
    random_split,
    save_split,
)
from src.data.transforms import build_train_transforms, build_val_transforms

ADAPTERS: dict[str, type[SegmentationDataset]] = {
    "decathlon": DecathlonAdapter,
    "hecktor": HecktorAdapter,
}


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_or_load_split(samples: list[dict], config: dict) -> dict[str, list[str]]:
    split_path = Path(config["split"]["path"])
    if split_path.exists():
        return load_split(split_path)

    strategy = config["split"]["strategy"]
    if strategy == "random":
        split = random_split(
            samples, config["split"]["val_frac"], config["split"]["test_frac"],
            config["seed"],
        )
    elif strategy == "group_by_centre":
        split = group_split_by_centre(
            samples, config["split"]["test_centres"], config["split"]["val_frac"],
            config["seed"],
        )
    else:
        raise ValueError(f"unknown split strategy {strategy!r}")

    save_split(split, split_path)
    return split


def _build_or_load_conditioning_stats(train_samples: list[dict], config: dict) -> dict:
    stats_path = Path(config["conditioning"]["stats_path"])
    if stats_path.exists():
        return load_stats(stats_path)
    stats = fit_conditioning_stats(train_samples)
    save_stats(stats, stats_path)
    return stats


def build_samples(config: dict) -> dict[str, list[dict]]:
    """Returns {"train": [...], "val": [...], "test": [...]}, patient-level
    sample dicts each carrying "conditioning" (fit on train, reused unchanged
    at val/test -- see conditioning.py)."""
    adapter_cls = ADAPTERS[config["dataset"]]
    adapter = adapter_cls(config["data_dir"])
    samples = adapter.training_samples()

    split = _build_or_load_split(samples, config)
    by_split = apply_split(samples, split)

    stats = _build_or_load_conditioning_stats(by_split["train"], config)
    return {name: attach_conditioning_vectors(subset, stats) for name, subset in by_split.items()}


def _expand_to_modality_items(samples: list[dict]) -> list[dict]:
    """One MONAI-ready dict per (patient, modality) pair.

    See module docstring: this flattening is what makes the per-modality
    transforms in transforms.py possible, and is also exactly why this
    module does not yet support joint dual-modality (CT+PT) sampling.
    """
    items = []
    for s in samples:
        for modality, image_path in zip(s["modalities"], s["image"]):
            items.append(
                {
                    "image": image_path,
                    "label": s["label"],
                    "patient_id": s["patient_id"],
                    "centre": s["centre"],
                    "modality": modality,
                    "cond_vector": s["conditioning"][modality],
                    "geometry": s["geometry"][modality],
                }
            )
    return items


def build_monai_datasets(
    config: dict,
    train_transform_fn=build_train_transforms,
    val_transform_fn=build_val_transforms,
) -> dict[str, MonaiDataset]:
    """Only meaningful for single-modality data (Decathlon) today -- see the
    module docstring for why HECKTOR's dual-modality case isn't handled yet.
    Raises if a split mixes more than one modality, since silently building a
    dataset that ignores that would be exactly the kind of silent breakage
    CLAUDE.md asks not to produce.

    train_transform_fn/val_transform_fn default to ACF's own resample-free
    pipeline (transforms.py). Baseline 1 (src/models/baselines.py) passes
    transforms.build_baseline_train_transforms/build_baseline_val_transforms
    instead, so it can reuse this same adapter/split/conditioning plumbing
    while getting the resampled preprocessing path a plain, non-spacing-
    conditioned U-Net needs. Every other stage (splits, conditioning stats)
    is identical either way.
    """
    all_samples = build_samples(config)
    modalities = {m for subset in all_samples.values() for s in subset for m in s["modalities"]}
    if len(modalities) > 1:
        raise NotImplementedError(
            f"build_monai_datasets() only supports single-modality data; found "
            f"{modalities}. Multi-modality (HECKTOR) joint sampling needs the "
            "resolution-asymmetric crop-alignment design called out in this "
            "module's docstring before it can be wired up."
        )
    modality = next(iter(modalities))

    datasets = {}
    for split_name, samples in all_samples.items():
        items = _expand_to_modality_items(samples)
        transform = (
            train_transform_fn(modality, config)
            if split_name == "train"
            else val_transform_fn(modality, config)
        )
        datasets[split_name] = MonaiDataset(data=items, transform=transform)
    return datasets
