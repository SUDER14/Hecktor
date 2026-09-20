"""MONAI transform pipelines.

No resampling stage anywhere here (per CLAUDE.md, ACF processes volumes at
native resolution) -- Orientationd only reorders/flips array axes to match
the affine's own canonical direction, it never interpolates, so it doesn't
violate that constraint.

PET intensity handling assumes the volume is already SUV-normalised. This
assumption is UNVERIFIED against real data: Task06_Lung (the only Decathlon
data in this project so far) is CT-only -- no Decathlon task has a PET
channel at all -- so the PET branch below has never been exercised against a
real file. Confirm it against the first real HECKTOR PET volume before
trusting it (see conditioning.py's PET units note / the run report for the
SUV-vs-raw-activity question).
"""

from __future__ import annotations

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)


def _intensity_transform(modality: str, config: dict):
    if modality == "CT":
        w = config["intensity"]["ct_window"]
        return ScaleIntensityRanged(
            keys=["image"], a_min=w["min"], a_max=w["max"],
            b_min=0.0, b_max=1.0, clip=True,
        )
    if modality in ("PT", "PET"):
        # ASSUMES SUV units -- unverified, see module docstring.
        suv_max = config["intensity"]["pet"]["suv_clip_max"]
        return ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=suv_max,
            b_min=0.0, b_max=1.0, clip=True,
        )
    raise ValueError(f"no intensity transform defined for modality {modality!r}")


def build_load_transforms(modality: str, config: dict) -> list:
    """Shared prefix: load -> channel-first -> canonical orientation -> intensity.

    Single-modality only (image_key='image' holds one volume). Multi-modality
    (dual-encoder CT+PT) samples are loaded as separate single-modality dicts
    upstream and only combined at the model input stage -- keeping this
    module dataset/modality-count agnostic.
    """
    return [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        _intensity_transform(modality, config),
    ]


def build_train_transforms(modality: str, config: dict) -> Compose:
    patch = config["patch"]
    transforms = build_load_transforms(modality, config) + [
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            image_key="image",
            spatial_size=patch["spatial_size"],
            pos=patch["pos"],
            neg=patch["neg"],
            num_samples=patch["num_samples"],
            image_threshold=0,
        ),
    ]
    return Compose(transforms)


def build_val_transforms(modality: str, config: dict) -> Compose:
    """No cropping: validation runs on full native-resolution volumes
    (sliding-window inference happens in the training loop, not here)."""
    return Compose(build_load_transforms(modality, config))


# ----------------------------------------------------------------------------
# Baseline-only pipeline (Baseline 1, src/models/baselines.py)
# ----------------------------------------------------------------------------
# This is the ONLY place in this codebase that resamples volumes. It exists
# for the plain, non-spacing-conditioned 3D U-Net baseline -- the standard
# reference against which ACF's native-resolution, spacing-conditioned
# approach gets compared (see baselines.py's module docstring). Do not reuse
# these for ACF's own train/val pipeline: build_train_transforms /
# build_val_transforms above must stay resample-free per CLAUDE.md.


def build_baseline_load_transforms(modality: str, config: dict) -> list:
    """Shared prefix: load -> channel-first -> canonical orientation ->
    resample to isotropic spacing -> intensity."""
    pixdim = config["baseline"]["resample_spacing"]
    return [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        _intensity_transform(modality, config),
    ]


def build_baseline_train_transforms(modality: str, config: dict) -> Compose:
    patch = config["patch"]
    transforms = build_baseline_load_transforms(modality, config) + [
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            image_key="image",
            spatial_size=patch["spatial_size"],
            pos=patch["pos"],
            neg=patch["neg"],
            num_samples=patch["num_samples"],
            image_threshold=0,
        ),
    ]
    return Compose(transforms)


def build_baseline_val_transforms(modality: str, config: dict) -> Compose:
    """No cropping: validation runs on the full resampled volume (sliding-window
    inference happens in the training loop, not here)."""
    return Compose(build_baseline_load_transforms(modality, config))
