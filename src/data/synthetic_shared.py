"""Synthetic PET/CT cohort in the HECKTOR 2026 on-disk layout (CT, PET, label on ONE grid).

Why: unit tests must not depend on the 144 GB archive. The real data has CT, PET and
label on one shared grid (VERIFIED, 782 cases), so each synthetic case does too: a
body-shaped CT, a blurry PET uptake at the same place, and an ellipsoid label, all
agreeing about where in millimetres the blob is. Different "centres" use different
spacings (including thick-slice ones), which is the property the model is conditioned on.

Layout written (root = hecktor2026_training):
    <CENTRE-NNN>/<CENTRE-NNN>__CT.nii.gz, <CENTRE-NNN>__PT.nii.gz, <CENTRE-NNN>.nii.gz
    HECKTOR_2026_training_data.csv   (PatientID, CenterID; synthetic stand-in)
Labels are {0, 1, 2}: label 1 = a larger blob, label 2 = a smaller blob (only the values matter here).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

# spacing (mm) per fake centre; includes thick-slice variants.
DEFAULT_CENTRES: dict[str, tuple[float, float, float]] = {
    "CHUM": (1.0, 1.0, 1.0),
    "CHUP": (1.0, 1.0, 2.0),
    "MDA": (2.0, 2.0, 2.0),
    "HGJ": (1.0, 1.0, 4.0),
    "HMR": (2.0, 2.0, 4.0),
    "USZ": (1.0, 1.0, 3.0),
}


@dataclass
class SynthConfig:
    fov_mm: tuple[float, float, float] = (96.0, 96.0, 96.0)
    n_per_centre: int = 4
    seed: int = 0
    centres: dict | None = None
    tumour_radius_mm: tuple[float, float] = (6.0, 14.0)
    tumour_prob: float = 0.85  # some cases are tumour-free, as in a real cohort


def _grid(fov_mm, spacing) -> tuple[int, int, int]:
    return tuple(int(round(f / s)) for f, s in zip(fov_mm, spacing))


def _affine(spacing, edge0) -> np.ndarray:
    a = np.eye(4)
    a[range(3), range(3)] = spacing
    a[:3, 3] = np.asarray(edge0) + 0.5 * np.asarray(spacing)
    return a


def _coords_mm(shape, spacing, edge0):
    axes = [edge0[i] + (np.arange(shape[i]) + 0.5) * spacing[i] for i in range(3)]
    return np.meshgrid(*axes, indexing="ij")


def make_case(rng: np.random.RandomState, sp, cfg: SynthConfig):
    """Returns (ct, pet, label, affine); all three on the same grid."""
    edge0 = rng.uniform(-200, 200, 3)  # arbitrary world offset: nothing may assume origin 0
    shape = _grid(cfg.fov_mm, sp)
    x, y, z = _coords_mm(shape, sp, edge0)
    fov = np.asarray(cfg.fov_mm)
    body_c = edge0 + fov / 2
    body = (((x - body_c[0]) / (0.45 * fov[0])) ** 2 + ((y - body_c[1]) / (0.40 * fov[1])) ** 2
            + ((z - body_c[2]) / (0.48 * fov[2])) ** 2) < 1
    ct = np.where(body, 40.0, -1000.0) + rng.normal(0, 15, shape)  # HU-like
    pet = body.astype(np.float64) + rng.normal(0, 0.1, shape)      # SUV-like
    label = np.zeros(shape, np.uint8)
    if rng.rand() < cfg.tumour_prob:
        for cls in range(1, rng.randint(1, 3) + 1):  # class 1, optionally class 2
            c = edge0 + fov * rng.uniform(0.3, 0.7, 3)
            r = rng.uniform(*cfg.tumour_radius_mm, 3) * (1.0 if cls == 1 else 0.6)  # ellipsoid radii, mm
            d = ((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2
            label[d < 1] = cls
            ct[d < 1] += 30.0              # faint CT contrast
            pet += 6.0 * np.exp(-d)        # strong, blurry PET uptake
    pet = np.clip(pet, 0, None)
    return ct.astype(np.float32), pet.astype(np.float32), label, _affine(sp, edge0)


def write_synthetic_hecktor2026(root: str | Path, cfg: SynthConfig | None = None) -> list[str]:
    """Writes the cohort; returns patient ids. Deterministic given cfg.seed."""
    cfg = cfg or SynthConfig()
    root = Path(root)
    rng = np.random.RandomState(cfg.seed)
    ids = []
    for centre, sp in (cfg.centres or DEFAULT_CENTRES).items():
        for k in range(1, cfg.n_per_centre + 1):
            pid = f"{centre}-{k:03d}"
            ct, pet, lab, aff = make_case(rng, sp, cfg)
            d = root / pid
            d.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(ct, aff), d / f"{pid}__CT.nii.gz")
            nib.save(nib.Nifti1Image(pet, aff), d / f"{pid}__PT.nii.gz")
            nib.save(nib.Nifti1Image(lab, aff), d / f"{pid}.nii.gz")
            ids.append(pid)
    with open(root / "HECKTOR_2026_training_data.csv", "w") as f:
        f.write("PatientID,CenterID\n")
        f.writelines(f"{p},{p.split('-')[0]}\n" for p in ids)
    return ids
