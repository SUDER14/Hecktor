"""Synthetic PET/CT cohort in the HECKTOR 2026 on-disk layout.

Why: HECKTOR access is pending, and the thing that must be tested is geometry --
a CT on a 1 mm grid and a PET on a 4 mm grid that cover the same physical field
of view. Real tumours are not needed for that; blobs are enough, as long as the
label, CT and PET agree about *where in millimetres* the blob is.

Alignment convention (matches src/data/physical_crop.py): voxel i of an axis is
centred at origin + i*spacing, leading edge at origin + (i-0.5)*spacing. Both
volumes of a patient start at the same world edge `edge0`, so
origin = edge0 + 0.5*spacing. Different "centres" use different spacings, which
is the property the model is conditioned on.

Layout written (root = hecktor2026_training):
    <CENTRE-NNN>/<CENTRE-NNN>__CT.nii.gz, <CENTRE-NNN>__PT.nii.gz, <CENTRE-NNN>.nii.gz
    HECKTOR_2026_Training_Task.csv   (patient id + centre; synthetic stand-in)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

# (CT spacing mm, PET spacing mm) per fake centre. Includes 1x1x3 mm CT variants.
DEFAULT_CENTRES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "CHUM": ((1.0, 1.0, 1.0), (4.0, 4.0, 4.0)),
    "CHUP": ((1.0, 1.0, 3.0), (4.0, 4.0, 4.0)),
    "MDA": ((1.0, 1.0, 1.0), (4.0, 4.0, 3.0)),
    "HGJ": ((1.0, 1.0, 3.0), (4.0, 4.0, 3.0)),
    "HMR": ((1.0, 1.0, 1.0), (4.0, 4.0, 4.0)),
    "USZ": ((1.0, 1.0, 3.0), (4.0, 4.0, 4.0)),
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


def make_case(rng: np.random.RandomState, ct_sp, pt_sp, cfg: SynthConfig):
    """Returns (ct, pet, label, ct_affine, pet_affine). Label lives on the CT grid."""
    edge0 = rng.uniform(-200, 200, 3)  # arbitrary world offset: nothing may assume origin 0
    ct_shape, pt_shape = _grid(cfg.fov_mm, ct_sp), _grid(cfg.fov_mm, pt_sp)
    cx, cy, cz = _coords_mm(ct_shape, ct_sp, edge0)
    px, py, pz = _coords_mm(pt_shape, pt_sp, edge0)
    fov = np.asarray(cfg.fov_mm)

    body_c = edge0 + fov / 2

    def body_mask(x, y, z):
        return (((x - body_c[0]) / (0.45 * fov[0])) ** 2 + ((y - body_c[1]) / (0.40 * fov[1])) ** 2
                + ((z - body_c[2]) / (0.48 * fov[2])) ** 2) < 1

    ct = np.where(body_mask(cx, cy, cz), 40.0, -1000.0) + rng.normal(0, 15, ct_shape)  # HU-like
    pet = body_mask(px, py, pz).astype(np.float64) + rng.normal(0, 0.1, pt_shape)      # SUV-like
    label = np.zeros(ct_shape, np.uint8)
    if rng.rand() < cfg.tumour_prob:
        for _ in range(rng.randint(1, 3)):
            c = edge0 + fov * rng.uniform(0.3, 0.7, 3)
            r = rng.uniform(*cfg.tumour_radius_mm, 3)  # ellipsoid radii in mm
            m_ct = (((cx - c[0]) / r[0]) ** 2 + ((cy - c[1]) / r[1]) ** 2 + ((cz - c[2]) / r[2]) ** 2) < 1
            d_pt = ((px - c[0]) / r[0]) ** 2 + ((py - c[1]) / r[1]) ** 2 + ((pz - c[2]) / r[2]) ** 2
            label[m_ct] = 1
            ct[m_ct] += 30.0                  # faint CT contrast
            pet += 6.0 * np.exp(-d_pt)        # strong, blurry PET uptake
    pet = np.clip(pet, 0, None)
    return (ct.astype(np.float32), pet.astype(np.float32), label,
            _affine(ct_sp, edge0), _affine(pt_sp, edge0))


def write_synthetic_hecktor2026(root: str | Path, cfg: SynthConfig | None = None) -> list[str]:
    """Writes the cohort; returns patient ids. Deterministic given cfg.seed."""
    cfg = cfg or SynthConfig()
    root = Path(root)
    rng = np.random.RandomState(cfg.seed)
    ids = []
    for centre, (ct_sp, pt_sp) in (cfg.centres or DEFAULT_CENTRES).items():
        for k in range(1, cfg.n_per_centre + 1):
            pid = f"{centre}-{k:03d}"
            ct, pet, lab, a_ct, a_pt = make_case(rng, ct_sp, pt_sp, cfg)
            d = root / pid
            d.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(ct, a_ct), d / f"{pid}__CT.nii.gz")
            nib.save(nib.Nifti1Image(pet, a_pt), d / f"{pid}__PT.nii.gz")
            nib.save(nib.Nifti1Image(lab, a_ct), d / f"{pid}.nii.gz")
            ids.append(pid)
    with open(root / "HECKTOR_2026_Training_Task.csv", "w") as f:
        f.write("PatientID,Centre\n")
        f.writelines(f"{p},{p.split('-')[0]}\n" for p in ids)
    return ids
