"""Shared, strictly read-only helpers for the exploration dashboard.

Nothing in frontend/ may write to data/, configs/ or results/, or start
training. That is why this module calls the dataset adapters' own
training_samples() (which only reads dataset.json / lists files) rather than
src.data.dataset.build_samples(): build_samples creates the split and
conditioning-stats files when they are missing, and that write must only ever
come from the pipeline itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.adapters import DecathlonAdapter, HecktorAdapter  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
CONFIGS_DIR = REPO_ROOT / "configs"

# Same registry as src/data/dataset.py, duplicated rather than imported because
# importing dataset.py pulls in MONAI; the dashboard only needs the adapters.
ADAPTERS = {"decathlon": DecathlonAdapter, "hecktor": HecktorAdapter}


def spacing_result_dirs() -> list[Path]:
    """Every results/ subfolder holding a spacing_analysis.py manifest, so a
    HECKTOR run's output shows up next to Task06_Lung's without code changes."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(p.parent for p in RESULTS_DIR.glob("*/spacing_manifest.csv"))


@st.cache_data
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def data_configs() -> dict[str, dict]:
    """configs/data_*.yaml files, keyed by filename."""
    out = {}
    for p in sorted(CONFIGS_DIR.glob("data_*.yaml")):
        with open(p) as f:
            out[p.name] = yaml.safe_load(f)
    return out


@st.cache_data
def load_samples(dataset: str, data_dir: str) -> list[dict]:
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    return ADAPTERS[dataset](data_path).training_samples()


def load_split_membership(config: dict) -> dict[str, str]:
    """patient_id -> "train"/"val"/"test", or {} if the split file isn't there
    yet. Read only -- never generated here."""
    path = REPO_ROOT / config.get("split", {}).get("path", "")
    if not path.is_file():
        return {}
    with open(path) as f:
        split = json.load(f)
    return {pid: name for name, pids in split.items() for pid in pids}


def _compact(arr: np.ndarray) -> np.ndarray:
    """Smallest integer dtype that holds arr exactly, else float32."""
    if np.issubdtype(arr.dtype, np.floating):
        if not np.array_equal(arr, np.round(arr)):
            return arr.astype(np.float32, copy=False)
    lo, hi = arr.min(), arr.max()
    for dt in (np.uint8, np.int16, np.int32):
        if np.iinfo(dt).min <= lo and hi <= np.iinfo(dt).max:
            return arr.astype(dt, copy=False)
    return arr.astype(np.float32, copy=False)


@st.cache_data(max_entries=3, show_spinner="Loading volume from disk...")
def load_volume(path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Returns (array, spacing_mm) reoriented to RAS+ by
    nib.as_closest_canonical, which only permutes/flips axes (no
    interpolation) -- the same thing Orientationd does in the training
    transforms, so what is shown here is the voxel grid the model sees before
    any baseline resampling.

    Memory matters on an 8GB machine: nibabel applies the header's scl_slope,
    so Task06_Lung CT comes back as float32 (~300MB for 512x512x288) even
    though every value is an integer HU. Integer-valued arrays are therefore
    downcast losslessly (int16 for CT, uint8 for labels); genuinely fractional
    data such as PET SUV stays float32. max_entries caps the cache at three
    arrays (an image and its label count separately).
    """
    img = nib.as_closest_canonical(nib.load(path))
    axcodes = nib.aff2axcodes(img.affine)
    if axcodes != ("R", "A", "S"):
        raise ValueError(f"{path}: expected RAS after canonicalisation, got {axcodes}")
    arr = np.asanyarray(img.dataobj)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected a 3D volume, got shape {arr.shape}")
    arr = _compact(arr)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return arr, spacing


# Radiological display convention for a RAS+ array (axis 0 -> patient Right,
# 1 -> Anterior, 2 -> Superior): patient right on screen left, anterior/superior
# at the top. For every view that is the in-plane slice transposed and flipped
# on both axes, drawn with imshow's default origin="upper".
VIEWS = {
    #            fixed axis, (row axis, col axis), edge labels (left, right, top, bottom)
    "Axial": (2, (1, 0), ("R", "L", "A", "P")),
    "Coronal": (1, (2, 0), ("R", "L", "S", "I")),
    "Sagittal": (0, (2, 1), ("A", "P", "S", "I")),
}


def oriented_slice(
    vol: np.ndarray, spacing: tuple[float, float, float], view: str, index: int
) -> tuple[np.ndarray, float, tuple[str, str, str, str]]:
    """Returns (2D slice ready for imshow(origin="upper"), aspect, edge labels).
    aspect = row spacing / column spacing, so anisotropic voxels are drawn at
    their true physical proportions instead of being squashed."""
    fixed, (row_ax, col_ax), labels = VIEWS[view]
    s = np.take(vol, index, axis=fixed)  # remaining axes keep their order
    s = s.T[::-1, ::-1]
    return s, spacing[row_ax] / spacing[col_ax], labels
