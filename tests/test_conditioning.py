"""Unit tests for src/data/conditioning.py, and an end-to-end check that a
sample's conditioning vector survives the real train-transform pipeline
unchanged (the "conditioning-vector passthrough" property)."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.data.adapters import DecathlonAdapter
from src.data.conditioning import (
    CONDITIONING_DIM,
    attach_conditioning_vectors,
    fit_conditioning_stats,
)
from src.data.transforms import build_train_transforms

REAL_DECATHLON_DIR = Path("data/Task06_Lung")


def _make_volume(path: Path, shape, spacing, fg_box=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.RandomState(0).uniform(-200, 200, size=shape).astype(np.float32)
    if fg_box is not None:
        x0, x1, y0, y1, z0, z1 = fg_box
        data[x0:x1, y0:y1, z0:z1] = 300.0  # unambiguously inside the CT window
    affine = np.diag(list(spacing) + [1.0])
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _make_label(path: Path, shape, spacing, fg_box):
    path.parent.mkdir(parents=True, exist_ok=True)
    label = np.zeros(shape, dtype=np.uint8)
    x0, x1, y0, y1, z0, z1 = fg_box
    label[x0:x1, y0:y1, z0:z1] = 1
    affine = np.diag(list(spacing) + [1.0])
    nib.save(nib.Nifti1Image(label, affine), str(path))


@pytest.mark.skipif(not REAL_DECATHLON_DIR.exists(), reason="Task06_Lung not downloaded")
def test_fit_and_attach_on_real_headers():
    adapter = DecathlonAdapter(REAL_DECATHLON_DIR)
    samples = adapter.training_samples()[:6]

    stats = fit_conditioning_stats(samples)
    assert "CT" in stats
    for feat in ("spacing_x", "spacing_y", "slice_thickness"):
        assert "mean" in stats["CT"][feat] and "std" in stats["CT"][feat]

    attached = attach_conditioning_vectors(samples, stats)
    for s in attached:
        vec = s["conditioning"]["CT"]
        assert vec.shape == (CONDITIONING_DIM,)
        assert vec.dtype == np.float32
        assert vec[3] == 1.0  # is_CT
        assert vec[4] == 0.0  # is_PT

    # Manually recompute one z-score and check it matches, rather than trusting
    # the function's own arithmetic.
    s0 = attached[0]
    sx_raw = s0["geometry"]["CT"]["spacing_x"]
    expected = (sx_raw - stats["CT"]["spacing_x"]["mean"]) / stats["CT"]["spacing_x"]["std"]
    assert s0["conditioning"]["CT"][0] == pytest.approx(expected, abs=1e-5)


def test_attach_raises_on_unseen_modality():
    samples = [{
        "patient_id": "p1", "centre": "UNKNOWN",
        "modalities": ["PT"], "image": ["fake.nii.gz"], "label": "fake_label.nii.gz",
    }]
    stats = {"CT": {"spacing_x": {"mean": 1.0, "std": 1.0},
                     "spacing_y": {"mean": 1.0, "std": 1.0},
                     "slice_thickness": {"mean": 1.0, "std": 1.0}}}
    with pytest.raises(KeyError):
        attach_conditioning_vectors(samples, stats)


def test_conditioning_vector_survives_train_transform_pipeline(tmp_path):
    """The crop transform must not touch, drop, or resize keys it wasn't told
    about -- cond_vector has to come out of RandCropByPosNegLabeld identical
    to what went in, for every crop it produces."""
    shape = (24, 24, 16)
    spacing = (1.0, 1.0, 2.0)
    fg_box = (8, 16, 8, 16, 4, 8)

    image_path = tmp_path / "img.nii.gz"
    label_path = tmp_path / "lbl.nii.gz"
    _make_volume(image_path, shape, spacing, fg_box=fg_box)
    _make_label(label_path, shape, spacing, fg_box)

    cond_vector = np.array([0.1, 0.1, -0.2, 1.0, 0.0], dtype=np.float32)
    sample = {
        "image": str(image_path),
        "label": str(label_path),
        "patient_id": "synthetic-001",
        "centre": "UNKNOWN",
        "modality": "CT",
        "cond_vector": cond_vector,
        "geometry": {"spacing_x": 1.0, "spacing_y": 1.0, "slice_thickness": 2.0},
    }

    config = {
        "intensity": {"ct_window": {"min": -1000, "max": 400}, "pet": {"suv_clip_max": 20.0}},
        "patch": {"spatial_size": [8, 8, 8], "pos": 1, "neg": 0, "num_samples": 3},
    }
    transform = build_train_transforms("CT", config)
    crops = transform(sample)

    assert len(crops) == config["patch"]["num_samples"]
    for crop in crops:
        assert np.array_equal(crop["cond_vector"], cond_vector)
        assert crop["patient_id"] == "synthetic-001"
        assert crop["geometry"]["spacing_x"] == 1.0
        assert tuple(crop["image"].shape[-3:]) == (8, 8, 8)
        # pos=1, neg=0 -> every crop must be centred on foreground
        assert crop["label"].sum() > 0
