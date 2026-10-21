"""Cache build + cached datasets, on a small synthetic anisotropic NIfTI (no real data needed)."""
import nibabel as nib
import numpy as np
import pytest
import torch
from monai.transforms import Compose

from src.data.cache import CachedCropDataset, CachedVolumeDataset, build_cache_entry, entry_is_valid, preprocess_signature
from src.data.transforms import build_baseline_load_transforms, build_load_transforms
from src.training.train import build_train_loader, train_one_epoch

CFG = {
    "seed": 7,
    "patch": {"spatial_size": [8, 8, 8], "pos": 2, "neg": 1, "num_samples": 3},
    "intensity": {"ct_window": {"min": -1000, "max": 400}, "pet": {"suv_clip_max": 20.0}},
    "baseline": {"resample_spacing": [2.0, 2.0, 2.0]},
}
SPACING = (1.0, 1.0, 2.5)


@pytest.fixture()
def item(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.integers(-900, 300, (24, 24, 10)).astype(np.int16)
    lab = np.zeros(img.shape, np.uint8)
    lab[10:16, 10:16, 3:7] = 1
    aff = np.diag([*SPACING, 1.0])
    nib.save(nib.Nifti1Image(img, aff), str(tmp_path / "img.nii.gz"))
    nib.save(nib.Nifti1Image(lab, aff), str(tmp_path / "lab.nii.gz"))
    return {"image": str(tmp_path / "img.nii.gz"), "label": str(tmp_path / "lab.nii.gz"),
            "patient_id": "p1", "modality": "CT"}


def _reference(item, native=False):
    stages = (build_load_transforms if native else build_baseline_load_transforms)("CT", CFG)
    return Compose(stages)({"image": item["image"], "label": item["label"]})


def test_cache_matches_pipeline_and_stores_original_spacing(item, tmp_path):
    meta = build_cache_entry(item, CFG, tmp_path / "c", image_dtype="float32", seed=7)
    ref = _reference(item)
    cached = np.load(tmp_path / "c" / "p1__CT.image.npy")
    np.testing.assert_array_equal(cached, ref["image"].numpy()[0])
    np.testing.assert_array_equal(np.load(tmp_path / "c" / "p1__CT.label.npy"), np.rint(ref["label"].numpy()[0]))
    assert meta["original_spacing"] == pytest.approx(SPACING)
    assert meta["cache_spacing"] == pytest.approx([2.0, 2.0, 2.0])


def test_native_cache_keeps_grid_and_spacing(item, tmp_path):
    meta = build_cache_entry(item, CFG, tmp_path / "c", native=True, image_dtype="float32")
    assert meta["shape"] == [24, 24, 10]
    assert meta["cache_spacing"] == pytest.approx(SPACING)
    assert meta["original_spacing"] == pytest.approx(SPACING)


def test_float16_error_is_within_documented_step(item, tmp_path):
    build_cache_entry(item, CFG, tmp_path / "c", image_dtype="float16")
    ref = _reference(item)["image"].numpy()[0]
    err = np.abs(np.load(tmp_path / "c" / "p1__CT.image.npy").astype(np.float32) - ref).max()
    assert err <= 2 ** -11 / 1.0 + 1e-9   # half-step of fp16 spacing near 1.0 is 2^-12; 2^-11 is a safe bound


def test_crops_are_slices_of_the_cached_volume_and_deterministic(item, tmp_path):
    build_cache_entry(item, CFG, tmp_path / "c", image_dtype="float32")
    ds = CachedCropDataset([item], tmp_path / "c", CFG)
    full = np.load(tmp_path / "c" / "p1__CT.image.npy")
    crops = ds[0]
    assert len(crops) == 3
    for c in crops:
        z = c["crop_start"].tolist()
        assert c["image"].shape == (1, 8, 8, 8) and c["label"].shape == (1, 8, 8, 8)
        np.testing.assert_array_equal(c["image"][0].numpy(), full[z[0]:z[0] + 8, z[1]:z[1] + 8, z[2]:z[2] + 8])
    again = ds[0]
    assert all(torch.equal(a["image"], b["image"]) for a, b in zip(crops, again))
    ds.set_epoch(1)
    assert not all(torch.equal(a["crop_start"], b["crop_start"]) for a, b in zip(crops, ds[0]))


def test_positive_crops_contain_foreground(item, tmp_path):
    build_cache_entry(item, CFG, tmp_path / "c", image_dtype="float32")
    cfg = {**CFG, "patch": {**CFG["patch"], "pos": 1, "neg": 0, "num_samples": 6}}
    ds = CachedCropDataset([item], tmp_path / "c", cfg)
    assert all(c["label"].sum() > 0 for c in ds[0])


def test_volume_dataset_returns_full_float32(item, tmp_path):
    build_cache_entry(item, CFG, tmp_path / "c", image_dtype="float16")
    out = CachedVolumeDataset([item], tmp_path / "c", CFG)[0]
    assert out["image"].dtype == torch.float32 and out["image"].shape == (1, 12, 12, 12) and out["label"].shape == out["image"].shape


def test_stale_or_missing_cache_is_rejected(item, tmp_path):
    with pytest.raises(FileNotFoundError):
        CachedVolumeDataset([item], tmp_path / "empty", CFG)
    build_cache_entry(item, CFG, tmp_path / "c")
    stale = {**CFG, "baseline": {"resample_spacing": [1.0, 1.0, 1.0]}}
    with pytest.raises(ValueError, match="rebuild"):
        CachedVolumeDataset([item], tmp_path / "c", stale)
    assert entry_is_valid(tmp_path / "c", item, preprocess_signature(CFG, "CT", False), "float16")
    assert not entry_is_valid(tmp_path / "c", item, preprocess_signature(stale, "CT", False))


def test_loader_workers_and_timing_breakdown(item, tmp_path):
    from monai.losses import DiceCELoss
    from monai.networks.nets import UNet
    from torch.amp import GradScaler

    build_cache_entry(item, CFG, tmp_path / "c", image_dtype="float32")
    ds = CachedCropDataset([item], tmp_path / "c", CFG)
    cfg = {**CFG, "train": {"loader_batch_size": 1, "micro_batch_size": 3, "accum_steps": 1,
                            "num_workers": "auto", "amp": False}}
    loader = build_train_loader(ds, cfg, torch.device("cpu"))
    import os
    assert loader.num_workers == (os.cpu_count() or 0)
    assert loader.persistent_workers == (loader.num_workers > 0)
    model = UNet(3, 1, 2, (4, 8), (2,), num_res_units=1)
    timing: dict = {}
    loss = train_one_epoch(model, loader, DiceCELoss(to_onehot_y=True, softmax=True),
                           torch.optim.AdamW(model.parameters()), GradScaler(device="cpu", enabled=False),
                           torch.device("cpu"), cfg, timing)
    assert np.isfinite(loss)
    assert set(timing) == {"data_s", "gpu_s", "total_s"}
    assert timing["data_s"] >= 0 and timing["gpu_s"] > 0
    assert timing["data_s"] + timing["gpu_s"] <= timing["total_s"] + 1e-3
