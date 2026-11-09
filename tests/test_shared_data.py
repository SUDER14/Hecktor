"""Phase 1: HECKTOR 2026 layout, conditioning, splits, shared-grid dual-channel crops -- synthetic data."""
import copy
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from src.data.conditioning import COND_DIM, make_cond
from src.data.shared_cache import DualCropDataset, DualVolumeDataset, build_dual_datasets
from src.data.hecktor2026 import Hecktor2026Adapter
from src.data.splits import check_test_centres, group_split_by_centre
from src.data.synthetic_shared import SynthConfig, write_synthetic_hecktor2026


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth") / "hecktor2026_training"
    ids = write_synthetic_hecktor2026(root, SynthConfig(fov_mm=(64, 64, 64), n_per_centre=3, seed=1))
    return root, ids


@pytest.fixture(scope="module")
def config(cohort, tmp_path_factory):
    root, _ = cohort
    work = tmp_path_factory.mktemp("work")
    cfg = yaml.safe_load(open("configs/synth_shared.yaml"))
    cfg["data_dir"] = str(root)
    cfg["split"]["path"] = str(work / "split.json")
    cfg["split"]["test_centres"] = ["MDA", "USZ"]
    cfg["conditioning"]["stats_path"] = str(work / "stats.json")
    cfg["cache"]["dir"] = str(work / "cache")
    cfg["patch"]["spatial_size"] = [32, 32, 32]
    return cfg


def test_adapter_reads_layout_and_centre(cohort):
    root, ids = cohort
    samples = Hecktor2026Adapter(root).training_samples()
    assert [s["patient_id"] for s in samples] == sorted(ids)
    assert {s["centre"] for s in samples} == {"CHUM", "CHUP", "MDA", "HGJ", "HMR", "USZ"}
    s = samples[0]
    assert s["modalities"] == ["CT", "PT"]
    assert s["image"][0].endswith("__CT.nii.gz") and s["image"][1].endswith("__PT.nii.gz")
    assert s["meta"]["CenterID"] == s["centre"]  # CSV attached


def test_adapter_missing_modality_is_an_error(cohort, tmp_path):
    root, _ = cohort
    dst = tmp_path / "d"
    shutil.copytree(root, dst)
    next(dst.glob("CHUM-001/*__PT.nii.gz")).unlink()
    with pytest.raises(FileNotFoundError, match="CHUM-001"):
        Hecktor2026Adapter(dst).training_samples()


def test_synthetic_ct_pet_label_share_one_grid_and_spacings_differ_by_centre(cohort):
    root, _ = cohort

    def geom(p):
        im = nib.load(p)
        return im.shape, tuple(round(float(v), 3) for v in im.header.get_zooms()[:3]), im.affine

    for pid, sp in (("CHUM-001", (1, 1, 1)), ("CHUP-001", (1, 1, 2)), ("HGJ-001", (1, 1, 4))):
        ct, pt, lab = (geom(root / pid / n) for n in (f"{pid}__CT.nii.gz", f"{pid}__PT.nii.gz", f"{pid}.nii.gz"))
        assert ct[1] == sp
        assert ct[0] == pt[0] == lab[0] and np.allclose(ct[2], pt[2]) and np.allclose(ct[2], lab[2])


def test_split_never_splits_a_patient_or_centre_and_tiny_only_test_rejected(cohort):
    samples = Hecktor2026Adapter(cohort[0]).training_samples()
    sp = group_split_by_centre(samples, ["MDA", "USZ"], 0.25, seed=3)
    sets = {k: set(v) for k, v in sp.items()}
    centre = {s["patient_id"]: s["centre"] for s in samples}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b])
            assert not ({centre[p] for p in sets[a]} & {centre[p] for p in sets[b]})
    with pytest.raises(ValueError, match="tiny"):
        check_test_centres(samples, ["HMR", "USZ"])


def test_cond3_and_train_only_stats_and_split_meta(config):
    ds = build_dual_datasets(config)
    stats = json.load(open(config["conditioning"]["stats_path"]))
    assert {"CT"} <= set(stats)
    # stats must be reproducible from the TRAIN patients only
    sz_ct = [nib.load(s["image"][0]).header.get_zooms()[2] for s in ds["train"].samples]
    assert stats["CT"]["slice_thickness"]["mean"] == pytest.approx(float(np.mean(sz_ct)), abs=1e-3)
    split = json.load(open(config["split"]["path"]))
    assert split["_meta"]["seed"] == config["seed"]
    v = ds["val"][0]
    assert v["cond"].shape == (COND_DIM,) == (3,)
    # the vector is exactly the z-scored spacing of that scan, with the train stats
    assert np.allclose(v["cond"].numpy(), make_cond(v["spacing"].numpy(), stats), atol=1e-5)
    z = (v["spacing"].numpy()[2] - stats["CT"]["slice_thickness"]["mean"]) / stats["CT"]["slice_thickness"]["std"]
    assert v["cond"][2].item() == pytest.approx(float(z), abs=1e-5)


def test_dual_crop_shapes_labels_and_spacing(config):
    ds = build_dual_datasets(config)
    tr: DualCropDataset = ds["train"]
    tr.set_epoch(1)
    for i in range(len(tr)):
        for item in tr[i]:
            assert item["image"].shape == (2, 32, 32, 32) and item["label"].shape == (1, 32, 32, 32)
            assert item["spacing"].shape == (3,) and item["cond"].shape == (3,)
            assert set(np.unique(item["label"].numpy())) <= {0.0, 1.0}   # K=2: any GTV is class 1
    v = ds["val"][0]  # val volumes: whole native volume, one grid
    assert isinstance(ds["val"], DualVolumeDataset) and v["image"].shape[0] == 2
    assert v["label"].shape[1:] == v["image"].shape[1:]


def test_label_map_k3_keeps_both_classes_and_k2_merges_them(config):
    c3 = copy.deepcopy(config)
    c3["model"]["out_channels"] = 3
    c3["cache"]["dir"] = config["cache"]["dir"]  # raw labels are cached, so the cache is reusable
    seen3, seen2 = set(), set()
    for cfg, seen in ((c3, seen3), (config, seen2)):
        ds = build_dual_datasets(cfg)["val"]
        for i in range(len(ds)):
            seen |= set(np.unique(ds[i]["label"].numpy()).tolist())
    assert seen2 <= {0.0, 1.0} and 2.0 in seen3 and 1.0 in seen3


def test_cache_holds_raw_labels_and_two_channel_image(config):
    build_dual_datasets(config)
    pid = "CHUM-001"
    cdir = Path(config["cache"]["dir"])
    img = np.load(cdir / f"{pid}.img.npy")
    lab = np.load(cdir / f"{pid}.label.npy")
    assert img.shape[0] == 2 and img.shape[1:] == lab.shape and lab.dtype == np.uint8
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    meta = json.loads((cdir / f"{pid}.meta.json").read_text())
    assert meta["spacing"] == pytest.approx([1.0, 1.0, 1.0]) and meta["orientation"] == "RAS"


def test_patch_with_absurd_physical_extent_is_rejected_at_dataset_construction(config):
    bad = copy.deepcopy(config)
    bad["patch"]["min_extent_mm"] = [500.0, 500.0, 500.0]
    with pytest.raises(ValueError, match="outside the allowed physical range"):
        build_dual_datasets(bad)


def test_crop_draws_are_deterministic_per_epoch(config):
    tr = build_dual_datasets(config)["train"]
    tr.set_epoch(2)
    a, b = tr[0], tr[0]
    assert all((x["image"] == y["image"]).all() for x, y in zip(a, b))
