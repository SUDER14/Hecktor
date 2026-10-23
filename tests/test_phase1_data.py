"""Phase 1: HECKTOR 2026 layout, conditioning, splits, aligned dual crops -- all on synthetic data."""
import copy
import json
import shutil

import nibabel as nib
import numpy as np
import pytest
import yaml

from src.data.conditioning import to_cond4
from src.data.dual_cache import DualCropDataset, DualVolumeDataset, build_dual_datasets
from src.data.hecktor2026 import Hecktor2026Adapter
from src.data.physical_crop import physical_crop
from src.data.splits import check_test_centres, group_split_by_centre
from src.data.synthetic import SynthConfig, write_synthetic_hecktor2026


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth") / "hecktor2026_training"
    ids = write_synthetic_hecktor2026(root, SynthConfig(fov_mm=(64, 64, 64), n_per_centre=3, seed=1))
    return root, ids


@pytest.fixture(scope="module")
def config(cohort, tmp_path_factory):
    root, _ = cohort
    work = tmp_path_factory.mktemp("work")
    cfg = yaml.safe_load(open("configs/synth_native.yaml"))
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
    assert s["meta"]["Centre"] == s["centre"]  # CSV attached


def test_adapter_missing_modality_is_an_error(cohort, tmp_path):
    root, _ = cohort
    dst = tmp_path / "d"
    shutil.copytree(root, dst)
    next(dst.glob("CHUM-001/*__PT.nii.gz")).unlink()
    with pytest.raises(FileNotFoundError, match="CHUM-001"):
        Hecktor2026Adapter(dst).training_samples()


def test_synthetic_spacings_differ_by_modality_and_centre(cohort):
    root, _ = cohort

    def z(p):
        return tuple(round(float(v), 3) for v in nib.load(p).header.get_zooms()[:3])

    assert z(root / "CHUM-001" / "CHUM-001__CT.nii.gz") == (1, 1, 1)
    assert z(root / "CHUM-001" / "CHUM-001__PT.nii.gz") == (4, 4, 4)
    assert z(root / "CHUP-001" / "CHUP-001__CT.nii.gz") == (1, 1, 3)  # 1x1x3 variant


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


def test_cond4_and_train_only_stats_and_split_meta(config):
    ds = build_dual_datasets(config)
    stats = json.load(open(config["conditioning"]["stats_path"]))
    assert set(stats) == {"CT", "PT"}
    # stats must be reproducible from the TRAIN patients only
    sz_ct = [nib.load(s["image"][0]).header.get_zooms()[2] for s in ds["train"].samples]
    assert stats["CT"]["slice_thickness"]["mean"] == pytest.approx(float(np.mean(sz_ct)), abs=1e-3)
    split = json.load(open(config["split"]["path"]))
    assert split["_meta"]["seed"] == config["seed"]
    v = ds["val"][0]
    assert v["cond_ct"].shape == (4,) and v["cond_ct"][3] == 0 and v["cond_pet"][3] == 1
    with pytest.raises(ValueError):
        to_cond4(np.zeros(5))


def test_dual_crop_alignment_and_shapes(config):
    ds = build_dual_datasets(config)
    tr: DualCropDataset = ds["train"]
    tr.set_epoch(1)
    for i in range(len(tr)):
        for item in tr[i]:
            assert item["ct"].shape == (1, 32, 32, 32)
            ext_ct = np.array(item["ct"].shape[1:]) * item["ct_spacing"].numpy()
            ext_pet = np.array(item["pet"].shape[1:]) * item["pet_spacing"].numpy()
            tol = np.maximum(item["ct_spacing"], item["pet_spacing"]).numpy() + 1e-4
            assert np.all(np.abs(ext_ct - ext_pet) <= tol)
            assert item["label"].shape == item["ct"].shape
    v = ds["val"][0]  # val volumes keep native, different grids
    assert isinstance(ds["val"], DualVolumeDataset) and v["ct"].shape != v["pet"].shape


def test_crop_draws_are_deterministic_per_epoch(config):
    tr = build_dual_datasets(config)["train"]
    tr.set_epoch(2)
    a, b = tr[0], tr[0]
    assert all((x["ct"] == y["ct"]).all() for x, y in zip(a, b))


def _aff(o, s):
    a = np.diag([s, s, s, 1.0])
    a[:3, 3] = o
    return a


def test_padding_near_borders_keeps_world_positions():
    ct = np.zeros((20, 20, 20), np.float32)
    ct[10, 10, 10] = 1.0                                   # 1 mm voxel, centre at world (10,10,10)
    pet = np.zeros((5, 5, 5), np.float32)
    pet[2, 2, 2] = 1.0                                     # 4 mm voxel, origin 2 -> centre at 2+2*4=10
    vols = {"ct": (ct, _aff([0, 0, 0], 1.0)), "pet": (pet, _aff([2.0, 2.0, 2.0], 4.0)),
            "label": (ct, _aff([0, 0, 0], 1.0))}
    with pytest.raises(ValueError):                        # default stays strict
        physical_crop(copy.deepcopy(vols), "ct", (32, 32, 32), (10, 10, 10))
    patches, info = physical_crop(vols, "ct", (32, 32, 32), (10, 10, 10), pad=True)
    assert patches["ct"].shape == (32, 32, 32) and patches["pet"].shape == (8, 8, 8)
    # start_mm is a leading edge; voxel centre = edge + 0.5 * spacing
    ct_world = info["start_mm"]["ct"] + (np.argwhere(patches["ct"] == 1)[0] + 0.5) * 1.0
    pet_world = info["start_mm"]["pet"] + (np.argwhere(patches["pet"] == 1)[0] + 0.5) * 4.0
    assert np.allclose(ct_world, [10, 10, 10], atol=1e-6)  # the lit CT voxel is still at its world position
    assert np.abs(pet_world - ct_world).max() <= 4.0
