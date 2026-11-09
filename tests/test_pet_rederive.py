"""pet_mode "rederived": coarse PET derived from CT-grid PET. An approximation, never "native" PET."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.dual_cache import build_dual_datasets
from src.data.pet_rederive import coarse_spacing, rederive_pet, roundtrip_metrics, upsample_to_ct
from src.data.synthetic_shared import SynthConfig, write_synthetic_hecktor2026


def _aff(sp, origin=(-100.0, -90.0, -80.0)):
    a = np.diag(list(sp) + [1.0])
    a[:3, 3] = origin
    return a


def _blob(shape, centre, sigma):
    g = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    return np.exp(-sum((x - c) ** 2 for x, c in zip(g, centre)) / (2 * sigma ** 2)).astype(np.float32)


def test_coarse_spacing_never_goes_finer_than_the_ct():
    assert coarse_spacing((1, 1, 5.0), (4, 4, 4)).tolist() == [4, 4, 5.0]


def test_shape_spacing_and_constant_volume():
    pet = np.full((64, 64, 32), 3.0, np.float32)
    c, aff, info = rederive_pet(pet, _aff((1, 1, 2)), (4, 4, 4))
    assert c.shape == (16, 16, 16) and np.allclose(np.diag(aff)[:3], 4.0)
    assert np.allclose(c, 3.0, atol=1e-5)  # blur + linear sampling preserve a constant
    assert info["sigma_ct_voxels"] == pytest.approx([1.5, 1.5, 0.5])  # (ratio-1)/2


def test_blob_keeps_its_world_position_within_one_coarse_voxel():
    sp, origin = (1.0, 1.0, 2.0), (-100.0, -90.0, -80.0)
    pet = _blob((64, 64, 32), (30, 20, 12), 3.0)
    c, caff, _ = rederive_pet(pet, _aff(sp, origin), (4, 4, 4))
    i = np.unravel_index(np.argmax(c), c.shape)
    world = caff[:3, :3] @ np.asarray(i) + caff[:3, 3]
    truth = np.asarray(origin) + np.asarray([30, 20, 12]) * np.asarray(sp)
    assert np.abs(world - truth).max() <= 4.0


def test_roundtrip_is_exact_for_constant_lossy_for_a_spike_and_good_for_smooth_pet():
    sp = (1.0, 1.0, 1.0)
    for pet, expect_lossy in ((np.full((32, 32, 32), 2.0, np.float32), False),
                              (_blob((32, 32, 32), (16, 16, 16), 9.0), False),
                              (np.pad(np.ones((1, 1, 1), np.float32), 15) * 10, True)):
        c, caff, _ = rederive_pet(pet, _aff(sp), (4, 4, 4))
        back = upsample_to_ct(c, caff, pet.shape, _aff(sp))
        m = roundtrip_metrics(pet, back, (pet > 0.5 * pet.max()).astype(np.uint8))
        if expect_lossy:
            assert m["gtv_max_rel_change"] < -0.5   # a one-voxel hot spot loses most of its peak
        else:
            assert m["max_abs_err_rel_pet_max"] < 0.2
            if pet.std() > 0:  # Pearson r is undefined for a constant volume
                assert m["pearson_r"] > 0.97


def test_oblique_affine_is_rejected():
    a = _aff((1, 1, 1))
    a[0, 1] = 0.3
    with pytest.raises(ValueError, match="diagonal-positive"):
        rederive_pet(np.zeros((8, 8, 8), np.float32), a)


@pytest.fixture(scope="module")
def rederived_cfg(tmp_path_factory):
    work = tmp_path_factory.mktemp("rd")
    root = work / "hecktor2026_training"
    write_synthetic_hecktor2026(root, SynthConfig(fov_mm=(64, 64, 64), n_per_centre=3, seed=1))
    cfg = yaml.safe_load(open("configs/synth_shared.yaml"))
    cfg.update(data_dir=str(root))
    cfg["data"] = {"mode": "native", "pet_mode": "rederived", "pet_spacing": [4.0, 4.0, 4.0]}
    cfg["split"].update(path=str(work / "split.json"), test_centres=["MDA", "USZ"])
    cfg["conditioning"]["stats_path"] = str(work / "stats.json")
    cfg["cache"]["dir"] = str(work / "cache")
    cfg["patch"]["spatial_size"] = [32, 32, 32]
    return cfg


def test_rederived_cache_has_a_coarser_pet_grid_and_the_old_dual_path_runs(rederived_cfg):
    ds = build_dual_datasets(rederived_cfg)
    tr = ds["train"]
    tr.set_epoch(1)
    item = tr[0][0]
    assert item["ct"].shape == (1, 32, 32, 32) and item["pet"].shape[1] < item["ct"].shape[1]  # coarser PET grid
    cdir = Path(rederived_cfg["cache"]["dir"])
    meta = json.loads((cdir / f"{tr.samples[0]['patient_id']}.meta.json").read_text())
    assert meta["sig"]["pet_mode"] == "rederived" and meta["pet_spacing"][0] == pytest.approx(4.0)
    assert meta["ct_spacing"][0] < meta["pet_spacing"][0]


def test_signature_changes_with_pet_mode_so_stale_caches_rebuild(rederived_cfg):
    from src.data.dual_cache import signature
    other = copy.deepcopy(rederived_cfg)
    other["data"]["pet_spacing"] = [3.0, 3.0, 3.0]
    assert signature(rederived_cfg, "native") != signature(other, "native")
    plain = copy.deepcopy(rederived_cfg)
    plain["data"].pop("pet_mode")
    assert "pet_mode" not in signature(plain, "native")
