"""`.acf` cache datasets: CT clip/scale, per-case PET normalisation (99.5th percentile of positive PET inside the CT
body mask), K = 3 labels, native and resampled modes, split and conditioning. Synthetic cohort for the logic;
two real cached cases (one from the MDA-402..447 block, one normal) when D:/hecktor_cache exists."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.data import crop_cache
from src.data.acf_dataset import (
    AcfCropDataset, AcfVolumeDataset, build_acf_datasets, ct_normalise, get_pet_scale, pet_scale,
    resample_target_shape, resample_volume, resampled_affine, write_scale_sidecar,
)
from src.data.conditioning import make_cond
from src.data.hecktor2026 import Hecktor2026Adapter
from src.data.splits import save_split, stratified_val_split
from src.data.synthetic_shared import SynthConfig, write_synthetic_hecktor2026

REAL = Path("D:/hecktor_cache")


def test_ct_clip_and_scale_endpoints():
    x = np.array([-3024, -1000, 0, 1000, 1964, 23152], np.float32)
    assert ct_normalise(x).tolist() == [0.0, 0.0, 0.5, 1.0, 1.0, 1.0]


def test_pet_scale_uses_only_positive_voxels_inside_the_body():
    ct = np.full((20, 20, 20), -1000, np.int16)
    ct[5:15, 5:15, 5:15] = 40                       # body
    pet = np.zeros(ct.shape, np.float32)
    pet[5:15, 5:15, 5:15] = np.linspace(0.1, 10.0, 1000).reshape(10, 10, 10)
    pet[0, 0, 0] = 1e6                              # hot voxel outside the body: must be ignored
    s = pet_scale(ct, pet)
    assert s == pytest.approx(np.percentile(np.linspace(0.1, 10.0, 1000), 99.5), rel=1e-5)
    assert pet_scale(ct, pet * 1000.0) == pytest.approx(s * 1000.0, rel=1e-5)   # equivariant to the unit
    with pytest.raises(ValueError, match="no positive PET"):
        pet_scale(ct, np.zeros_like(pet))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    work = tmp_path_factory.mktemp("acfds")
    root = work / "d"
    write_synthetic_hecktor2026(root, SynthConfig(fov_mm=(96, 96, 96), n_per_centre=3, seed=1, tumour_prob=1.0))
    samples = Hecktor2026Adapter(root).training_samples()
    cache = work / "cache"
    for s in samples:
        crop_cache.build_case(s, cache, length_mm=96.0, width_mm=96.0)
    split = stratified_val_split(samples, ["HGJ", "USZ"], 0.2, seed=3)
    save_split(split, work / "split.json", meta={"seed": 3})
    train = set(split["train"])
    ct_sp = {s["patient_id"]: crop_cache.read_header(cache / f"{s['patient_id']}.acf")["spacing"] for s in samples}
    sx = [ct_sp[p] for p in train]
    stats = {"CT": {"spacing_x": {"mean": float(np.mean([a[0] for a in sx])), "std": float(np.std([a[0] for a in sx]) or 1)},
                    "spacing_y": {"mean": float(np.mean([a[1] for a in sx])), "std": float(np.std([a[1] for a in sx]) or 1)},
                    "slice_thickness": {"mean": float(np.mean([a[2] for a in sx])), "std": float(np.std([a[2] for a in sx]) or 1)}}}
    (work / "stats.json").write_text(json.dumps(stats))
    cfg = {"seed": 7, "data": {"mode": "native", "pet_mode": "shared", "label_map": [0, 1, 2]},
           "cache": {"dir": str(cache)}, "split": {"path": str(work / "split.json")},
           "conditioning": {"stats_path": str(work / "stats.json")},
           "patch": {"spatial_size": [32, 32, 32], "pos": 2, "neg": 1, "num_samples": 2, "min_extent_mm": [8, 8, 8]},
           "baseline": {"resample_spacing": [2.0, 2.0, 2.0]}, "model": {"out_channels": 3}}
    return cfg, split, stats, cache


def test_train_items_have_two_normalised_channels_k3_labels_and_zscored_spacing(world):
    cfg, split, stats, cache = world
    ds = build_acf_datasets(cfg)
    assert isinstance(ds["train"], AcfCropDataset) and isinstance(ds["val"], AcfVolumeDataset)
    assert {p for d in ds.values() for p in [s["patient_id"] for s in d.samples]} == {p for k in split if not k.startswith("_") for p in split[k]}
    ds["train"].set_epoch(1)
    seen = set()
    for i in range(len(ds["train"])):
        for it in ds["train"][i]:
            assert it["image"].shape == (2, 32, 32, 32) and it["label"].shape == (1, 32, 32, 32)
            assert 0.0 <= it["image"][0].min() and it["image"][0].max() <= 1.0     # CT in [0, 1]
            assert it["image"][1].min() >= 0 and np.isfinite(it["image"].numpy()).all()
            seen |= set(np.unique(it["label"].numpy()).tolist())
            assert np.allclose(it["cond"].numpy(), make_cond(it["spacing"].numpy(), stats), atol=1e-5)
    assert seen <= {0.0, 1.0, 2.0} and 1.0 in seen


def test_pet_channel_is_scaled_by_the_per_case_scale_and_is_unit_free(world):
    cfg, split, _, cache = world
    pid = split["val"][0]
    h, arr = crop_cache.open_case(cache / f"{pid}.acf")
    sc = pet_scale(arr["ct"], arr["pet"])
    ds = build_acf_datasets(cfg)["val"]
    item = next(ds[i] for i in range(len(ds)) if ds.samples[i]["patient_id"] == pid)
    assert np.allclose(item["image"][1].numpy(), np.asarray(arr["pet"]).astype(np.float32) / sc, rtol=1e-5)
    # rescaling the PET unit of one case leaves the normalised channel unchanged
    big = np.asarray(arr["pet"]).astype(np.float32) * 1000.0
    assert np.allclose(big / pet_scale(arr["ct"], big), item["image"][1].numpy(), rtol=1e-4)


def test_draws_are_deterministic_per_epoch_and_differ_across_epochs(world):
    cfg, *_ = world
    tr = build_acf_datasets(cfg)["train"]
    tr.set_epoch(2)
    a, b = tr[0], tr[0]
    assert all(np.array_equal(x["image"].numpy(), y["image"].numpy()) for x, y in zip(a, b))
    tr.set_epoch(3)
    c = tr[0]
    assert not all(np.array_equal(x["image"].numpy(), y["image"].numpy()) for x, y in zip(a, c))


def test_native_val_item_carries_native_label_and_affine(world):
    cfg, *_ = world
    v = build_acf_datasets(cfg)["val"][0]
    assert v["image"].shape[1:] == v["label"].shape[1:] == tuple(v["native_label"].shape)
    assert np.array_equal(v["affine"].numpy(), v["native_affine"].numpy())


def test_resampled_mode_gives_the_fixed_target_spacing_for_every_case(world):
    cfg, split, stats, cache = world
    cfg = copy.deepcopy(cfg)
    cfg["data"]["mode"] = "resampled"
    ds = build_acf_datasets(cfg)
    v = ds["val"][0]
    h = crop_cache.read_header(cache / f"{ds['val'].samples[0]['patient_id']}.acf")
    assert v["image"].shape[1:] == resample_target_shape(h["shape"], h["spacing"], [2.0, 2.0, 2.0])
    assert np.allclose(np.diag(v["affine"].numpy())[:3], 2.0) and v["spacing"].tolist() == [2.0, 2.0, 2.0]
    assert np.allclose(v["cond"].numpy(), make_cond([2.0, 2.0, 2.0], stats), atol=1e-5)
    assert set(np.unique(v["label"].numpy()).tolist()) <= {0.0, 1.0, 2.0}      # nearest: no blended classes
    ds["train"].set_epoch(1)
    assert ds["train"][0][0]["image"].shape == (2, 32, 32, 32)


def test_resample_volume_preserves_a_blob_position_and_constant_fields():
    sp, tgt = (1.0, 1.0, 2.0), (2.0, 2.0, 2.0)
    a = np.zeros((40, 40, 20), np.float32)
    a[20, 20, 10] = 1.0
    r = resample_volume({"a": a, "c": np.full(a.shape, 3.0, np.float32)}, sp, tgt, {"a": 1, "c": 1}, {"a": 0.0, "c": 0.0})
    assert r["a"].shape == (20, 20, 20) and np.allclose(r["c"][2:-2, 2:-2, 2:-2], 3.0)
    aff = resampled_affine(np.diag([1.0, 1.0, 2.0, 1.0]), sp, tgt)
    i = np.unravel_index(np.argmax(r["a"]), r["a"].shape)
    world_new = aff[:3, :3] @ np.asarray(i) + aff[:3, 3]
    assert np.abs(world_new - np.array([20.0, 20.0, 20.0])).max() <= 1.0     # within one target voxel (2 mm)


def test_sidecar_scales_match_computed_scales(world, tmp_path):
    cfg, split, _, cache = world
    pids = split["val"][:2]
    sc = write_scale_sidecar(cache, pids)
    assert (cache / "pet_scale.json").exists()
    for p in pids:
        _, arr = crop_cache.open_case(cache / f"{p}.acf")
        assert sc[p] == pytest.approx(pet_scale(arr["ct"], arr["pet"]))
        assert get_pet_scale(cache, p) == pytest.approx(sc[p])


@pytest.mark.skipif(not (REAL / "MDA-410.acf").exists(), reason="real cache not built")
def test_real_cases_block_and_normal_end_up_on_the_same_scale():
    """MDA-410 (inside the MDA-402..447 block, header PET max 91,587, stored float32) versus CHUM-001 (normal)."""
    out = {}
    for pid in ("MDA-410", "CHUM-001"):
        h, arr = crop_cache.open_case(REAL / f"{pid}.acf")
        raw = np.asarray(arr["pet"]).astype(np.float32)
        sc = pet_scale(arr["ct"], arr["pet"])
        lab = np.asarray(arr["label"]) > 0
        body = (np.asarray(arr["ct"]) > -500) & (raw > 0)
        norm = raw / sc
        out[pid] = {"scale": sc, "raw_label_mean": float(raw[lab].mean()), "norm_label_mean": float(norm[lab].mean()),
                    "norm_p99_5": float(np.percentile(norm[body], 99.5)), "finite": bool(np.isfinite(norm).all())}
    b, n = out["MDA-410"], out["CHUM-001"]
    assert b["scale"] > 100 * n["scale"]                          # the raw scales differ by orders of magnitude
    for o in out.values():
        assert o["finite"] and o["norm_p99_5"] == pytest.approx(1.0, rel=1e-4)    # the rule's own definition
        assert 0.1 < o["norm_label_mean"] < 5.0                    # the tumour lands at an O(1) value
    assert 0.1 < b["norm_label_mean"] / n["norm_label_mean"] < 10.0   # and on a comparable scale across the two
    ds = AcfVolumeDataset([{"patient_id": "MDA-410", "centre": "MDA"}], REAL,
                          {"baseline": {"resample_spacing": [1, 1, 3]}, "model": {"out_channels": 3},
                           "data": {"label_map": [0, 1, 2]}},
                          {"CT": {k: {"mean": 1.0, "std": 1.0} for k in ("spacing_x", "spacing_y", "slice_thickness")}})
    it = ds[0]
    assert np.isfinite(it["image"].numpy()).all() and it["image"][0].max() <= 1.0 and it["image"][1].max() > 1.0


def _fake(centres):
    return [{"patient_id": f"{c}-{i:03d}", "centre": c} for c, n in centres.items() for i in range(1, n + 1)]


def test_stratified_val_split_is_disjoint_seeded_and_stratified_by_centre():
    samples = _fake({"CHUS": 20, "CHUV": 10, "MDA": 100, "CHUM": 30, "HGJ": 5})
    sp = stratified_val_split(samples, ["CHUS", "CHUV"], 0.10, seed=42)
    assert sp == stratified_val_split(samples, ["CHUS", "CHUV"], 0.10, seed=42)          # seeded
    assert sp != stratified_val_split(samples, ["CHUS", "CHUV"], 0.10, seed=43)
    tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
    assert not (tr & va) and not (tr & te) and not (va & te) and len(tr | va | te) == len(samples)
    assert {p.split("-")[0] for p in te} == {"CHUS", "CHUV"} and not {p.split("-")[0] for p in tr | va} & {"CHUS", "CHUV"}
    n_val = {c: sum(p.startswith(c) for p in va) for c in ("MDA", "CHUM", "HGJ")}
    assert n_val == {"MDA": 10, "CHUM": 3, "HGJ": 1}                                      # round(0.1 n), at least 1
    with pytest.raises(ValueError, match="tiny"):
        stratified_val_split(_fake({"HMR": 5, "USZ": 5, "MDA": 20}), ["HMR", "USZ"], 0.1, 1)  # only tiny centres held out


def test_committed_real_split_matches_the_decision():
    p = Path("configs/splits/hecktor2026_seed42.json")
    sp = json.loads(p.read_text())
    assert sp["_meta"]["test_centres"] == ["CHUS", "CHUV"] and sp["_meta"]["seed"] == 42
    ids = {k: set(sp[k]) for k in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    assert sum(map(len, ids.values())) == 782 and {i.split("-")[0] for i in ids["test"]} == {"CHUS", "CHUV"}
    assert not {i.split("-")[0] for i in ids["train"] | ids["val"]} & {"CHUS", "CHUV"}
