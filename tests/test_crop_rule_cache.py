"""Label-free head-and-neck crop rule and the one-file-per-case cropped cache (synthetic data only)."""
import nibabel as nib
import numpy as np
import pytest

from src.data import crop_cache
from src.data.crop_rule import body_slice_stats, box_from_stats, head_neck_box, margin_mm

SP = (1.0, 1.0, 2.0)


def _ct(shape=(80, 70, 60), top=50, bottom=5, centre=(30, 35), half=(25, 20)):
    """Air (-1000) with a body-like block (HU 40) from slice `bottom` to `top`, x/y around `centre`."""
    ct = np.full(shape, -1000.0, np.float32)
    ct[centre[0] - half[0]:centre[0] + half[0], centre[1] - half[1]:centre[1] + half[1], bottom:top + 1] = 40.0   # area >= 1500 mm2 per slice
    return ct


def test_top_of_head_is_the_most_superior_body_slice():
    st = body_slice_stats(_ct(), SP, (100.0,))
    assert st["top_k"] == 50 and st["bottom_k"] == 5


def test_stray_noise_voxels_above_the_head_do_not_move_the_top():
    ct = _ct()
    ct[5, 5, 58] = 500.0  # one bright voxel: area 1 mm2 << 1500 mm2
    assert body_slice_stats(ct, SP, (100.0,))["top_k"] == 50


def test_box_is_fixed_physical_length_going_down_from_the_top():
    ct = _ct()
    box = head_neck_box(ct, SP, length_mm=40.0, width_mm=30.0)  # 20 slices at 2 mm, 30 voxels at 1 mm
    assert box[2] == (31, 51)
    assert box[0][1] - box[0][0] == 30 and box[1][1] - box[1][0] == 30
    cx = (box[0][0] + box[0][1]) / 2
    assert abs(cx - 30) <= 1.5  # centred on the body centroid


def test_rule_never_reads_a_label_and_is_the_same_function_for_every_split():
    import inspect
    assert "label" not in inspect.getsource(head_neck_box).lower().replace("labelled", "")
    assert "label" not in inspect.getsource(body_slice_stats).lower()


def test_box_is_clamped_by_shifting_not_shrinking_and_short_scans_keep_everything():
    box = box_from_stats((80, 70, 60), SP, top_k=50, cx=3.0, cy=35.0, length_mm=40.0, width_mm=30.0)
    assert box[0] == (0, 30)  # near the border: shifted inside, still 30 wide
    short = box_from_stats((80, 70, 10), SP, top_k=9, cx=40, cy=35, length_mm=400.0, width_mm=400.0)
    assert short == ((0, 80), (0, 70), (0, 10))


def test_margin_is_negative_when_a_label_leaves_the_box_and_positive_inside():
    box = ((10, 40), (10, 40), (10, 30))
    assert margin_mm(box, (1, 1, 2), ((15, 35), (15, 35), (12, 28))) == pytest.approx(4.0)   # z: (12-10)*2
    assert margin_mm(box, (1, 1, 2), ((5, 35), (15, 35), (12, 28))) == pytest.approx(-5.0)


def _write_case(root, pid="CHUM-001"):
    ct = _ct((40, 40, 30), top=25, bottom=2, centre=(20, 20), half=(20, 20))
    pet = np.random.RandomState(0).rand(*ct.shape).astype(np.float32) * 5
    lab = np.zeros(ct.shape, np.uint8)
    lab[15:20, 15:20, 18:23] = 1
    lab[22:25, 22:25, 15:17] = 2
    aff = np.diag([1.0, 1.0, 2.0, 1.0])
    aff[:3, 3] = [-20, -20, -40]
    d = root / "src" / pid
    d.mkdir(parents=True)
    for name, a in ((f"{pid}__CT.nii.gz", ct), (f"{pid}__PT.nii.gz", pet), (f"{pid}.nii.gz", lab)):
        nib.save(nib.Nifti1Image(a, aff), d / name)
    return {"patient_id": pid, "centre": "CHUM", "modalities": ["CT", "PT"],
            "image": [str(d / f"{pid}__CT.nii.gz"), str(d / f"{pid}__PT.nii.gz")], "label": str(d / f"{pid}.nii.gz")}, ct, pet, lab


def test_cache_file_roundtrip_dtypes_raw_labels_and_geometry(tmp_path):
    s, ct, pet, lab = _write_case(tmp_path)
    row = crop_cache.build_case(s, tmp_path / "cache", length_mm=40.0, width_mm=30.0)
    path = tmp_path / "cache" / "CHUM-001.acf"
    assert path.exists() and row["bytes"] == path.stat().st_size and not list((tmp_path / "cache").glob("*.tmp"))
    h, a = crop_cache.open_case(path)
    assert a["ct"].dtype == np.int16 and a["pet"].dtype == np.float16 and a["label"].dtype == np.uint8
    assert a["ct"].shape == a["pet"].shape == a["label"].shape == tuple(h["shape"])
    assert h["label_values"] == [0, 1, 2]                      # raw values kept, so K can be chosen later
    (x0, x1), (y0, y1), (z0, z1) = h["box"]
    sl = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
    assert np.array_equal(a["label"], lab[sl]) and np.array_equal(a["ct"], np.rint(ct[sl]).astype(np.int16))
    assert np.allclose(a["pet"].astype(np.float32), pet[sl], atol=5e-3)
    # the cropped affine places voxel 0 where the full volume's voxel (x0,y0,z0) was
    full = np.diag([1.0, 1.0, 2.0, 1.0])
    full[:3, 3] = [-20, -20, -40]
    assert np.allclose(np.asarray(h["affine"]) @ [0, 0, 0, 1], full @ [x0, y0, z0, 1])
    assert crop_cache.is_valid(path, crop_cache.signature(40.0, 30.0))
    assert not crop_cache.is_valid(path, crop_cache.signature(50.0, 30.0))  # a different rule invalidates it


def test_memmap_reads_without_loading_and_labels_are_inside_the_box(tmp_path):
    s, *_ = _write_case(tmp_path)
    crop_cache.build_case(s, tmp_path / "cache", length_mm=60.0, width_mm=40.0)
    h, a = crop_cache.open_case(tmp_path / "cache" / "CHUM-001.acf", mmap=True)
    assert isinstance(a["ct"], np.memmap)
    assert int((np.asarray(a["label"]) > 0).sum()) == 5 * 5 * 5 + 3 * 3 * 2   # no label voxel was cut


def test_pet_above_threshold_is_stored_as_float32_not_inf_and_other_cases_stay_float16(tmp_path):
    s, ct, pet, lab = _write_case(tmp_path)
    big = pet.copy()
    big[20, 20, 20] = 311_000.0   # beyond float16 range (65504)
    nib.save(nib.Nifti1Image(big, nib.load(s["image"][1]).affine), s["image"][1])
    crop_cache.build_case(s, tmp_path / "c_big", length_mm=60.0, width_mm=40.0)
    h, a = crop_cache.open_case(tmp_path / "c_big" / "CHUM-001.acf")
    assert a["pet"].dtype == np.float32 and h["pet_dtype"] == "float32"
    assert np.isfinite(np.asarray(a["pet"])).all() and float(np.asarray(a["pet"]).max()) == pytest.approx(311_000.0)  # not clipped
    nib.save(nib.Nifti1Image(pet, nib.load(s["image"][1]).affine), s["image"][1])
    crop_cache.build_case(s, tmp_path / "c_small", length_mm=60.0, width_mm=40.0)
    h2, a2 = crop_cache.open_case(tmp_path / "c_small" / "CHUM-001.acf")
    assert a2["pet"].dtype == np.float16 and h2["pet_dtype"] == "float16"
