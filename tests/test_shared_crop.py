"""Fixed-size voxel crops on one shared CT/PET/label grid (src/data/crop.py).

Shared-grid counterpart of tests/test_physical_crop.py (which stays, for the two-grid path). Under pet_mode shared
one voxel window cuts every channel; the properties checked (a known blob lands where it should,
padding keeps positions, oblique affines are rejected, CT/PET/label are cut by one window)
are asserted here."""
import nibabel as nib
import numpy as np
import pytest

from src.data.crop import (
    crop_patch, crop_start, extract_padded, grid_from_affine, sample_centre_vox,
)

SHAPE = (40, 36, 30)


def _vols(blob):
    img = np.zeros((2,) + SHAPE, np.float32)
    lab = np.zeros(SHAPE, np.uint8)
    img[(0,) + tuple(blob)] = 1.0   # CT blob
    img[(1,) + tuple(blob)] = 2.0   # PET blob at the same voxel
    lab[tuple(blob)] = 1
    return {"image": img, "label": lab}


@pytest.mark.parametrize("blob", [(20, 18, 15), (3, 2, 1), (38, 35, 29)])
def test_blob_lands_at_the_same_place_in_ct_pet_and_label(blob):
    p, start = crop_patch(_vols(blob), blob, (16, 16, 16), pad=True)
    pos = {k: np.unravel_index(np.argmax(p["image"][i]), (16,) * 3) for i, k in enumerate(("ct", "pet"))}
    assert pos["ct"] == pos["pet"] == tuple(np.argwhere(p["label"] == 1)[0])
    assert tuple(np.asarray(pos["ct"]) + start) == blob  # world (voxel) position preserved


def test_window_is_centred_on_the_requested_voxel_when_it_fits():
    p, start = crop_patch(_vols((20, 18, 15)), (20, 18, 15), (16, 16, 16), pad=True)
    assert start.tolist() == [12, 10, 7]
    assert np.argwhere(p["label"] == 1)[0].tolist() == [8, 8, 8]


def test_pad_false_clamps_inside_the_volume_and_always_returns_full_size():
    p, start = crop_patch(_vols((0, 0, 0)), (0, 0, 0), (16, 16, 16), pad=False)
    assert start.tolist() == [0, 0, 0] and p["image"].shape == (2, 16, 16, 16)
    p, start = crop_patch(_vols((39, 35, 29)), (39, 35, 29), (16, 16, 16), pad=False)
    assert (start + 16).tolist() == [40, 36, 30]


def test_pad_false_raises_if_volume_smaller_than_window():
    with pytest.raises(ValueError, match="pad=True"):
        crop_start((5, 5, 5), (64, 64, 64), (32, 32, 32), pad=False)


def test_pad_true_gives_full_size_patch_with_zeros_outside_and_keeps_positions():
    p, start = crop_patch(_vols((0, 0, 0)), (0, 0, 0), (16, 16, 16), pad=True)
    assert p["image"].shape == (2, 16, 16, 16) and start.tolist() == [-8, -8, -8]
    assert p["image"][0, 8, 8, 8] == 1.0 and p["image"][1, 8, 8, 8] == 2.0   # the blob, not shifted
    assert p["image"][:, :8].sum() == 0 and p["label"][:8].sum() == 0         # padding is zeros


def test_window_larger_than_volume_is_padded_not_cropped():
    small = {"image": np.ones((2, 10, 10, 10), np.float32), "label": np.ones((10, 10, 10), np.uint8)}
    p, _ = crop_patch(small, (5, 5, 5), (16, 16, 16), pad=True)
    assert p["image"].shape == (2, 16, 16, 16) and p["image"].sum() == 2 * 1000


def test_arrays_not_on_one_grid_are_an_error():
    with pytest.raises(ValueError, match="not on one grid"):
        crop_patch({"image": np.zeros((2, 8, 8, 8)), "label": np.zeros((8, 8, 9))}, (4, 4, 4), (4, 4, 4))


def test_extract_padded_rejects_a_window_that_misses_the_volume():
    with pytest.raises(ValueError, match="does not overlap"):
        extract_padded(np.zeros((8, 8, 8)), (100, 0, 0), (4, 4, 4))


def test_sample_centre_is_foreground_when_pos_only_and_uniform_when_neg_only():
    fg = np.array([[1, 2, 3], [4, 5, 6]])
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert tuple(sample_centre_vox(fg, SHAPE, 1.0, 0.0, rng)) in {(1, 2, 3), (4, 5, 6)}
    c = sample_centre_vox(fg, SHAPE, 0.0, 1.0, rng)
    assert ((c >= 0) & (c <= np.asarray(SHAPE) - 1)).all()
    assert sample_centre_vox(np.zeros((0, 3), int), SHAPE, 1.0, 0.0, rng).shape == (3,)  # no fg -> uniform


def test_oblique_affine_is_rejected_not_approximated():
    a = np.diag([1.0, 1.0, 1.0, 1.0])
    a[0, 1] = 0.3
    with pytest.raises(ValueError, match="not diagonal-positive"):
        grid_from_affine(SHAPE, a)


def test_grid_from_affine_reads_spacing_and_origin():
    a = np.diag([0.98, 0.98, 3.27, 1.0])
    a[:3, 3] = [-250.0, -250.0, -100.0]
    g = grid_from_affine(SHAPE, a)
    assert g.spacing.tolist() == [0.98, 0.98, 3.27] and g.origin.tolist() == [-250.0, -250.0, -100.0]
    assert np.allclose(g.voxel_to_world((1, 0, 2)), [-249.02, -250.0, -93.46])


def test_nifti_roundtrip_with_flipped_axis_then_canonical_orientation(tmp_path):
    """LPS-stored volume (as in HECKTOR) -> closest canonical RAS: the blob keeps its world position."""
    blob = np.array([10, 12, 5])
    arr = np.zeros(SHAPE, np.float32)
    arr[tuple(blob)] = 1.0
    aff = np.diag([-1.0, -1.0, 2.0, 1.0])   # LPS
    aff[:3, 3] = [100.0, 90.0, -40.0]
    nib.save(nib.Nifti1Image(arr, aff), tmp_path / "x.nii.gz")
    img = nib.as_closest_canonical(nib.load(tmp_path / "x.nii.gz"))
    world_before = aff[:3, :3] @ blob + aff[:3, 3]
    idx = np.unravel_index(np.argmax(np.asarray(img.dataobj)), SHAPE)
    world_after = img.affine[:3, :3] @ np.asarray(idx) + img.affine[:3, 3]
    assert np.allclose(world_before, world_after)
    grid_from_affine(SHAPE, img.affine)  # diagonal-positive after reorientation
