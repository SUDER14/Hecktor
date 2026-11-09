import pytest

from src.data.geometry import check_patch_extent, patch_extent_mm


def test_extent_is_voxels_times_spacing():
    assert patch_extent_mm((96, 96, 64), (1, 1, 2)).tolist() == [96.0, 96.0, 128.0]


def test_extent_varies_with_spacing_for_a_fixed_voxel_patch():
    # same 96^3 patch: the physical box depends on the scan, which is what FiLM must absorb
    assert check_patch_extent((96,) * 3, (0.5, 0.5, 1.0)).tolist() == [48.0, 48.0, 96.0]
    assert check_patch_extent((96,) * 3, (1.0, 1.0, 5.0)).tolist() == [96.0, 96.0, 480.0]


def test_too_small_patch_raises():
    with pytest.raises(ValueError, match="outside the allowed physical range"):
        check_patch_extent((96,) * 3, (0.05, 0.05, 0.05))


def test_too_large_patch_raises_which_usually_means_wrong_unit():
    with pytest.raises(ValueError, match="outside the allowed physical range"):
        check_patch_extent((96,) * 3, (50.0, 50.0, 50.0))


def test_bounds_are_overridable_and_context_is_named():
    check_patch_extent((16,) * 3, (1.0,) * 3, min_mm=(10, 10, 10))
    with pytest.raises(ValueError, match="MDA-001"):
        check_patch_extent((16,) * 3, (1.0,) * 3, min_mm=(40, 40, 40), context="MDA-001")


@pytest.mark.parametrize("bad", [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (float("nan"), 1.0, 1.0), (1.0, 1.0)])
def test_bad_spacing_raises(bad):
    with pytest.raises(ValueError, match="spacing must be 3 positive finite"):
        check_patch_extent((96,) * 3, bad)
