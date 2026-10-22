import pytest

from src.data.geometry import check_physical_extent, patch_extent_mm


def test_extent_is_voxels_times_spacing():
    assert patch_extent_mm((96, 96, 64), (1, 1, 2)).tolist() == [96.0, 96.0, 128.0]


def test_24_pet_with_96_ct_at_1mm_4mm_passes():
    check_physical_extent((96,) * 3, (1.0,) * 3, (24,) * 3, (4.0,) * 3)


def test_mismatched_extent_raises_with_numbers():
    with pytest.raises(ValueError, match=r"axes \[0, 1, 2\].*96.*192"):
        check_physical_extent((96,) * 3, (1.0,) * 3, (24,) * 3, (8.0,) * 3)


def test_single_axis_mismatch_names_that_axis():
    with pytest.raises(ValueError, match=r"axes \[2\]"):
        check_physical_extent((96, 96, 96), (1, 1, 1), (24, 24, 16), (4, 4, 4))


def test_tolerance_is_one_coarser_voxel_and_configurable():
    # 100 mm CT vs 24 x 4 = 96 mm PET: 4 mm apart == exactly one coarser voxel -> ok.
    check_physical_extent((100,) * 3, (1.0,) * 3, (24,) * 3, (4.0,) * 3)
    # 5 mm apart -> too far at the default, fine at 2 coarser voxels.
    with pytest.raises(ValueError):
        check_physical_extent((101,) * 3, (1.0,) * 3, (24,) * 3, (4.0,) * 3)
    check_physical_extent((101,) * 3, (1.0,) * 3, (24,) * 3, (4.0,) * 3, tol_voxels=2.0)


def test_anisotropic_spacings_are_per_axis():
    check_physical_extent((96, 96, 48), (1, 1, 2), (24, 24, 24), (4, 4, 4))
