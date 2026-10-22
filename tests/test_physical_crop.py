"""Synthetic CT (1 mm) + PET (4 mm) volumes with different origins and a bright blob
at a known world position. No real dual-modality data exists locally."""
import nibabel as nib
import numpy as np
import pytest
import torch
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd

from src.data.physical_crop import RandPhysicalCropd, grid_from_affine, physical_crop
from src.models.dual_encoder import build_dual_encoder_unet

CT_ORIGIN = np.array([-60.0, -70.0, -80.0])
PET_ORIGIN = np.array([-57.3, -66.1, -78.4])  # not a multiple of anything shared with CT
CT_SHAPE, PET_SHAPE = (128, 128, 128), (32, 32, 32)


def _affine(origin, spacing):
    a = np.diag([spacing, spacing, spacing, 1.0])
    a[:3, 3] = origin
    return a


def _volumes(blob_mm, pet_origin=PET_ORIGIN):
    """CT/PET arrays + label (on CT grid), each with one bright voxel nearest blob_mm."""
    pet_origin = np.asarray(pet_origin)
    ct_aff, pet_aff = _affine(CT_ORIGIN, 1.0), _affine(pet_origin, 4.0)
    ct, pet, lab = (np.zeros(s, np.float32) for s in (CT_SHAPE, PET_SHAPE, CT_SHAPE))
    ct[tuple(np.rint((blob_mm - CT_ORIGIN) / 1.0).astype(int))] = 1.0
    lab[tuple(np.rint((blob_mm - CT_ORIGIN) / 1.0).astype(int))] = 1.0
    pet[tuple(np.rint((blob_mm - pet_origin) / 4.0).astype(int))] = 1.0
    return {"ct": (ct[None], ct_aff), "pet": (pet[None], pet_aff), "label": (lab[None], ct_aff)}


def _blob_world(patch, start_mm, spacing):
    idx = np.unravel_index(np.argmax(patch[0]), patch[0].shape)
    return start_mm + (np.asarray(idx) + 0.5) * spacing  # voxel centre = leading edge + half a voxel


@pytest.mark.parametrize("blob", [(10.0, 20.0, 5.0), (-20.3, 0.7, 11.9), (30.0, -25.0, -10.0)])
def test_blob_world_position_agrees_within_one_pet_voxel(blob):
    blob = np.array(blob)
    patches, info = physical_crop(_volumes(blob), "ct", (48, 48, 48), centre_mm=blob)
    assert patches["pet"].shape[1:] == (12, 12, 12)  # 48 mm / 4 mm
    ct_pos = _blob_world(patches["ct"], info["start_mm"]["ct"], info["spacing"]["ct"])
    pet_pos = _blob_world(patches["pet"], info["start_mm"]["pet"], info["spacing"]["pet"])
    assert np.abs(ct_pos - blob).max() <= 1.0     # CT blob lands within one CT voxel of truth
    assert np.abs(pet_pos - ct_pos).max() <= 4.0  # the required property: within one PET voxel
    # both windows start at the same world position, up to half a voxel of snapping each
    assert np.abs(info["start_mm"]["pet"] - info["start_mm"]["ct"]).max() <= 2.0 + 0.5


def test_large_origin_offset_still_aligned_where_index_scaling_would_not_be():
    """PET origin 20 mm off the CT origin: scaling the CT start index by 4 (a crop
    that ignores the affines) lands >4 mm from the CT window; the physical crop doesn't."""
    blob = np.array([10.0, 20.0, 5.0])
    pet_origin = CT_ORIGIN + np.array([21.7, 13.1, 9.4])
    vols = _volumes(blob, pet_origin)
    patches, info = physical_crop(vols, "ct", (48, 48, 48), centre_mm=blob)
    pet_grid = grid_from_affine(PET_SHAPE, vols["pet"][1])
    naive_start = pet_grid.start_edge(info["start_idx"]["ct"] // 4)
    assert np.abs(naive_start - info["start_mm"]["ct"]).max() > 4.0
    ct_pos = _blob_world(patches["ct"], info["start_mm"]["ct"], info["spacing"]["ct"])
    pet_pos = _blob_world(patches["pet"], info["start_mm"]["pet"], info["spacing"]["pet"])
    assert np.abs(pet_pos - ct_pos).max() <= 4.0


def test_centre_near_edge_is_clamped_jointly_so_windows_stay_aligned():
    patches, info = physical_crop(_volumes(np.array([0.0, 0.0, 0.0])), "ct", (48, 48, 48),
                                  centre_mm=np.array([-1000.0, 0.0, 1000.0]))
    assert patches["ct"].shape[1:] == (48,) * 3 and patches["pet"].shape[1:] == (12,) * 3
    assert np.abs(info["start_mm"]["pet"] - info["start_mm"]["ct"]).max() <= 2.5


def test_crop_output_passes_model_extent_check_and_wrong_pet_size_raises():
    blob = np.array([10.0, 20.0, 5.0])
    patches, info = physical_crop(_volumes(blob), "ct", (48, 48, 48), centre_mm=blob)
    net = build_dual_encoder_unet({"model": {"channels": [4, 8, 16], "strides": [2, 2], "out_channels": 2}})
    ct = torch.as_tensor(patches["ct"])[None]
    pet = torch.as_tensor(patches["pet"])[None]
    sp = (torch.tensor(info["spacing"]["ct"])[None], torch.tensor(info["spacing"]["pet"])[None])
    assert net(ct, pet, *sp).shape == (1, 2, 48, 48, 48)
    wrong_pet = torch.zeros(1, 1, 16, 16, 16)  # 64 mm, not 48 mm
    with pytest.raises(ValueError, match="different physical extents"):
        net(ct, wrong_pet, *sp)


def test_oblique_affine_is_rejected_not_approximated():
    a = _affine(CT_ORIGIN, 1.0)
    a[0, 1] = 0.3
    with pytest.raises(ValueError, match="not diagonal-positive"):
        grid_from_affine(CT_SHAPE, a)


def test_non_overlapping_fovs_raise():
    vols = _volumes(np.array([0.0, 0.0, 0.0]))
    vols["pet"] = (vols["pet"][0], _affine(PET_ORIGIN + 500.0, 4.0))
    with pytest.raises(ValueError, match="no common physical window"):
        physical_crop(vols, "ct", (48, 48, 48), centre_mm=np.zeros(3))


def test_end_to_end_transform_from_nifti_with_flipped_axis(tmp_path):
    """Files on disk (CT stored with a flipped x axis, as LAS) -> MONAI load ->
    Orientationd RAS -> RandPhysicalCropd. Blob world position is read back from the
    returned start_mm only."""
    blob = np.array([10.0, 20.0, 5.0])
    vols = _volumes(blob)
    ct_arr, ct_aff = vols["ct"][0][0], vols["ct"][1]
    lab_arr = vols["label"][0][0]
    flip_aff = ct_aff.copy()
    flip_aff[0, 0] = -1.0
    flip_aff[0, 3] = CT_ORIGIN[0] + 127.0  # the same physical volume, stored x-reversed
    for name, arr, aff in [("ct", ct_arr[::-1], flip_aff), ("lab", lab_arr[::-1], flip_aff),
                           ("pet", vols["pet"][0][0], vols["pet"][1])]:
        nib.save(nib.Nifti1Image(np.ascontiguousarray(arr), aff), tmp_path / f"{name}.nii.gz")
    data = {"ct": str(tmp_path / "ct.nii.gz"), "pet": str(tmp_path / "pet.nii.gz"),
            "label": str(tmp_path / "lab.nii.gz"), "patient_id": "SYN-001"}
    keys = ["ct", "pet", "label"]
    crop = RandPhysicalCropd("ct", "pet", "label", (48, 48, 48), pos=1, neg=0, num_samples=3)
    crop.set_random_state(seed=0)
    out = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys),
                   Orientationd(keys=keys, axcodes="RAS"), crop])(data)
    assert len(out) == 3
    for item in out:
        assert item["patient_id"] == "SYN-001" and item["pet"].shape[1:] == (12,) * 3
        ct_pos = _blob_world(item["ct"].numpy(), item["ct_start_mm"].numpy(), item["ct_spacing"].numpy())
        pet_pos = _blob_world(item["pet"].numpy(), item["pet_start_mm"].numpy(), item["pet_spacing"].numpy())
        assert np.abs(pet_pos - ct_pos).max() <= 4.0
        assert item["label"].shape == item["ct"].shape and item["label"].sum() >= 1
