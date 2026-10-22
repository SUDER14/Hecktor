"""Synthetic-tensor tests only: no dual-modality dataset exists locally (Decathlon has no PET)."""
import pytest
import torch

from src.models.dual_encoder import build_dual_encoder_unet

CFG = {"model": {"channels": [4, 8, 16], "strides": [2, 2], "out_channels": 2}}


def test_output_on_ct_grid_with_different_pet_grid():
    net = build_dual_encoder_unet(CFG)
    out = net(torch.randn(1, 1, 16, 16, 16), torch.randn(1, 1, 8, 10, 12))
    assert out.shape == (1, 2, 16, 16, 16)


def test_gradients_reach_both_encoders():
    net = build_dual_encoder_unet(CFG)
    net(torch.randn(1, 1, 16, 16, 16), torch.randn(1, 1, 8, 8, 8)).sum().backward()
    for enc in (net.ct_enc, net.pet_enc):
        assert all(p.grad is not None for p in enc.parameters())


def test_rejects_bad_ct_size():
    net = build_dual_encoder_unet(CFG)
    with pytest.raises(ValueError):
        net(torch.randn(1, 1, 18, 16, 16), torch.randn(1, 1, 8, 8, 8))


def test_rejects_pet_collapsing_to_one_voxel():
    net = build_dual_encoder_unet(CFG)  # total stride 4 -> PET sides must be >= 8
    with pytest.raises(ValueError):
        net(torch.randn(1, 1, 16, 16, 16), torch.randn(1, 1, 4, 8, 8))


# --- physical-extent check (replaces the old fixed-minimum-side PET guard) ---

def test_24_pet_with_96_ct_at_1mm_4mm_runs_on_default_depth():
    net = build_dual_encoder_unet(
        {"model": {"channels": [4, 8, 16, 32, 64], "strides": [2, 2, 2, 2], "out_channels": 2}})
    out = net(torch.randn(1, 1, 96, 96, 96), torch.randn(1, 1, 24, 24, 24),
              ct_spacing=(1.0, 1.0, 1.0), pet_spacing=(4.0, 4.0, 4.0))
    assert out.shape == (1, 2, 96, 96, 96)


def test_forward_rejects_mismatched_physical_extent():
    net = build_dual_encoder_unet(CFG)
    with pytest.raises(ValueError, match="different physical extents"):
        net(torch.randn(1, 1, 16, 16, 16), torch.randn(1, 1, 16, 16, 16),
            ct_spacing=(1.0, 1.0, 1.0), pet_spacing=(4.0, 4.0, 4.0))


def test_extent_check_is_per_batch_item_and_names_the_item():
    net = build_dual_encoder_unet(CFG)
    pet_sp = torch.tensor([[2.0] * 3, [4.0] * 3])  # item 1 does not match 16 mm CT / 8 vox
    with pytest.raises(ValueError, match="batch item 1"):
        net(torch.randn(2, 1, 16, 16, 16), torch.randn(2, 1, 8, 8, 8),
            ct_spacing=(1.0, 1.0, 1.0), pet_spacing=pet_sp)


def test_extent_tolerance_is_configurable():
    # 16 mm CT vs 10 vox x 2 mm = 20 mm PET: 4 mm off, one coarser voxel is 2 mm.
    args = (torch.randn(1, 1, 16, 16, 16), torch.randn(1, 1, 10, 10, 10))
    kw = dict(ct_spacing=(1.0,) * 3, pet_spacing=(2.0,) * 3)
    with pytest.raises(ValueError):
        build_dual_encoder_unet(CFG)(*args, **kw)
    loose = build_dual_encoder_unet({"model": {**CFG["model"], "extent_tol_voxels": 3.0}})
    assert loose(*args, **kw).shape == (1, 2, 16, 16, 16)


def test_only_one_spacing_is_an_error():
    net = build_dual_encoder_unet(CFG)
    with pytest.raises(ValueError, match="both"):
        net(torch.randn(1, 1, 16, 16, 16), torch.randn(1, 1, 8, 8, 8), ct_spacing=(1.0,) * 3)
