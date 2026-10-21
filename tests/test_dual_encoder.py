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
