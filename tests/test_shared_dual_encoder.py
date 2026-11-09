"""B2 on a shared CT/PET grid: input (B, 2, X, Y, Z), channel 0 = CT, channel 1 = PET.

Synthetic tensors only. The cross-resolution tests of the old two-grid design (PET on its
own coarser grid, PET depth, CT/PET extent checks) are removed with that design: HECKTOR
2026 CT and PET share one grid (VERIFIED on all 782 cases)."""
import pytest
import torch

from src.models.dual_encoder_shared import build_dual_encoder_unet

CFG = {"model": {"channels": [4, 8, 16], "strides": [2, 2], "out_channels": 2}}
FOUR = {"model": {"channels": [4, 8, 16, 32, 64], "strides": [2, 2, 2, 2], "out_channels": 2}}


def test_output_on_the_shared_grid():
    net = build_dual_encoder_unet(CFG)
    assert net(torch.randn(1, 2, 16, 16, 16)).shape == (1, 2, 16, 16, 16)


def test_anisotropic_patch_shape_is_preserved():
    net = build_dual_encoder_unet(CFG)
    assert net(torch.randn(1, 2, 16, 16, 8)).shape == (1, 2, 16, 16, 8)


def test_gradients_reach_both_encoders():
    net = build_dual_encoder_unet(CFG)
    net(torch.randn(1, 2, 16, 16, 16)).sum().backward()
    for enc in (net.ct_enc, net.pet_enc):
        assert all(p.grad is not None for p in enc.parameters())


def test_each_encoder_sees_only_its_own_channel():
    net = build_dual_encoder_unet(CFG).eval()
    x = torch.randn(1, 2, 16, 16, 16)
    with torch.no_grad():
        base = net(x)
        no_pet = x.clone()
        no_pet[:, 1] = 0
        assert not torch.allclose(base, net(no_pet))  # PET channel matters
        no_ct = x.clone()
        no_ct[:, 0] = 0
        assert not torch.allclose(base, net(no_ct))   # CT channel matters


def test_rejects_bad_size():
    net = build_dual_encoder_unet(CFG)
    with pytest.raises(ValueError, match="divisible"):
        net(torch.randn(1, 2, 18, 16, 16))


def test_rejects_wrong_channel_count():
    net = build_dual_encoder_unet(CFG)
    with pytest.raises(ValueError, match="channel 0 = CT"):
        net(torch.randn(1, 1, 16, 16, 16))


# --- physical-extent sanity check on the patch (replaces the CT/PET cross-grid check) ---

def test_96_cubed_at_1mm_passes_the_extent_check():
    net = build_dual_encoder_unet(FOUR)
    out = net(torch.randn(1, 2, 32, 32, 32), spacing=(1.0, 1.0, 1.0))
    assert out.shape == (1, 2, 32, 32, 32)


def test_forward_rejects_physically_absurd_patch():
    net = build_dual_encoder_unet(CFG)
    with pytest.raises(ValueError, match="outside the allowed physical range"):
        net(torch.randn(1, 2, 16, 16, 16), spacing=(0.1, 0.1, 0.1))  # 1.6 mm box


def test_extent_check_is_per_batch_item_and_names_the_item():
    net = build_dual_encoder_unet(CFG)
    sp = torch.tensor([[1.0] * 3, [0.1] * 3])  # item 1 is a 1.6 mm box
    with pytest.raises(ValueError, match="batch item 1"):
        net(torch.randn(2, 2, 16, 16, 16), spacing=sp)


def test_extent_bounds_are_configurable():
    cfg = {"model": {**CFG["model"], "min_extent_mm": [1.0, 1.0, 1.0]}}
    assert build_dual_encoder_unet(cfg)(torch.randn(1, 2, 16, 16, 16), spacing=(0.1,) * 3).shape[0] == 1


def test_level_shapes_agree_with_forward():
    net = build_dual_encoder_unet(FOUR).eval()
    x = torch.randn(1, 2, 32, 32, 32)
    seen = {}
    hooks = [m.register_forward_hook(lambda _m, _i, o, k=k: seen.__setitem__(k, tuple(o.shape[2:])))
             for k, m in enumerate(net.ct_enc.stages)]
    with torch.no_grad():
        net(x)
    [h.remove() for h in hooks]
    for r in net.level_shapes((32,) * 3):
        assert seen[r["level"]] == r["shape"]
    assert [r["shape"][0] for r in net.level_shapes((96,) * 3)] == [96, 48, 24, 12, 6]
