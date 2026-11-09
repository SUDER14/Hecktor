"""Baseline 2: dual-encoder CT+PET 3D segmentation network (no conditioning).

Two independent encoders, one per modality, with per-level fusion into a shared
decoder. This module deliberately has NO spacing conditioning, FiLM or uncertainty
head, so it isolates what two-stream fusion alone buys over Baseline 1 (CT only).

CT and PET share one voxel grid (HECKTOR 2026: VERIFIED identical shape, spacing and
affine for all 782 cases), so the input is one tensor (B, 2, X, Y, Z) with channel 0 =
CT and channel 1 = PET. Both encoders have the same depth and strides, so level i of
each has the same shape and fusion is concat + 1x1x1 conv at every level -- there is no
resizing and no cross-resolution bookkeeping. Output logits (B, K, X, Y, Z), no final
activation (the loss applies softmax). X, Y, Z must be divisible by prod(strides).

Foundation for the spacing conditioning added in ACF: HyperSpace (Joutard et al.,
MICCAI 2024, arXiv:2407.03681).

`spacing` (mm, (3,) or (B, 3)) is optional and only feeds the physical-extent sanity
check on the patch (src/data/geometry.py); the network itself never uses it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.blocks import UnetOutBlock, UnetResBlock, UnetrUpBlock

from src.data.geometry import check_patch_extent

_NORM = "instance"


def _down(size: int, strides) -> int:
    """Spatial size after stride-s, k=3, pad=1 convs: n -> ceil(n / s) per stage."""
    for s in strides:
        size = -(-size // s)
    return size


def _per_sample(spacing, batch: int, name: str) -> list[tuple[float, float, float]]:
    """Accept one (3,) spacing for the whole batch or a (B, 3) tensor/array."""
    arr = torch.as_tensor(spacing, dtype=torch.float64).reshape(-1, 3)
    if arr.shape[0] == 1:
        arr = arr.expand(batch, 3)
    if arr.shape[0] != batch:
        raise ValueError(f"{name} has {arr.shape[0]} rows for a batch of {batch}")
    return [tuple(r) for r in arr.tolist()]


def check_input(model, image: torch.Tensor, spacing=None) -> None:
    """Shared input validation: 2 channels, divisible size, patch extent sane in mm."""
    if image.ndim != 5 or image.shape[1] != 2:
        raise ValueError(f"expected image (B, 2, X, Y, Z) with channel 0 = CT, 1 = PET; got {tuple(image.shape)}")
    total = 1
    for s in model.strides:
        total *= s
    if any(d % total for d in image.shape[2:]):
        raise ValueError(f"spatial size {tuple(image.shape[2:])} must be divisible by {total}")
    if spacing is not None:
        for i, sp in enumerate(_per_sample(spacing, image.shape[0], "spacing")):
            check_patch_extent(image.shape[2:], sp, model.min_extent_mm, model.max_extent_mm,
                               context=f"batch item {i}")


class _Encoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], strides: list[int]):
        super().__init__()
        self.stages = nn.ModuleList()
        prev, stage_strides = in_channels, [1] + list(strides)
        for c, s in zip(channels, stage_strides):
            self.stages.append(UnetResBlock(3, prev, c, kernel_size=3, stride=s, norm_name=_NORM))
            prev = c

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class DualEncoderUNet(nn.Module):
    def __init__(self, channels, strides, out_channels: int,
                 min_extent_mm=None, max_extent_mm=None):
        super().__init__()
        from src.data.geometry import DEFAULT_MAX_EXTENT_MM, DEFAULT_MIN_EXTENT_MM

        channels, strides = list(channels), list(strides)
        if len(strides) != len(channels) - 1:
            raise ValueError(f"need len(strides) == len(channels) - 1, got {strides} vs {channels}")
        self.strides, self.channels = strides, channels
        self.min_extent_mm = min_extent_mm or DEFAULT_MIN_EXTENT_MM
        self.max_extent_mm = max_extent_mm or DEFAULT_MAX_EXTENT_MM
        self.ct_enc = _Encoder(1, channels, strides)
        self.pet_enc = _Encoder(1, channels, strides)
        self.fuse = nn.ModuleList(nn.Conv3d(2 * c, c, kernel_size=1) for c in channels)
        # decoder level i takes level i+1 output and level i fused skip
        self.ups = nn.ModuleList(
            UnetrUpBlock(3, channels[i + 1], channels[i], kernel_size=3,
                         upsample_kernel_size=strides[i], norm_name=_NORM, res_block=True)
            for i in range(len(strides))
        )
        self.out = UnetOutBlock(3, channels[0], out_channels)

    def level_shapes(self, size) -> list[dict]:
        """Per-level feature shapes (analytic, no forward pass) for a given spatial size."""
        return [{"level": i, "channels": c, "shape": tuple(_down(d, self.strides[:i]) for d in size)}
                for i, c in enumerate(self.channels)]

    def forward(self, image: torch.Tensor, spacing=None) -> torch.Tensor:
        check_input(self, image, spacing)
        ct_f = self.ct_enc(image[:, :1])   # list of (B, C_i, X/2^i, Y/2^i, Z/2^i)
        pet_f = self.pet_enc(image[:, 1:])  # same shapes: one shared grid
        fused = [f(torch.cat([c, p], dim=1)) for f, c, p in zip(self.fuse, ct_f, pet_f)]  # (B, C_i, ...)
        x = fused[-1]  # (B, C_last, X/2^last, ...) bottleneck
        for i in reversed(range(len(self.ups))):
            x = self.ups[i](x, fused[i])  # upsample to level i grid, concat skip, res block -> (B, C_i, X/2^i, ...)
        return self.out(x)  # (B, C_0, X, Y, Z) -> (B, K, X, Y, Z) logits


def build_dual_encoder_unet(config: dict) -> DualEncoderUNet:
    """config["model"] keys: channels, strides, out_channels; optional min_extent_mm /
    max_extent_mm (per-axis patch-size sanity bounds in mm)."""
    m = config["model"]
    return DualEncoderUNet(m["channels"], m["strides"], m["out_channels"],
                           m.get("min_extent_mm"), m.get("max_extent_mm"))
