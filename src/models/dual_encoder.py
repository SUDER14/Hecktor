"""Baseline 2: dual-encoder CT+PET 3D segmentation network (no conditioning yet).

Two independent encoders, one per modality, with per-level fusion into a shared
decoder. Independent weights matter because the next step adds a separate
FiLM conditioning path per modality (the PET/CT 4-5x resolution asymmetry is
the part HyperSpace, Joutard et al. MICCAI 2024, arXiv:2407.03681, does not
cover -- it is single-modality). This module deliberately has NO spacing
conditioning, FiLM or uncertainty head, so it isolates what two-stream fusion
alone buys over Baseline 1.

PET and CT arrive on their own voxel grids (native resolution, no resampling).
At each level the PET feature map is trilinearly interpolated onto the CT
feature grid before concatenation, so the fusion point is the only place the
grids need to agree. Output logits live on the CT grid, shape
(B, out_channels, *ct_spatial), no final activation (the loss applies softmax).

The CT spatial size must be divisible by prod(strides); PET sizes need not be,
but each PET side must be >= 2*prod(strides) so the deepest level keeps >1 voxel
(instance norm raises on a single voxel; checked in forward).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetOutBlock, UnetResBlock, UnetrUpBlock

_NORM = "instance"


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
    def __init__(self, channels, strides, out_channels: int, ct_channels: int = 1, pet_channels: int = 1):
        super().__init__()
        channels, strides = list(channels), list(strides)
        if len(strides) != len(channels) - 1:
            raise ValueError(f"need len(strides) == len(channels) - 1, got {strides} vs {channels}")
        self.strides = strides
        self.ct_enc = _Encoder(ct_channels, channels, strides)
        self.pet_enc = _Encoder(pet_channels, channels, strides)
        self.fuse = nn.ModuleList(nn.Conv3d(2 * c, c, kernel_size=1) for c in channels)
        # decoder level i takes level i+1 output and level i fused skip
        self.ups = nn.ModuleList(
            UnetrUpBlock(3, channels[i + 1], channels[i], kernel_size=3,
                         upsample_kernel_size=strides[i], norm_name=_NORM, res_block=True)
            for i in range(len(strides))
        )
        self.out = UnetOutBlock(3, channels[0], out_channels)

    def forward(self, ct: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
        total = 1
        for s in self.strides:
            total *= s
        if any(d % total for d in ct.shape[2:]):
            raise ValueError(f"CT spatial size {tuple(ct.shape[2:])} must be divisible by {total}")
        if any(d < 2 * total for d in pet.shape[2:]):
            raise ValueError(f"PET spatial size {tuple(pet.shape[2:])} must be >= {2 * total} per side (instance norm needs >1 voxel at the deepest level)")
        ct_f, pet_f = self.ct_enc(ct), self.pet_enc(pet)
        fused = []
        for fuse, c, p in zip(self.fuse, ct_f, pet_f):
            if p.shape[2:] != c.shape[2:]:
                p = F.interpolate(p, size=c.shape[2:], mode="trilinear", align_corners=False)
            fused.append(fuse(torch.cat([c, p], dim=1)))
        x = fused[-1]
        for i in reversed(range(len(self.ups))):
            x = self.ups[i](x, fused[i])
        return self.out(x)


def build_dual_encoder_unet(config: dict) -> DualEncoderUNet:
    """Expected config["model"] keys: channels, strides, out_channels, and
    optionally ct_channels / pet_channels (default 1 each)."""
    m = config["model"]
    return DualEncoderUNet(m["channels"], m["strides"], m["out_channels"],
                           m.get("ct_channels", 1), m.get("pet_channels", 1))
