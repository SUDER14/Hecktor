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

The CT spatial size must be divisible by prod(strides). PET sizes need not be,
but the PET deepest level must keep >1 voxel per axis (instance norm raises on a
single voxel). That is computed exactly (stride-2 conv: n -> ceil(n/2)), not by a
fixed minimum side, so e.g. a 24^3 PET patch is fine.

Whether a PET patch pairs with a CT patch is a physical question, not a voxel
count one: pass ct_spacing / pet_spacing (mm) to forward() and the two patch
extents in mm are compared (src/data/geometry.py). Without spacings the check is
skipped, since the model cannot know them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetOutBlock, UnetResBlock, UnetrUpBlock

from src.data.geometry import check_physical_extent

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
    def __init__(self, channels, strides, out_channels: int, ct_channels: int = 1, pet_channels: int = 1,
                 extent_tol_voxels: float = 1.0):
        super().__init__()
        self.extent_tol_voxels = extent_tol_voxels
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

    @staticmethod
    def _per_sample(spacing, batch: int, name: str) -> list[tuple[float, float, float]]:
        """Accept one (3,) spacing for the whole batch or a (B, 3) tensor/array."""
        arr = torch.as_tensor(spacing, dtype=torch.float64).reshape(-1, 3)
        if arr.shape[0] == 1:
            arr = arr.expand(batch, 3)
        if arr.shape[0] != batch:
            raise ValueError(f"{name} has {arr.shape[0]} rows for a batch of {batch}")
        return [tuple(r) for r in arr.tolist()]

    def forward(self, ct: torch.Tensor, pet: torch.Tensor, ct_spacing=None, pet_spacing=None) -> torch.Tensor:
        total = 1
        for s in self.strides:
            total *= s
        if any(d % total for d in ct.shape[2:]):
            raise ValueError(f"CT spatial size {tuple(ct.shape[2:])} must be divisible by {total}")
        deepest = []
        for d in pet.shape[2:]:
            for s in self.strides:
                d = -(-d // s)  # stride-s, k=3, pad=1 conv: ceil(d / s)
            deepest.append(d)
        if any(d < 2 for d in deepest):
            raise ValueError(f"PET spatial size {tuple(pet.shape[2:])} collapses to {tuple(deepest)} at the deepest level (instance norm needs >1 voxel per axis)")
        if (ct_spacing is None) != (pet_spacing is None):
            raise ValueError("pass both ct_spacing and pet_spacing, or neither")
        if ct_spacing is not None:
            batch = ct.shape[0]
            for i, (cs, ps) in enumerate(zip(self._per_sample(ct_spacing, batch, "ct_spacing"),
                                             self._per_sample(pet_spacing, batch, "pet_spacing"))):
                check_physical_extent(ct.shape[2:], cs, pet.shape[2:], ps,
                                      self.extent_tol_voxels, context=f"batch item {i}")
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
    optionally ct_channels / pet_channels (default 1 each) and extent_tol_voxels
    (CT/PET physical-extent tolerance in coarser voxels, default 1.0)."""
    m = config["model"]
    return DualEncoderUNet(m["channels"], m["strides"], m["out_channels"],
                           m.get("ct_channels", 1), m.get("pet_channels", 1),
                           m.get("extent_tol_voxels", 1.0))
