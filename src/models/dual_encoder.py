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
grids need to agree. The PET encoder is shallower than the CT one (pet_depth);
levels below it are CT-only. print(net.describe(ct_size, pet_size)) lists the shapes. Output logits live on the CT grid, shape
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


def _down(size: int, strides) -> int:
    """Spatial size after stride-s, k=3, pad=1 convs: n -> ceil(n / s) per stage."""
    for s in strides:
        size = -(-size // s)
    return size


def default_pet_depth(strides, ref_size: int = 24, min_size: int = 3) -> int:
    """Most downsampling steps (<= len(strides)) that keep a `ref_size` PET side at
    >= `min_size` voxels. 24 / 3 with four stride-2 stages gives 3: 24->12->6->3."""
    depth = 0
    for d in range(1, len(strides) + 1):
        if _down(ref_size, strides[:d]) >= min_size:
            depth = d
    return depth


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
                 extent_tol_voxels: float = 1.0, pet_depth: int | None = None,
                 pet_ref_size: int = 24, pet_min_size: int = 3):
        super().__init__()
        self.extent_tol_voxels = extent_tol_voxels
        channels, strides = list(channels), list(strides)
        if len(strides) != len(channels) - 1:
            raise ValueError(f"need len(strides) == len(channels) - 1, got {strides} vs {channels}")
        self.strides, self.channels = strides, channels
        self.ct_enc = _Encoder(ct_channels, channels, strides)
        # PET is coarse (4-5x), so it gets fewer downsampling steps than CT: by default
        # as many as keep a 24-voxel PET side at >= 3 voxels. Levels deeper than the
        # PET encoder are CT-only (Identity in `fuse`).
        if pet_depth is None:
            pet_depth = default_pet_depth(strides, pet_ref_size, pet_min_size)
        if not 0 <= pet_depth <= len(strides):
            raise ValueError(f"pet_depth must be in [0, {len(strides)}], got {pet_depth}")
        self.pet_depth = pet_depth
        self.pet_enc = _Encoder(pet_channels, channels[: pet_depth + 1], strides[:pet_depth])
        self.fuse = nn.ModuleList(
            nn.Conv3d(2 * c, c, kernel_size=1) if i <= pet_depth else nn.Identity()
            for i, c in enumerate(channels))
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

    def level_shapes(self, ct_size, pet_size) -> list[dict]:
        """Per-level feature shapes (analytic, no forward pass) for given spatial sizes."""
        rows = []
        for i, c in enumerate(self.channels):
            ct_s = tuple(_down(d, self.strides[:i]) for d in ct_size)
            pet_s = tuple(_down(d, self.strides[:i]) for d in pet_size) if i <= self.pet_depth else None
            rows.append({"level": i, "channels": c, "ct": ct_s, "pet": pet_s,
                         "fused": ct_s, "fusion": "concat+1x1" if pet_s else "CT only"})
        return rows

    def describe(self, ct_size, pet_size) -> str:
        lines = [f"pet_depth={self.pet_depth} of {len(self.strides)} downsampling steps",
                 f"{'lvl':>3} {'ch':>4}  {'CT':>14}  {'PET':>14}  fusion"]
        for r in self.level_shapes(ct_size, pet_size):
            pet = "x".join(map(str, r["pet"])) if r["pet"] else "-"
            lines.append(f"{r['level']:>3} {r['channels']:>4}  {'x'.join(map(str, r['ct'])):>14}  {pet:>14}  {r['fusion']}")
        return "\n".join(lines)

    def forward(self, ct: torch.Tensor, pet: torch.Tensor, ct_spacing=None, pet_spacing=None) -> torch.Tensor:
        total = 1
        for s in self.strides:
            total *= s
        if any(d % total for d in ct.shape[2:]):
            raise ValueError(f"CT spatial size {tuple(ct.shape[2:])} must be divisible by {total}")
        deepest = [_down(d, self.strides[: self.pet_depth]) for d in pet.shape[2:]]
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
        ct_f, pet_f = self.ct_enc(ct), self.pet_enc(pet)  # lists of (B, C_i, X/2^i, Y/2^i, Z/2^i) maps (PET list is shorter)
        fused = []
        for i, (fuse, c) in enumerate(zip(self.fuse, ct_f)):
            if i > self.pet_depth:  # no PET feature at this depth
                fused.append(c)
                continue
            p = pet_f[i]  # (B, C_i, x/2^i, y/2^i, z/2^i): PET map at the same level index
            if p.shape[2:] != c.shape[2:]:
                p = F.interpolate(p, size=c.shape[2:], mode="trilinear", align_corners=False)  # -> (B, C_i, X_i, Y_i, Z_i) on the CT grid
            fused.append(fuse(torch.cat([c, p], dim=1)))  # cat (B, 2*C_i, ...) -> 1x1x1 conv -> (B, C_i, X_i, Y_i, Z_i)
        x = fused[-1]  # (B, C_last, X/2^last, ...) bottleneck
        for i in reversed(range(len(self.ups))):
            x = self.ups[i](x, fused[i])  # upsample to level i grid, concat skip, res block -> (B, C_i, X/2^i, ...)
        return self.out(x)  # (B, C_0, X, Y, Z) -> (B, K, X, Y, Z) logits


def build_dual_encoder_unet(config: dict) -> DualEncoderUNet:
    """Expected config["model"] keys: channels, strides, out_channels, and
    optionally ct_channels / pet_channels (default 1 each) and extent_tol_voxels
    (CT/PET physical-extent tolerance in coarser voxels, default 1.0), pet_depth
    (PET downsampling steps; default keeps pet_ref_size=24 at >= pet_min_size=3)."""
    m = config["model"]
    return DualEncoderUNet(m["channels"], m["strides"], m["out_channels"],
                           m.get("ct_channels", 1), m.get("pet_channels", 1),
                           m.get("extent_tol_voxels", 1.0), m.get("pet_depth"),
                           m.get("pet_ref_size", 24), m.get("pet_min_size", 3))
