"""Acquisition-Conditioned Fusion (ACF): native-spacing dual encoder + FiLM + evidential head.

Foundation: HyperSpace (Joutard et al., MICCAI 2024, arXiv:2407.03681) conditions a
single-modality U-Net on voxel spacing. ACF applies conditioning to each PET/CT
encoder independently and fuses them at physically matching depths.

Geometry (the part that is easy to get wrong). With a 96 mm box, CT at 1 mm and PET
at 4 mm, and four stride-2 downsamplings:

    CT  level  L1   L2   L3   L4   L5        voxel   1    2    4    8   16 mm
        shape  96^3 48^3 24^3 12^3  6^3
    PET level            P1   P2   P3        voxel             4    8   16 mm
        shape            24^3 12^3  6^3

PET level P1 has the same voxel size as CT level L3, so PET is fused at L3, L4, L5
(`pet_offset = 2`, i.e. log2(4 mm / 1 mm)); L1 and L2 are CT-only skips. Where the
voxel sizes are not exactly equal (anisotropic 1x1x3 CT, 3 mm PET slices) the PET
feature map is resized onto the CT feature grid -- but only after checking that the
two maps cover the same physical extent (patch_mm / voxel_mm), failing loudly if not.

Module names deliberately mirror DualEncoderUNet (ct_enc, pet_enc, fuse, ups, out):
with pet_offset=0 and the same depths, ACF with `conditioning_zeroed` is that baseline
plus identity-initialised FiLM, which tests/test_phase2_models.py asserts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetOutBlock, UnetrUpBlock

from src.data.geometry import check_physical_extent
from src.models.dual_encoder import _NORM, DualEncoderUNet, _down, _Encoder
from src.models.evidential import EvidentialHead, dirichlet_summary, softmax_summary
from src.models.hypernet import FiLMHypernet, HyperFuse1x1, film


class _FiLMEncoder(_Encoder):
    """_Encoder whose every block output is modulated by FiLM (gamma, beta) of shape (B, C)."""

    def forward(self, x, film_params=None):  # type: ignore[override]
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)                          # (B, C_{i-1}, X, Y, Z) -> (B, C_i, X/s, Y/s, Z/s)
            if film_params is not None:
                x = film(x, *film_params[i])      # (B, C_i, ...) unchanged shape; gamma*x + beta
            feats.append(x)
        return feats


class ACF(nn.Module):
    def __init__(
        self,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        pet_channels=(16, 32, 64),
        pet_offset: int = 2,
        out_channels: int = 2,
        head: str = "softmax",
        cond_dim: int = 4,
        hyper_hidden: int = 64,
        conditioning_zeroed: bool = False,
        use_film: bool = True,
        tier2_kernels: bool = False,
        extent_tol_voxels: float = 1.0,
    ):
        super().__init__()
        channels, strides, pet_channels = list(channels), list(strides), list(pet_channels)
        if len(strides) != len(channels) - 1:
            raise ValueError(f"need len(strides) == len(channels) - 1, got {strides} vs {channels}")
        if not 0 <= pet_offset or pet_offset + len(pet_channels) > len(channels):
            raise ValueError(f"PET levels {len(pet_channels)} at offset {pet_offset} do not fit in "
                             f"{len(channels)} CT levels")
        if head not in ("softmax", "evidential"):
            raise ValueError(f"head must be 'softmax' or 'evidential', got {head!r}")
        self.channels, self.strides, self.pet_channels = channels, strides, pet_channels
        self.pet_offset, self.pet_levels = pet_offset, len(pet_channels)
        self.head_type, self.out_channels = head, out_channels
        self.conditioning_zeroed, self.use_film, self.tier2 = conditioning_zeroed, use_film, tier2_kernels
        self.extent_tol_voxels = extent_tol_voxels

        self.ct_enc = _FiLMEncoder(1, channels, strides)
        self.pet_enc = _FiLMEncoder(1, pet_channels, [2] * (self.pet_levels - 1))
        # PET encoder always halves per level; CT strides need not (checked via pet_to_ct_ratio below)
        if any(s != 2 for s in strides[pet_offset:pet_offset + self.pet_levels - 1]):
            raise ValueError("fused CT levels must use stride 2 to line up with the PET encoder")

        # Tier 1: one hypernetwork per modality (independent conditioning)
        self.ct_hyper = FiLMHypernet(cond_dim, channels, hyper_hidden) if use_film else None
        self.pet_hyper = FiLMHypernet(cond_dim, pet_channels, hyper_hidden) if use_film else None

        fused = set(range(pet_offset, pet_offset + self.pet_levels))
        if tier2_kernels:
            self.fuse = nn.ModuleList(HyperFuse1x1(2 * c, c, 2 * cond_dim) if i in fused else nn.Identity()
                                      for i, c in enumerate(channels))
        else:
            self.fuse = nn.ModuleList(nn.Conv3d(2 * c, c, kernel_size=1) if i in fused else nn.Identity()
                                      for i, c in enumerate(channels))
        self.pet_proj = nn.ModuleList(  # PET channels -> CT channels at that level, so concat is 2*c
            nn.Conv3d(pc, channels[pet_offset + j], kernel_size=1) if pc != channels[pet_offset + j] else nn.Identity()
            for j, pc in enumerate(pet_channels))
        self.ups = nn.ModuleList(
            UnetrUpBlock(3, channels[i + 1], channels[i], kernel_size=3, upsample_kernel_size=strides[i],
                         norm_name=_NORM, res_block=True)
            for i in range(len(strides)))
        self.out = EvidentialHead(channels[0], out_channels) if head == "evidential" \
            else UnetOutBlock(3, channels[0], out_channels)

    # ------------------------------------------------------------------ geometry
    def _check_level(self, j: int, ct_f, pet_f, ct_sp, pet_sp, ctx: str) -> None:
        """Feature maps fused at CT level L = offset + j must cover the same mm extent.

        A CT level-L voxel is ct_sp * prod(strides[:L]); a PET level-j voxel is pet_sp * 2^j.
        """
        lvl = self.pet_offset + j
        ct_vox = [s * _prod(self.strides[:lvl]) for s in ct_sp]
        pet_vox = [s * 2 ** j for s in pet_sp]
        check_physical_extent(ct_f.shape[2:], ct_vox, pet_f.shape[2:], pet_vox, self.extent_tol_voxels,
                              context=f"{ctx} level L{lvl + 1} vs P{j + 1}")

    def level_table(self, ct_size, pet_size, ct_spacing, pet_spacing) -> list[dict]:
        """Analytic per-level shapes and voxel sizes (no forward pass), for docs and tests."""
        rows = []
        for i, c in enumerate(self.channels):
            j = i - self.pet_offset
            has_pet = 0 <= j < self.pet_levels
            rows.append({
                "level": f"L{i + 1}", "channels": c,
                "ct_shape": tuple(_down(d, self.strides[:i]) for d in ct_size),
                "ct_voxel_mm": tuple(s * _prod(self.strides[:i]) for s in ct_spacing),
                "pet_shape": tuple(_down(d, [2] * j) for d in pet_size) if has_pet else None,
                "pet_voxel_mm": tuple(s * 2 ** j for s in pet_spacing) if has_pet else None,
                "fusion": f"P{j + 1}+L{i + 1}" if has_pet else "CT only"})
        return rows

    # ------------------------------------------------------------------ forward
    def forward(self, ct, pet, cond_ct, cond_pet, ct_spacing, pet_spacing):
        """ct (B,1,X,Y,Z); pet (B,1,x,y,z); cond_* (B,4); *_spacing (B,3) or (3,) in mm.
        Returns (B, K, X, Y, Z): logits (softmax head) or non-negative evidence (evidential head)."""
        b = ct.shape[0]
        total = _prod(self.strides)
        if any(d % total for d in ct.shape[2:]):
            raise ValueError(f"CT spatial size {tuple(ct.shape[2:])} must be divisible by {total}")
        ct_sp = DualEncoderUNet._per_sample(ct_spacing, b, "ct_spacing")   # list of B (sx,sy,sz)
        pet_sp = DualEncoderUNet._per_sample(pet_spacing, b, "pet_spacing")
        for i in range(b):  # input-level physical extent check, per sample
            check_physical_extent(ct.shape[2:], ct_sp[i], pet.shape[2:], pet_sp[i],
                                  self.extent_tol_voxels, context=f"batch item {i}")
        if self.conditioning_zeroed:  # ablation: hypernet sees no acquisition info (only its bias remains)
            cond_ct, cond_pet = torch.zeros_like(cond_ct), torch.zeros_like(cond_pet)
        ct_film = self.ct_hyper(cond_ct) if self.use_film else None        # list of (gamma, beta), each (B, C_i)
        pet_film = self.pet_hyper(cond_pet) if self.use_film else None

        ct_f = self.ct_enc(ct, ct_film)        # list of L1..L5 maps, (B, C_i, X/2^i, Y/2^i, Z/2^i)
        pet_f = self.pet_enc(pet, pet_film)    # list of P1..P3 maps, (B, Cp_j, x/2^j, y/2^j, z/2^j)

        joint = torch.cat([cond_ct, cond_pet], dim=1)  # (B, 8) used only by tier-2 kernels
        skips = []
        for i, c in enumerate(ct_f):
            j = i - self.pet_offset
            if not 0 <= j < self.pet_levels:
                skips.append(c)                                   # L1, L2: CT-only skip
                continue
            p = self.pet_proj[j](pet_f[j])                        # (B, Cp_j, ...) -> (B, C_i, ...)
            for s in range(b):
                self._check_level(j, c[s:s + 1], p[s:s + 1], ct_sp[s], pet_sp[s], f"batch item {s}")
            if p.shape[2:] != c.shape[2:]:
                p = F.interpolate(p, size=c.shape[2:], mode="trilinear", align_corners=False)  # onto CT grid
            cat = torch.cat([c, p], dim=1)                        # (B, 2*C_i, X_i, Y_i, Z_i)
            skips.append(self.fuse[i](cat, joint) if self.tier2 else self.fuse[i](cat))  # (B, C_i, X_i, Y_i, Z_i)
        x = skips[-1]                                             # (B, C_5, X/16, Y/16, Z/16) bottleneck
        for i in reversed(range(len(self.ups))):
            x = self.ups[i](x, skips[i])                          # up to level i: (B, C_i, X/2^i, ...)
        return self.out(x)                                        # (B, C_1, X, Y, Z) -> (B, K, X, Y, Z)

    @torch.no_grad()
    def predict(self, *args, **kw) -> dict[str, torch.Tensor]:
        """Probabilities p (B,K,...) and uncertainty u (B,1,...) for either head."""
        out = self.forward(*args, **kw)
        return dirichlet_summary(out) if self.head_type == "evidential" else softmax_summary(out)


def _prod(xs) -> int:
    n = 1
    for x in xs:
        n *= x
    return n


def build_acf(config: dict) -> ACF:
    """config["model"] keys: channels, strides, pet_channels, pet_offset, out_channels, head,
    hyper_hidden, conditioning_zeroed, use_film, tier2_kernels, extent_tol_voxels."""
    m = {k: v for k, v in config["model"].items() if k != "name"}
    return ACF(**m)
