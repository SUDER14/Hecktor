"""Acquisition-Conditioned Fusion (ACF): dual encoder + spacing-conditioned FiLM + evidential head.

Foundation: HyperSpace (Joutard et al., MICCAI 2024, arXiv:2407.03681) conditions a
single-modality U-Net on voxel spacing through a hypernetwork and processes volumes at
native resolution. ACF reuses that idea and adds (a) a CT stream and a PET stream with
their own encoders and their own hypernetworks, and (b) an optional evidential
(Dirichlet) output head for a per-voxel uncertainty map.

Data geometry (HECKTOR 2026, VERIFIED on all 782 cases): CT, PET and label share one
voxel grid. The input is one tensor (B, 2, X, Y, Z), channel 0 = CT, 1 = PET, and both
encoders have identical depth, so level i of each stream has the same shape; fusion is
concat + 1x1x1 conv at every level. There is no cross-resolution fusion.

What conditioning sees: the scan's voxel spacing [sx, sy, sz] (z-scored with TRAIN-ONLY
statistics). Patches are fixed in voxels at native spacing, so their extent in mm
varies with the scan -- that is the variation FiLM has to absorb. Because the two
modalities now share one grid, the CT and PET hypernetworks receive the *same* vector
and differ only in their parameters; "independent conditioning per modality" therefore
means independent weights, not different inputs.

With `conditioning_zeroed` the hypernetwork input is zeroed (ablation: same parameters,
no acquisition information). At initialisation gamma = 1, beta = 0 exactly, so the
network starts equal to the unconditioned dual encoder (tests/test_phase2_models.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.blocks import UnetOutBlock, UnetrUpBlock

from src.data.geometry import DEFAULT_MAX_EXTENT_MM, DEFAULT_MIN_EXTENT_MM
from src.models.dual_encoder_shared import _NORM, _down, _Encoder, check_input
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
        out_channels: int = 2,
        head: str = "softmax",
        cond_dim: int = 3,
        hyper_hidden: int = 64,
        conditioning_zeroed: bool = False,
        use_film: bool = True,
        tier2_kernels: bool = False,
        min_extent_mm=None,
        max_extent_mm=None,
    ):
        super().__init__()
        channels, strides = list(channels), list(strides)
        if len(strides) != len(channels) - 1:
            raise ValueError(f"need len(strides) == len(channels) - 1, got {strides} vs {channels}")
        if head not in ("softmax", "evidential"):
            raise ValueError(f"head must be 'softmax' or 'evidential', got {head!r}")
        self.channels, self.strides, self.cond_dim = channels, strides, cond_dim
        self.head_type, self.out_channels = head, out_channels
        self.conditioning_zeroed, self.use_film, self.tier2 = conditioning_zeroed, use_film, tier2_kernels
        self.min_extent_mm = min_extent_mm or DEFAULT_MIN_EXTENT_MM
        self.max_extent_mm = max_extent_mm or DEFAULT_MAX_EXTENT_MM

        self.ct_enc = _FiLMEncoder(1, channels, strides)
        self.pet_enc = _FiLMEncoder(1, channels, strides)
        # one hypernetwork per modality stream (independent weights, same spacing input)
        self.ct_hyper = FiLMHypernet(cond_dim, channels, hyper_hidden) if use_film else None
        self.pet_hyper = FiLMHypernet(cond_dim, channels, hyper_hidden) if use_film else None
        if tier2_kernels:
            self.fuse = nn.ModuleList(HyperFuse1x1(2 * c, c, cond_dim) for c in channels)
        else:
            self.fuse = nn.ModuleList(nn.Conv3d(2 * c, c, kernel_size=1) for c in channels)
        self.ups = nn.ModuleList(
            UnetrUpBlock(3, channels[i + 1], channels[i], kernel_size=3, upsample_kernel_size=strides[i],
                         norm_name=_NORM, res_block=True)
            for i in range(len(strides)))
        self.out = EvidentialHead(channels[0], out_channels) if head == "evidential" \
            else UnetOutBlock(3, channels[0], out_channels)

    def level_shapes(self, size) -> list[dict]:
        """Analytic per-level feature shapes (no forward pass), for docs and tests."""
        return [{"level": f"L{i + 1}", "channels": c, "shape": tuple(_down(d, self.strides[:i]) for d in size)}
                for i, c in enumerate(self.channels)]

    def forward(self, image, cond, spacing=None):
        """image (B,2,X,Y,Z) [CT, PET]; cond (B,3) z-scored spacing; spacing (B,3)|(3,) mm, optional
        and used only for the patch-extent sanity check.
        Returns (B, K, X, Y, Z): logits (softmax head) or non-negative evidence (evidential head)."""
        check_input(self, image, spacing)
        if cond.shape != (image.shape[0], self.cond_dim):
            raise ValueError(f"cond must be (B, {self.cond_dim}), got {tuple(cond.shape)}")
        if self.conditioning_zeroed:  # ablation: hypernet sees no acquisition info (only its bias remains)
            cond = torch.zeros_like(cond)
        ct_film = self.ct_hyper(cond) if self.use_film else None    # list of (gamma, beta), each (B, C_i)
        pet_film = self.pet_hyper(cond) if self.use_film else None

        ct_f = self.ct_enc(image[:, :1], ct_film)      # L1..L5, (B, C_i, X/2^i, Y/2^i, Z/2^i)
        pet_f = self.pet_enc(image[:, 1:], pet_film)   # same shapes: one shared grid
        skips = []
        for i, (c, p) in enumerate(zip(ct_f, pet_f)):
            cat = torch.cat([c, p], dim=1)             # (B, 2*C_i, X_i, Y_i, Z_i)
            skips.append(self.fuse[i](cat, cond) if self.tier2 else self.fuse[i](cat))  # (B, C_i, ...)
        x = skips[-1]                                  # (B, C_5, X/16, Y/16, Z/16) bottleneck
        for i in reversed(range(len(self.ups))):
            x = self.ups[i](x, skips[i])               # up to level i: (B, C_i, X/2^i, ...)
        return self.out(x)                             # (B, C_1, X, Y, Z) -> (B, K, X, Y, Z)

    @torch.no_grad()
    def predict(self, *args, **kw) -> dict[str, torch.Tensor]:
        """Probabilities p (B,K,...) and uncertainty u (B,1,...) for either head."""
        out = self.forward(*args, **kw)
        return dirichlet_summary(out) if self.head_type == "evidential" else softmax_summary(out)


def build_acf(config: dict) -> ACF:
    """config["model"] keys: channels, strides, out_channels (K), head, cond_dim, hyper_hidden,
    conditioning_zeroed, use_film, tier2_kernels, min_extent_mm, max_extent_mm."""
    m = {k: v for k, v in config["model"].items() if k != "name"}
    return ACF(**m)
