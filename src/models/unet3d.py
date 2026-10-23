"""B1: single-modality 3D U-Net (CT only) for resampled input.

Plain on purpose -- no PET, no conditioning, no uncertainty head. It is the floor
every other model is compared to. Built from the same MONAI blocks as the dual
encoder (UnetResBlock encoder, UnetrUpBlock decoder) so differences between B1, B2
and ACF come from the design, not from different building blocks.

Input  ct: (B, 1, X, Y, Z), X/Y/Z divisible by prod(strides).
Output   : (B, K, X, Y, Z) raw logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.blocks import UnetOutBlock, UnetrUpBlock

from src.models.dual_encoder import _NORM, _Encoder


class UNet3D(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), out_channels: int = 2,
                 in_channels: int = 1):
        super().__init__()
        channels, strides = list(channels), list(strides)
        if len(strides) != len(channels) - 1:
            raise ValueError(f"need len(strides) == len(channels) - 1, got {strides} vs {channels}")
        self.channels, self.strides = channels, strides
        self.enc = _Encoder(in_channels, channels, strides)
        self.ups = nn.ModuleList(
            UnetrUpBlock(3, channels[i + 1], channels[i], kernel_size=3, upsample_kernel_size=strides[i],
                         norm_name=_NORM, res_block=True)
            for i in range(len(strides)))
        self.out = UnetOutBlock(3, channels[0], out_channels)

    def forward(self, ct: torch.Tensor, *_unused, **__unused) -> torch.Tensor:
        total = 1
        for s in self.strides:
            total *= s
        if any(d % total for d in ct.shape[2:]):
            raise ValueError(f"spatial size {tuple(ct.shape[2:])} must be divisible by {total}")
        feats = self.enc(ct)             # [(B,16,X,..), (B,32,X/2,..), (B,64,X/4,..), (B,128,X/8,..), (B,256,X/16,..)]
        x = feats[-1]                    # (B, 256, X/16, Y/16, Z/16) bottleneck
        for i in reversed(range(len(self.ups))):
            x = self.ups[i](x, feats[i])  # upsample x to level i's grid, concat skip, res block -> (B, C_i, X/2^i, ..)
        return self.out(x)               # (B, 16, X, Y, Z) -> (B, K, X, Y, Z)


def build_unet3d(config: dict) -> UNet3D:
    m = {k: v for k, v in config["model"].items() if k != "name"}
    return UNet3D(**m)
