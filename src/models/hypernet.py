"""Acquisition hypernetwork and FiLM modulation.

Prior art: conditioning a segmentation U-Net on voxel spacing with a hypernetwork
is HyperSpace (Joutard et al., MICCAI 2024, arXiv:2407.03681). FiLM is Perez et
al. 2018 (arXiv:1709.07871); hypernetworks are Ha et al. 2016 (arXiv:1609.09106).
What this module adds is only the *per-modality independent* use (PET and CT each
get their own hypernetwork) -- see CLAUDE.md for the scope of that claim.

Tier 1 (default): an MLP maps a conditioning vector v = [sx, sy, sz, flag] to a
(gamma, beta) pair for every encoder block of ONE modality. FiLM:

        F' = gamma * F + beta         F: (B, C, X, Y, Z); gamma, beta: (B, C)

with gamma = 1 + delta_gamma, beta = delta_beta. The last MLP layer is zero
initialised, so at initialisation gamma = 1, beta = 0 exactly and the network
starts as the unconditioned dual encoder: conditioning can only be learned, never
imposed by a random init.

Why zero-init does not block learning: the last layer receives gradient from step
0; once its weights move off zero, gradient flows into the hidden layers too.

Tier 2 (flag, off by default): HyperFuse1x1 generates a low-rank update to the
1x1x1 fusion kernels directly from the joint conditioning vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLMHypernet(nn.Module):
    """MLP: (B, cond_dim) -> per-block (gamma, beta) for one modality's encoder."""

    def __init__(self, cond_dim: int, block_channels: list[int], hidden: int = 64):
        super().__init__()
        self.block_channels = list(block_channels)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head = nn.Linear(hidden, 2 * sum(block_channels))  # gamma and beta delta for every channel
        nn.init.zeros_(self.head.weight)  # identity at init: gamma = 1, beta = 0
        nn.init.zeros_(self.head.bias)

    def forward(self, cond: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        h = self.head(self.mlp(cond))               # (B, cond_dim) -> (B, hidden) -> (B, 2*sumC)
        delta_g, delta_b = h.chunk(2, dim=1)        # two (B, sumC)
        gammas = delta_g.split(self.block_channels, dim=1)  # tuple of (B, C_i)
        betas = delta_b.split(self.block_channels, dim=1)   # tuple of (B, C_i)
        return [(1.0 + g, b) for g, b in zip(gammas, betas)]  # gamma = 1 + delta


def film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """F' = gamma * F + beta, with (B, C) parameters broadcast over (X, Y, Z)."""
    return x * gamma[:, :, None, None, None] + beta[:, :, None, None, None]  # (B,C,X,Y,Z)


class HyperFuse1x1(nn.Module):
    """Tier 2: 1x1x1 fusion conv whose kernel is generated from conditioning.

    W_b = W_base + A_b @ B_b, A_b: (Cout, r), B_b: (r, Cin) emitted by an MLP from
    the joint vector [v_ct, v_pet]. Low rank keeps the generator small (a full
    Cout x Cin output at 256 channels would be 131k numbers per sample). The
    generator's last layer is zero-initialised, so it starts equal to a plain conv.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, rank: int = 4, hidden: int = 64):
        super().__init__()
        self.in_ch, self.out_ch, self.rank = in_ch, out_ch, rank
        self.base = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.gen = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, rank * (out_ch + in_ch)))
        # delta = A @ C must start at 0, but zeroing BOTH factors also zeroes both gradients
        # (d delta/dA = C = 0, d delta/dC = A = 0) and it could never train. So only the
        # C half is zero; A gets small random weights, giving C a non-zero gradient.
        nn.init.zeros_(self.gen[-1].weight)
        nn.init.zeros_(self.gen[-1].bias)
        with torch.no_grad():
            self.gen[-1].weight[: out_ch * rank].normal_(0, 0.02)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        g = self.gen(cond)                                            # (B, r*(Cout+Cin))
        a = g[:, : self.out_ch * self.rank].view(b, self.out_ch, self.rank)   # (B, Cout, r)
        c = g[:, self.out_ch * self.rank:].view(b, self.rank, self.in_ch)     # (B, r, Cin)
        delta = torch.einsum("bor,bri->boi", a, c)                    # (B, Cout, Cin) per-sample kernel update
        y = self.base(x)                                              # (B, Cin, ...) -> (B, Cout, ...)
        return y + torch.einsum("boi,bixyz->boxyz", delta, x)         # add the generated 1x1x1 conv
