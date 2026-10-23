"""Evidential (Dirichlet) output head, after Sensoy et al. 2018 (arXiv:1806.01768).

Idea: instead of K softmax probabilities, the network emits K non-negative
"evidence" values e_k per voxel. They define a Dirichlet distribution over the
class probabilities:

    alpha_k = e_k + 1            (the +1 is a uniform prior: no evidence -> alpha = 1)
    S       = sum_k alpha_k      (Dirichlet strength: total evidence + K)
    p_k     = alpha_k / S        (expected class probability -> the segmentation)
    u       = K / S              (vacuity in (0, 1]: u = 1 when there is no evidence)

One forward pass therefore yields both the mask (argmax p) and an uncertainty map
(u) -- no sampling, no ensemble.

K = 2 here (background, tumour) but it is a constructor argument.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """1x1x1 conv -> K logits -> softplus -> evidence (>= 0).

    softplus (not exp) because exp makes evidence explode early in training and
    destabilises the digamma terms in the loss; softplus grows linearly.
    """

    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)  # (B, C, X, Y, Z) -> (B, K, X, Y, Z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)       # (B, C, X, Y, Z) -> (B, K, X, Y, Z)
        return F.softplus(logits)   # (B, K, X, Y, Z) evidence e_k >= 0


def dirichlet_summary(evidence: torch.Tensor) -> dict[str, torch.Tensor]:
    """evidence (B, K, ...) -> alpha, strength S, probabilities p, vacuity u.

    Shapes: alpha (B, K, ...), S (B, 1, ...), p (B, K, ...), u (B, 1, ...).
    """
    k = evidence.shape[1]
    alpha = evidence + 1.0                       # (B, K, ...) Dirichlet parameters
    strength = alpha.sum(dim=1, keepdim=True)    # (B, 1, ...) S = sum_k alpha_k
    return {"alpha": alpha, "S": strength, "p": alpha / strength, "u": k / strength}


def softmax_summary(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Same interface for a plain softmax head, so evaluation code is head-agnostic.

    u is the predictive entropy divided by log K (so it also lies in [0, 1]). It is
    NOT the same quantity as the Dirichlet vacuity and is only a baseline
    uncertainty for comparison.
    """
    k = logits.shape[1]
    p = torch.softmax(logits, dim=1)                                   # (B, K, ...)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1, keepdim=True)  # (B, 1, ...)
    return {"p": p, "u": ent / torch.log(torch.tensor(float(k)))}
