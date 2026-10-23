"""Losses: soft Dice, Dice+CE, evidential CE and annealed KL (Sensoy et al. 2018, arXiv:1806.01768).

Conventions: predictions are (B, K, ...) tensors; labels are integer class maps
(B, 1, ...) or (B, ...). Class 0 is background, so Dice is averaged over classes
1..K-1 only (a tumour-free crop would otherwise be dominated by the background term).
Worked 3-voxel example: docs/learn/06_losses_and_metrics.md.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _onehot(labels: torch.Tensor, k: int) -> torch.Tensor:
    """(B, 1, ...) or (B, ...) integer labels -> (B, K, ...) float one-hot."""
    if labels.ndim >= 2 and labels.shape[1] == 1:
        labels = labels[:, 0]                                  # (B, ...) class indices
    return F.one_hot(labels.long(), k).movedim(-1, 1).float()  # (B, ..., K) -> (B, K, ...)


def soft_dice_loss(prob: torch.Tensor, labels: torch.Tensor, smooth: float = 0.0, eps: float = 1e-8) -> torch.Tensor:
    """1 - mean_{k>=1} 2*sum(p_k*y_k) / (sum p_k + sum y_k), sums over batch and space jointly.

    Summing over the whole batch (not per sample) keeps tumour-free samples from
    contributing an undefined 0/0 Dice. `smooth` > 0 stabilises tiny targets.
    """
    y = _onehot(labels, prob.shape[1])                              # (B, K, ...)
    dims = [0] + list(range(2, prob.ndim))                          # sum over batch and space, keep K
    inter = (prob * y).sum(dims)                                    # (K,)
    denom = prob.sum(dims) + y.sum(dims)                            # (K,)
    dice = (2 * inter + smooth) / (denom + smooth + eps)            # (K,)
    return 1.0 - dice[1:].mean()


def dice_ce_loss(logits: torch.Tensor, labels: torch.Tensor, smooth: float = 1.0,
                 lambda_dice: float = 1.0, lambda_ce: float = 1.0) -> torch.Tensor:
    """Softmax Dice + cross-entropy on raw logits (B, K, ...)."""
    lab = labels[:, 0] if labels.shape[1] == 1 and labels.ndim == logits.ndim else labels
    ce = F.cross_entropy(logits, lab.long())
    return lambda_dice * soft_dice_loss(torch.softmax(logits, 1), labels, smooth) + lambda_ce * ce


def evidential_ce(evidence: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean over voxels of sum_k y_k (digamma(S) - digamma(alpha_k)), alpha = e + 1."""
    alpha = evidence + 1.0
    s = alpha.sum(1, keepdim=True)
    y = _onehot(labels, evidence.shape[1])
    return (y * (torch.digamma(s) - torch.digamma(alpha))).sum(1).mean()


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1,...,1) ) per voxel, shape (B, ...).

    = lgamma(S) - sum lgamma(alpha_k) - lgamma(K) + sum (alpha_k - 1)(digamma(alpha_k) - digamma(S))
    """
    k = alpha.shape[1]
    s = alpha.sum(1, keepdim=True)
    log_norm = torch.lgamma(s[:, 0]) - torch.lgamma(alpha).sum(1) - torch.lgamma(torch.tensor(float(k), device=alpha.device))
    return log_norm + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(s))).sum(1)


def evidential_kl(evidence: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean KL of alpha_tilde = y + (1-y)*alpha to the uniform Dirichlet.

    alpha_tilde removes the evidence for the true class, so only evidence for
    WRONG classes is penalised (it pushes misleading evidence back to 0, i.e. u -> 1).
    """
    alpha = evidence + 1.0
    y = _onehot(labels, evidence.shape[1])
    return dirichlet_kl_to_uniform(y + (1 - y) * alpha).mean()


def kl_weight(epoch: int, anneal_epochs: int) -> float:
    """min(1, epoch / anneal_epochs); the KL term is switched on gradually so the
    network learns to fit before it is pushed towards 'I don't know'."""
    return 1.0 if anneal_epochs <= 0 else min(1.0, epoch / anneal_epochs)


def evidential_loss(evidence: torch.Tensor, labels: torch.Tensor, epoch: int, anneal_epochs: int = 10,
                    dice_weight: float = 1.0, smooth: float = 1.0) -> tuple[torch.Tensor, dict]:
    """evidential CE + annealed KL + (optional) soft Dice on p = alpha/S. Returns (loss, parts)."""
    alpha = evidence + 1.0
    prob = alpha / alpha.sum(1, keepdim=True)
    ce, kl = evidential_ce(evidence, labels), evidential_kl(evidence, labels)
    w = kl_weight(epoch, anneal_epochs)
    dice = soft_dice_loss(prob, labels, smooth) if dice_weight else torch.zeros((), device=evidence.device)
    total = ce + w * kl + dice_weight * dice
    return total, {"ce": ce.item(), "kl": kl.item(), "kl_w": w, "dice": float(dice.detach())}
