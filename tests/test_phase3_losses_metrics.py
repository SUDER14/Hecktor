"""Phase 3: losses against hand-computed values; metrics on shapes with known answers."""
import math

import numpy as np
import pytest
import torch

from src.metrics.calibration import ece, reliability_bins, reliability_diagram, uncertainty_error_overlap
from src.metrics.segmentation import dice, hd95, surface_dice
from src.training.losses import (
    dice_ce_loss, dirichlet_kl_to_uniform, evidential_ce, evidential_kl, evidential_loss, kl_weight, soft_dice_loss,
)

# Worked example (docs/learn/06): 3 voxels, K=2 (0 background, 1 tumour)
EVIDENCE = torch.tensor([[0.2, 9.0], [0.4, 0.6], [12.0, 0.1]]).T.reshape(1, 2, 3)  # (B=1, K=2, 3 voxels)
LABELS = torch.tensor([[1, 1, 0]])                                                 # tumour, tumour, background


def test_dice_loss_hand_value():
    alpha = EVIDENCE + 1
    p = alpha / alpha.sum(1, keepdim=True)
    # tumour class: inter = .8929 + .5333; sum p = 1.5042; sum y = 2 -> dice = 0.8140 -> loss 0.1860
    assert soft_dice_loss(p, LABELS).item() == pytest.approx(0.186, abs=5e-4)


def test_evidential_ce_hand_value():
    assert evidential_ce(EVIDENCE, LABELS).item() == pytest.approx(0.333, abs=5e-4)


def test_evidential_kl_hand_value_and_zero_for_correct_confident_prior():
    assert evidential_kl(EVIDENCE, LABELS).item() == pytest.approx(0.024, abs=5e-4)
    # alpha_tilde == 1 everywhere (no wrong-class evidence) -> KL to uniform is exactly 0
    ev = torch.zeros(1, 2, 3)
    ev[0, 1, :2], ev[0, 0, 2] = 5.0, 5.0
    assert evidential_kl(ev, LABELS).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_closed_form_matches_pytorch_dirichlet():
    a = torch.tensor([[2.5], [1.3], [4.0]]).T.reshape(1, 3, 1)
    ref = torch.distributions.kl_divergence(torch.distributions.Dirichlet(a[0, :, 0]),
                                            torch.distributions.Dirichlet(torch.ones(3)))
    assert dirichlet_kl_to_uniform(a).item() == pytest.approx(ref.item(), abs=1e-5)


def test_kl_weight_anneals():
    assert kl_weight(0, 10) == 0 and kl_weight(5, 10) == 0.5 and kl_weight(10, 10) == 1 and kl_weight(99, 10) == 1


def test_evidential_loss_total_and_gradient():
    ev = EVIDENCE.clone().requires_grad_(True)
    total, parts = evidential_loss(ev, LABELS, epoch=5, anneal_epochs=10, dice_weight=1.0, smooth=0.0)
    assert total.item() == pytest.approx(0.3333 + 0.5 * 0.0236 + 0.186, abs=2e-3)
    assert parts["kl_w"] == 0.5
    total.backward()
    assert ev.grad.abs().sum() > 0


def test_dice_ce_perfect_and_worst():
    labels = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)
    labels[..., :2, :, :] = 1
    perfect = torch.zeros(1, 2, 4, 4, 4)
    perfect[:, 1, :2] = 20.0
    perfect[:, 0, 2:] = 20.0
    assert dice_ce_loss(perfect, labels, smooth=1e-6).item() < 1e-3
    assert dice_ce_loss(-perfect, labels, smooth=1e-6).item() > 1.5


def test_dice_loss_tumour_free_batch_is_finite_with_smooth():
    labels = torch.zeros(2, 1, 4, 4, 4, dtype=torch.long)
    p = torch.full((2, 2, 4, 4, 4), 0.0)
    p[:, 0] = 1.0
    assert math.isfinite(soft_dice_loss(p, labels, smooth=1.0).item())
    assert soft_dice_loss(p, labels, smooth=1.0).item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- segmentation metrics
def _cube(n=20, lo=5, hi=15):
    m = np.zeros((n, n, n), bool)
    m[lo:hi, lo:hi, lo:hi] = True
    return m


def test_dice_known_and_empty_conventions():
    a, b = _cube(), _cube(hi=10)
    assert dice(a, a) == 1.0
    assert dice(a, b) == pytest.approx(2 * 125 / (1000 + 125))
    assert math.isnan(dice(a, np.zeros_like(a)))      # empty reference -> NaN
    assert dice(np.zeros_like(a), a) == 0.0           # empty prediction -> 0


def test_hd95_in_mm_with_anisotropic_spacing():
    a = _cube()
    shifted = np.roll(a, 3, axis=2)                   # shift 3 voxels along z
    r = hd95(shifted, a, (1.0, 1.0, 2.0))
    assert r.value == pytest.approx(6.0, abs=0.5) and not r.pred_empty  # 3 voxels * 2 mm
    assert math.isnan(hd95(a, np.zeros_like(a), (1, 1, 1)).value)
    e = hd95(np.zeros_like(a), a, (1, 1, 1))
    assert e.pred_empty and e.value == pytest.approx(math.sqrt(3) * 20)  # volume diagonal penalty


def test_surface_dice_tolerance():
    a = _cube()
    shifted = np.roll(a, 2, axis=0)
    assert surface_dice(a, a, (1, 1, 1), 1.0) == 1.0
    assert surface_dice(shifted, a, (1, 1, 1), 1.0) < surface_dice(shifted, a, (1, 1, 1), 3.0) == 1.0
    assert surface_dice(shifted, a, (2, 1, 1), 3.0) < 1.0  # same shift is 4 mm along x at 2 mm spacing
    assert math.isnan(surface_dice(a, np.zeros_like(a), (1, 1, 1), 2.0))
    assert surface_dice(np.zeros_like(a), a, (1, 1, 1), 2.0) == 0.0


# ---------------------------------------------------------------- calibration
def test_ece_perfectly_calibrated_and_overconfident():
    rng = np.random.default_rng(0)
    n = 200_000
    conf = rng.uniform(0.5, 1.0, n)
    correct = rng.random(n) < conf                         # accuracy == confidence by construction
    prob = np.stack([1 - conf, conf])
    labels = np.where(correct, 1, 0)                       # argmax is class 1 (conf >= .5)
    assert ece(prob, labels) < 0.01
    # always 99% confident, 50% right -> ECE ~ 0.49
    prob2 = np.stack([np.full(n, 0.01), np.full(n, 0.99)])
    labels2 = (rng.random(n) < 0.5).astype(int)
    assert ece(prob2, labels2) == pytest.approx(0.49, abs=0.01)


def test_ece_bins_15_and_roi_and_diagram(tmp_path):
    prob = np.stack([np.full((4, 4), 0.2), np.full((4, 4), 0.8)])
    labels = np.ones((4, 4), int)
    b = reliability_bins(prob, labels)
    assert len(b["count"]) == 15 and b["count"].sum() == 16
    assert ece(prob, labels) == pytest.approx(0.2, abs=1e-9)  # conf .8, acc 1
    roi = np.zeros((4, 4), bool)
    assert math.isnan(ece(prob, labels, roi=roi))
    out = tmp_path / "rel.png"
    reliability_diagram(prob, labels, str(out), title="t")
    assert out.stat().st_size > 1000


def test_uncertainty_error_overlap():
    rng = np.random.default_rng(1)
    label = (rng.random((20, 20, 20)) < 0.3).astype(int)
    pred = label.copy()
    flip = rng.random(label.shape) < 0.1
    pred[flip] = 1 - pred[flip]
    informative = flip * 0.9 + 0.05 * rng.random(label.shape)   # high where wrong
    r = uncertainty_error_overlap(informative, pred, label)
    assert r["overlap_dice"] > 0.99 and r["auroc"] > 0.99 and r["mean_u_error"] > r["mean_u_correct"]
    r2 = uncertainty_error_overlap(rng.random(label.shape), pred, label)
    assert abs(r2["auroc"] - 0.5) < 0.05
    assert math.isnan(uncertainty_error_overlap(informative, label, label)["auroc"])  # no errors -> NaN
