"""Phase 4 gate: every reported metric against toy volumes whose answer is known by construction,
including an anisotropic grid (HD95 and surface Dice must be in millimetres, not voxels)."""
import math

import numpy as np
import pytest

from src.metrics.calibration import brier, ece
from src.metrics.segmentation import aggregated_dice, dice, hd95, surface_dice

ANISO = (1.0, 1.0, 3.0)   # mm; z slices three times thicker


def _box(shape=(24, 24, 24), lo=(6, 6, 6), hi=(16, 16, 16)):
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


def _two_label_volume():
    lab = np.zeros((24, 24, 24), np.uint8)
    lab[2:10, 2:10, 2:10] = 1      # 512 voxels
    lab[14:20, 14:20, 14:20] = 2   # 216 voxels
    return lab


def test_dice_per_label_known_values():
    lab = _two_label_volume()
    pred = lab.copy()
    pred[2:10, 2:10, 2:6] = 0          # label 1: half of it missed -> 2*256 / (256 + 512)
    assert dice(pred == 1, lab == 1) == pytest.approx(2 * 256 / (256 + 512))
    assert dice(pred == 2, lab == 2) == 1.0
    assert dice(pred > 0, lab > 0) == pytest.approx(2 * (256 + 216) / (256 + 216 + 512 + 216))
    assert math.isnan(dice(pred == 1, np.zeros_like(lab, bool)))   # label absent -> undefined, not 0 or 1


def test_aggregated_dice_differs_from_mean_dice_and_handles_absent_label():
    a, small = _box(), _box(lo=(6, 6, 6), hi=(8, 8, 8))            # 1000 and 8 voxels
    perfect_big, missed_small = (a, a), (np.zeros_like(small), small)
    preds, labs = [perfect_big[0], missed_small[0]], [perfect_big[1], missed_small[1]]
    assert aggregated_dice(preds, labs) == pytest.approx(2 * 1000 / (1000 + 1000 + 8))   # 0.996, mean Dice is 0.5
    assert np.mean([dice(p, t) for p, t in zip(preds, labs)]) == pytest.approx(0.5)
    fp_only = (small, np.zeros_like(small))                         # label absent, 8 false-positive voxels
    assert aggregated_dice([a, fp_only[0]], [a, fp_only[1]]) == pytest.approx(2 * 1000 / (1000 + 1000 + 8))
    assert math.isnan(aggregated_dice([np.zeros_like(a)], [np.zeros_like(a)]))
    with pytest.raises(ValueError):
        aggregated_dice([a], [a, a])


def test_hd95_is_in_mm_on_an_anisotropic_grid():
    a = _box()
    assert hd95(a, a, ANISO).value == 0.0
    assert hd95(np.roll(a, 2, axis=2), a, ANISO).value == pytest.approx(6.0, abs=1.0)   # 2 voxels * 3 mm along z
    assert hd95(np.roll(a, 2, axis=0), a, ANISO).value == pytest.approx(2.0, abs=1.0)   # 2 voxels * 1 mm along x


def test_surface_dice_at_2_and_3_mm_on_an_anisotropic_grid():
    a = _box()
    x2, z1 = np.roll(a, 2, axis=0), np.roll(a, 1, axis=2)     # 2 mm shift in-plane; 3 mm shift through-plane
    assert surface_dice(a, a, ANISO, 2.0) == surface_dice(a, a, ANISO, 3.0) == 1.0
    assert surface_dice(x2, a, ANISO, 2.0) == 1.0              # every surface point within 2 mm
    assert surface_dice(z1, a, ANISO, 2.0) < 1.0               # the z faces are 3 mm away: outside 2 mm
    assert surface_dice(z1, a, ANISO, 3.0) == 1.0              # ... and inside 3 mm
    assert surface_dice(z1, a, (1.0, 1.0, 1.0), 2.0) == 1.0    # same voxel shift is only 1 mm on an isotropic grid


def test_ece_15_bins_known_value():
    # 1000 voxels, all predicted class 1 with confidence 0.8, 600 correct -> ECE = |0.6 - 0.8| = 0.2
    prob = np.stack([np.full(1000, 0.2), np.full(1000, 0.8)])
    labels = np.r_[np.ones(600), np.zeros(400)].astype(int)
    assert ece(prob, labels, n_bins=15) == pytest.approx(0.2)
    # two confidence groups in different bins: 0.5*|0.9-0.9| + 0.5*|0.5-0.7| = 0.1
    prob = np.stack([np.r_[np.full(500, 0.1), np.full(500, 0.3)], np.r_[np.full(500, 0.9), np.full(500, 0.7)]])
    labels = np.r_[np.ones(450), np.zeros(50), np.ones(250), np.zeros(250)].astype(int)
    assert ece(prob, labels, n_bins=15) == pytest.approx(0.1)


def test_brier_known_values():
    onehot = np.eye(3)[:, [0, 1, 2, 1]]                      # (K=3, 4 voxels), labels 0,1,2,1
    labels = np.array([0, 1, 2, 1])
    assert brier(onehot, labels) == 0.0
    assert brier(onehot, (labels + 1) % 3) == pytest.approx(2.0)          # confidently wrong everywhere
    assert brier(np.full((3, 4), 1 / 3), labels) == pytest.approx(2 / 3)  # uniform: (2/3)^2 + 2*(1/3)^2
    p = np.array([[0.7], [0.2], [0.1]])                                   # one voxel, true class 1
    assert brier(p, np.array([1])) == pytest.approx(0.7 ** 2 + 0.8 ** 2 + 0.1 ** 2)
    assert brier(onehot.reshape(3, 2, 2), labels.reshape(2, 2), roi=np.array([[True, False], [False, False]])) == 0.0
    assert math.isnan(brier(onehot, labels, roi=np.zeros(4, bool)))
    with pytest.raises(ValueError):
        brier(onehot, labels[:3])
