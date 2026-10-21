import math

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from src.eval.metrics import hd95, surface_voxels


def _ball(shape, centre, radius, spacing):
    """Ball of physical radius `radius` mm (an ellipsoid in voxel space when spacing is anisotropic)."""
    grids = np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing)], indexing="ij")
    d2 = sum((g - c * s) ** 2 for g, c, s in zip(grids, centre, spacing))
    return d2 <= radius**2


def test_identical_masks_give_zero():
    m = _ball((40, 40, 40), (20, 20, 20), 10, (1, 1, 1))
    r = hd95(m, m, (1, 1, 1))
    assert r.value == 0.0 and not r.pred_empty


def test_translated_spheres_isotropic():
    # Translation by d: surface distance at polar angle t is ~d*|cos t|; |cos t| is uniform on a
    # sphere, so the 95th percentile is ~0.95*d. Tolerance covers voxelisation (~1 voxel).
    d = 10
    a = _ball((90, 90, 90), (35, 45, 45), 20, (1, 1, 1))
    b = _ball((90, 90, 90), (35 + d, 45, 45), 20, (1, 1, 1))
    assert hd95(a, b, (1, 1, 1)).value == pytest.approx(0.95 * d, abs=1.0)


def test_translated_spheres_anisotropic_spacing_scales_the_shift_axis():
    # Same 10-voxel shift along axis 0 where a voxel is 2.5 mm: physical shift 25 mm.
    sp = (2.5, 1.0, 1.0)
    d_vox = 10
    a = _ball((60, 90, 90), (20, 45, 45), 20, sp)
    b = _ball((60, 90, 90), (20 + d_vox, 45, 45), 20, sp)
    assert hd95(a, b, sp).value == pytest.approx(0.95 * d_vox * 2.5, abs=1.5)
    # ...and the same voxel shift along a 1 mm axis stays ~0.95*10 mm.
    sp2 = (1.0, 2.5, 1.0)
    a2 = _ball((90, 60, 90), (45, 20, 45), 20, sp2)
    b2 = _ball((90, 60, 90), (45 + d_vox, 20, 45), 20, sp2)
    assert hd95(a2, b2, sp2).value == pytest.approx(0.95 * d_vox * 1.0, abs=1.5)


def test_matches_brute_force_on_random_blobs_with_anisotropy():
    rng = np.random.default_rng(0)
    sp = (3.0, 1.0, 2.0)
    shape = (14, 16, 12)
    for _ in range(3):
        a = _ball(shape, tuple(rng.integers(4, 9, 3)), rng.uniform(3, 5), (1, 1, 1))
        b = _ball(shape, tuple(rng.integers(4, 9, 3)), rng.uniform(3, 5), (1, 1, 1))
        pa = np.argwhere(surface_voxels(a)) * np.array(sp)
        pb = np.argwhere(surface_voxels(b)) * np.array(sp)
        dm = cdist(pa, pb)
        expect = max(np.percentile(dm.min(1), 95), np.percentile(dm.min(0), 95))
        assert hd95(a, b, sp).value == pytest.approx(expect, rel=1e-9)


def test_empty_label_is_nan():
    m = _ball((20, 20, 20), (10, 10, 10), 5, (1, 1, 1))
    r = hd95(m, np.zeros_like(m), (1, 1, 1))
    assert math.isnan(r.value) and not r.pred_empty
    assert math.isnan(hd95(np.zeros_like(m), np.zeros_like(m), (1, 1, 1)).value)


def test_empty_prediction_gets_documented_penalty_and_flag():
    m = _ball((20, 30, 40), (10, 15, 20), 5, (1, 1, 1))
    sp = (1.0, 2.0, 0.5)
    r = hd95(np.zeros_like(m), m, sp)
    assert r.pred_empty
    assert r.value == pytest.approx(math.sqrt(20**2 + 60**2 + 20**2))


def test_input_validation():
    a = np.ones((4, 4, 4), bool)
    with pytest.raises(ValueError):
        hd95(a, np.ones((4, 4, 5), bool), (1, 1, 1))
    with pytest.raises(ValueError):
        hd95(a, a, (1, 1))


def test_validate_reports_hd95_and_empty_pred_count():
    """validate() end to end on a tiny synthetic set: no MONAI quantile call, and an
    all-background prediction is counted via hd95_n_empty_pred instead of vanishing."""
    import torch
    from monai.losses import DiceCELoss

    from src.training.train import validate

    class AllBackground(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], 2, *x.shape[2:])
            out[:, 0] = 5.0
            return out

    label = torch.zeros(1, 16, 16, 16)
    label[:, 6:10, 6:10, 6:10] = 1
    empty = torch.zeros_like(label)
    ds = [{"image": torch.zeros(1, 16, 16, 16), "label": label},
          {"image": torch.zeros(1, 16, 16, 16), "label": empty}]
    cfg = {"patch": {"spatial_size": [16, 16, 16]}, "train": {}, "model": {"out_channels": 2},
           "baseline": {"resample_spacing": [1.0, 1.0, 1.0]}}
    m = validate(AllBackground(), ds, DiceCELoss(to_onehot_y=True, softmax=True), torch.device("cpu"), cfg)
    assert m["hd95_n_empty_pred"] == 1      # sample 0: label present, prediction empty
    assert m["hd95_n_failed"] == 1          # sample 1: empty label -> NaN, excluded
    assert m["hd95"] == pytest.approx(math.sqrt(3 * 16**2))   # the penalty for sample 0
