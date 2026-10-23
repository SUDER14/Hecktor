"""Phase 4 pieces that need no training: sliding window, resampling, degradation, statistics."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.eval.evaluate_cases import call_model, coarsen_z
from src.eval.infer import resample_to_grid, sliding_window_dual, window_starts
from src.eval.stats import compare_models, paired_comparison, per_centre_table


def _aff(sp, o=(0, 0, 0)):
    a = np.diag([*sp, 1.0])
    a[:3, 3] = o
    return a


def test_window_starts_cover_volume_and_end_flush():
    assert window_starts(10, 16, 0.5) == [0]
    s = window_starts(100, 32, 0.5)
    assert s[0] == 0 and s[-1] == 68 and all(b - a <= 16 for a, b in zip(s, s[1:]))


def test_sliding_window_is_world_aligned_across_grids():
    """A fake model that segments 'CT bright AND PET bright' only works if CT/PET windows coincide."""
    ct = np.zeros((40, 40, 40), np.float32)
    ct[10:20, 10:20, 10:20] = 1
    pet = np.zeros((10, 10, 10), np.float32)       # 4 mm grid over the same 40 mm FOV
    pet[2:5, 2:5, 2:5] = 1                          # covers 8..20 mm
    pet_aff = _aff((4, 4, 4), (0, 0, 0))

    def fwd(c, p):
        up = torch.nn.functional.interpolate(p, size=c.shape[2:], mode="nearest")  # PET onto CT window grid
        m = ((c > 0.5) & (up > 0.5)).float()
        return torch.cat([1 - m, m], 1) * 10                                         # logits (b,2,...)

    res = sliding_window_dual(fwd, ct, pet, _aff((1, 1, 1)), pet_aff, (16, 16, 16), 2, "softmax", overlap=0.5)
    assert res["pred"].shape == ct.shape and res["p"].shape == (2,) + ct.shape and res["u"].shape == (1,) + ct.shape
    # inside the overlap of both bright regions the model says tumour; far outside it does not
    assert res["pred"][14, 14, 14] == 1 and res["pred"][35, 35, 35] == 0 and res["pred"][2, 2, 2] == 0


def test_sliding_window_volume_smaller_than_patch_is_padded():
    ct = np.ones((10, 12, 14), np.float32)
    pet = np.ones((3, 3, 4), np.float32)
    res = sliding_window_dual(lambda c, p: torch.cat([torch.zeros_like(c), torch.ones_like(c)], 1), ct, pet,
                              _aff((1, 1, 1)), _aff((4, 4, 4)), (16, 16, 16), 2, "softmax")
    assert res["pred"].shape == (10, 12, 14) and res["pred"].min() == 1


def test_resample_to_grid_linear_ramp_and_identity():
    x = np.arange(8, dtype=np.float32)[:, None, None] * np.ones((8, 4, 4), np.float32)
    same = resample_to_grid(x, _aff((2, 2, 2)), _aff((2, 2, 2)), x.shape)
    assert np.allclose(same, x)
    fine = resample_to_grid(x, _aff((2, 2, 2)), _aff((1, 1, 1)), (15, 7, 7))  # 2 mm -> 1 mm
    assert np.allclose(fine[:, 0, 0], np.arange(15) / 2.0, atol=1e-5)          # ramp value at x mm is x/2
    lead = resample_to_grid(np.stack([x, 2 * x]), _aff((2, 2, 2)), _aff((2, 2, 2)), x.shape)
    assert lead.shape == (2,) + x.shape and np.allclose(lead[1], 2 * x)


def test_coarsen_z_block_average_and_spacing():
    v = np.arange(6, dtype=np.float32)[None, None, :] * np.ones((3, 3, 6), np.float32)
    out, aff = coarsen_z(v, _aff((1, 1, 1), (5, 5, 5)), 3.0)
    assert out.shape == (3, 3, 2) and aff[2, 2] == 3 and np.allclose(out[0, 0], [1, 4])
    assert aff[2, 3] == 5 + 1.0  # block centre moves by (f-1)/2 voxels
    same, aff2 = coarsen_z(v, _aff((1, 1, 1)), 1.0)
    assert same is v and aff2[2, 2] == 1


def test_paired_comparison_detects_shift_and_handles_ties_and_nan():
    rng = np.random.default_rng(0)
    b = rng.uniform(0.3, 0.7, 30)
    a = b + 0.1 + rng.normal(0, 0.01, 30)
    r = paired_comparison(a, b)
    assert r["p_value"] < 1e-4 and r["rank_biserial"] == pytest.approx(1.0) and r["cohen_dz"] > 5
    assert r["ci_low"] > 0.05 and r["n_pairs"] == 30
    same = paired_comparison(b, b)
    assert same["p_value"] == 1.0 and same["rank_biserial"] == 0.0
    a2 = a.copy()
    a2[:3] = np.nan
    r2 = paired_comparison(a2, b)
    assert r2["n_pairs"] == 27 and r2["n_dropped_nan"] == 3


def test_bonferroni_and_per_centre():
    rng = np.random.default_rng(1)
    rows = []
    for m, shift in (("B3", 0.0), ("B4", 0.08), ("B5", 0.0)):
        for i in range(25):
            rows.append({"model": m, "patient_id": f"P{i}", "centre": "A" if i < 12 else "B",
                         "dice": 0.5 + shift + 0.02 * (i % 5) + rng.normal(0, 0.005)})
    df = pd.DataFrame(rows)
    res = compare_models(df, "dice", [("B4", "B3"), ("B5", "B3")], git="abc")
    assert res.loc[0, "p_bonferroni"] == pytest.approx(min(res.loc[0, "p_value"] * 2, 1))
    assert res.loc[0, "significant"] and not res.loc[1, "significant"] and (res["git"] == "abc").all()
    t = per_centre_table(df, ["dice"], git="abc")
    assert set(t["centre"]) == {"A", "B"} and "dice_mean" in t and (t["git"] == "abc").all()


def test_call_model_dispatch():
    class UNet3D(torch.nn.Module):
        def forward(self, ct):
            return ct

    ct = torch.zeros(1, 1, 2, 2, 2)
    assert call_model(UNet3D(), ct, None, None, None, None, None) is ct
