"""On-the-fly resampling for B1/B2 on three REAL training cases (never a test case): thinnest, median and
thickest slice spacing in the training split. The stored resampled cache is not built (disk limit), so the
load-time path has to be right: fixed target spacing out, and predictions mapped back to the native grid."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data import crop_cache
from src.data.acf_dataset import AcfVolumeDataset, resample_target_shape
from src.data.conditioning import load_stats
from src.data.splits import load_split
from src.eval.infer import resample_to_grid

ROOT = Path(__file__).resolve().parents[1]
REAL = Path("D:/hecktor_cache")
pytestmark = pytest.mark.skipif(not (REAL / "CHUM-001.acf").exists(), reason="real cache not built")


# Measured 2026-11-16 (foreground Dice of a PERFECT resampled-grid prediction after mapping back to native):
# MDA-303 (z 1.25 mm) 0.9025, MDA-071 (z 3.27 mm) 1.0000, MDA-284 (z 5.0 mm) 0.9883. The loss on thin-slice
# cases is the information removed by resampling 1.25 mm slices to 3.27 mm; it is a ceiling on what B1/B2 can
# score on such cases, not a bug. The bound sits just under the worst measured value to catch regressions.
FG_ROUND_TRIP_MIN = 0.88


def _dice(a, b):
    return 2.0 * (a & b).sum() / max(1, a.sum() + b.sum())


def _three_train_cases(split):
    z = sorted((float(crop_cache.read_header(REAL / f"{p}.acf")["spacing"][2]), p) for p in split["train"])
    return [z[0][1], z[len(z) // 2][1], z[-1][1]]


def test_three_real_train_cases_resample_to_target_and_map_back_to_native():
    cfg = yaml.safe_load((ROOT / "configs/shared_b1_unet_resample.yaml").read_text(encoding="utf-8"))
    split = load_split(ROOT / cfg["split"]["path"])
    pids = _three_train_cases(split)
    assert not set(pids) & set(split["test"])
    target = [float(x) for x in cfg["baseline"]["resample_spacing"]]
    ds = AcfVolumeDataset([{"patient_id": p, "centre": p.split("-")[0]} for p in pids], REAL, cfg,
                          load_stats(ROOT / cfg["conditioning"]["stats_path"]))
    for i, pid in enumerate(pids):
        h = crop_cache.read_header(REAL / f"{pid}.acf")
        it = ds[i]
        img, lab, aff = it["image"].numpy(), it["label"][0].numpy(), it["affine"].numpy()
        native, naff = it["native_label"].numpy(), it["native_affine"].numpy()
        assert img.shape[1:] == resample_target_shape(native.shape, h["spacing"], target) == lab.shape
        assert np.allclose(np.linalg.norm(aff[:3, :3], axis=0), target, atol=1e-6)      # the one fixed spacing
        assert np.isfinite(img).all() and 0.0 <= img.min() and img[0].max() <= 1.0 and img[1].max() <= 20.0
        assert set(np.unique(lab)) <= {0, 1, 2}
        # a perfect prediction on the resampled grid, mapped back exactly as src/eval/evaluate_shared.py does
        probs = np.stack([(lab == k) for k in range(3)]).astype(np.float32)
        back = resample_to_grid(probs, aff, naff, native.shape, order=1).argmax(0)
        assert back.shape == native.shape
        fg = _dice(back > 0, native > 0)
        per = {k: round(float(_dice(back == k, native == k)), 4) for k in (1, 2) if (native == k).any()}
        print(f"{pid} spacing {np.round(h['spacing'], 3).tolist()} native {native.shape} -> {lab.shape} "
              f"round-trip fg Dice {fg:.4f} per label {per}")
        assert fg > FG_ROUND_TRIP_MIN
