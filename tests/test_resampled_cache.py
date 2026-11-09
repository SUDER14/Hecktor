"""Resampled B1/B2 cache (derived from the native .acf arrays) and the Colab timing script's no-GPU branch."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data import crop_cache, resampled_cache
from src.data.acf_dataset import resample_target_shape, resample_volume
from src.data.hecktor2026 import Hecktor2026Adapter
from src.data.synthetic_shared import SynthConfig, write_synthetic_hecktor2026

TARGET = (2.0, 2.0, 2.0)


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    work = tmp_path_factory.mktemp("rs")
    root = work / "d"
    write_synthetic_hecktor2026(root, SynthConfig(fov_mm=(96, 96, 96), n_per_centre=1, seed=4, tumour_prob=1.0))
    cache = work / "native"
    for s in Hecktor2026Adapter(root).training_samples():
        crop_cache.build_case(s, cache, length_mm=96.0, width_mm=96.0)
    return work, cache


def test_resampled_case_has_target_spacing_dtypes_and_matches_direct_resampling(native):
    work, cache = native
    src = cache / "CHUP-001.acf"          # synthetic CHUP: 1 x 1 x 2 mm
    row = resampled_cache.build_case(src, work / "rs", TARGET)
    h, a = crop_cache.open_case(work / "rs" / "CHUP-001.acf")
    hs, s = crop_cache.open_case(src)
    assert h["spacing"] == [2.0, 2.0, 2.0] and tuple(h["shape"]) == resample_target_shape(hs["shape"], hs["spacing"], TARGET)
    assert np.allclose(np.diag(np.asarray(h["affine"]))[:3], 2.0)
    assert a["ct"].dtype == np.int16 and a["pet"].dtype == np.float16 and a["label"].dtype == np.uint8
    assert set(np.unique(a["label"]).tolist()) <= {0, 1, 2} and row["bytes"] == (work / "rs" / "CHUP-001.acf").stat().st_size
    r = resample_volume(s, hs["spacing"], TARGET, {"ct": 1, "pet": 1, "label": 0}, {"ct": -1000.0, "pet": 0.0, "label": 0.0})
    assert np.array_equal(np.asarray(a["label"]), np.rint(r["label"]).astype(np.uint8))
    assert np.array_equal(np.asarray(a["ct"]), np.rint(r["ct"]).astype(np.int16))
    assert not crop_cache.is_valid(work / "rs" / "CHUP-001.acf", crop_cache.signature(96.0, 96.0))  # not mistaken for a native file
    assert crop_cache.read_header(work / "rs" / "CHUP-001.acf")["sig"] == resampled_cache.signature(TARGET)


def test_resampled_pet_above_threshold_is_float32_and_finite(native):
    work, cache = native
    src = work / "big_native"
    (src).mkdir(exist_ok=True)
    h, a = crop_cache.open_case(cache / "CHUM-001.acf", mmap=False)
    pet = a["pet"].astype(np.float32)
    pet[a["ct"] > -300] += 1.0
    pet[10:30, 10:30, 10:30] = 311_000.0
    header = {k: v for k, v in h.items() if k != "arrays"}
    header["pet_dtype"] = "float32"
    crop_cache.write_case(src / "CHUM-001.acf", header, {"ct": a["ct"], "pet": pet, "label": a["label"]})
    resampled_cache.build_case(src / "CHUM-001.acf", work / "rs_big", TARGET)
    h2, a2 = crop_cache.open_case(work / "rs_big" / "CHUM-001.acf")
    assert a2["pet"].dtype == np.float32 and np.isfinite(np.asarray(a2["pet"])).all() and float(np.asarray(a2["pet"]).max()) > 65504


def test_colab_timing_script_reports_plainly_when_there_is_no_gpu(capsys, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    spec = importlib.util.spec_from_file_location("colab_timing", Path("scripts") / "colab_timing.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["colab_timing"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["colab_timing.py"])
    mod.main()
    out = capsys.readouterr().out
    assert out.startswith("=== PASTE FROM HERE ===") and "NO CUDA GPU AVAILABLE" in out and out.strip().endswith("=== END ===")


@pytest.mark.parametrize("kind", ["b1", "b4"])
def test_colab_timing_variant_logic_runs_on_cpu_with_tiny_shapes(kind, monkeypatch):
    """The GPU path cannot run here, so the step loop is exercised on CPU: CUDA bookkeeping calls are stubbed,
    shapes are tiny. Checks the step loop, accumulation, loss and the returned fields -- not any speed."""
    import torch
    spec = importlib.util.spec_from_file_location("colab_timing2", Path("scripts") / "colab_timing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, fn, lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 0)
    cfg = {"train": {"micro_batch_size": 2, "accum_steps": 2, "lr": 1e-3, "weight_decay": 0.0},
           "patch": {"spatial_size": [16, 16, 16]},
           "model": {"name": "unet3d" if kind == "b1" else "acf", "channels": [4, 8, 16], "strides": [2, 2], "out_channels": 3}}
    if kind == "b4":
        cfg["model"]["hyper_hidden"] = 8
    r = mod.time_variant(cfg, kind, False, warmup=1, steps=3, dev=torch.device("cpu"))
    assert r["finite"] and r["batch"] == 2 and r["step_mean"] > 0 and r["params"] > 0 and "oom" not in r
