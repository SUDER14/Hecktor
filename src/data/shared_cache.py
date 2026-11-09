"""CT+PET preprocessing cache and datasets, on one shared grid.

One cache entry per patient under cache_dir (put it on FAST LOCAL disk, outside the repo):

    <pid>.img.npy     (2, X, Y, Z) float16|float32: channel 0 = CT windowed to [0,1],
                      channel 1 = PET clipped to [0, suv_clip_max] then / suv_clip_max;
                      RAS orientation
    <pid>.label.npy   (X, Y, Z) uint8, RAW label values (0/1/2...); the class definition
                      (K, `data.label_map`) is applied when a sample is read, so changing
                      it never invalidates the cache
    <pid>.fg.npy      (N,3) int32 sampled foreground (label > 0) voxel indices
    <pid>.meta.json   affine, shape, spacing, orientation, timings, preprocess signature.
                      Written LAST, so an interrupted build is detected and redone.

mode "native":    no resampling; each scan keeps its own spacing (ACF).
mode "resampled": image and label are resampled to data.resample_spacing (B1/B2), the
                  ONLY place resampling happens.

Patches are fixed in voxels at native spacing (physical extent therefore varies by scan;
src/data/geometry.py sanity-checks it). Crop randomness is a pure function of
(seed, epoch, patient index), so a resumed run draws the same crops as an uninterrupted one.
"""

from __future__ import annotations

import json
import os
import time
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.data.conditioning import make_cond
from src.data.crop import crop_patch, sample_centre_vox
from src.data.geometry import DEFAULT_MAX_EXTENT_MM, DEFAULT_MIN_EXTENT_MM, check_patch_extent
from src.data.zip_reader import load_nifti

CACHE_VERSION = 3
MAX_FG = 100_000


def label_map(config: dict) -> np.ndarray:
    """raw label value -> class index. `data.label_map` (list indexed by raw value), else
    identity for K >= 3 and any-foreground-is-1 for K = 2."""
    lm = config.get("data", {}).get("label_map")
    if lm is None:
        k = config["model"].get("out_channels", 2)
        lm = [0, 1, 1] if k == 2 else list(range(k))
    return np.asarray(lm, dtype=np.int64)


def apply_label_map(raw: np.ndarray, lm: np.ndarray) -> np.ndarray:
    if raw.max(initial=0) >= len(lm):
        raise ValueError(f"raw label value {int(raw.max())} has no entry in label_map {lm.tolist()}")
    return lm[raw].astype(np.float32)


def signature(config: dict, mode: str) -> dict:
    w, suv = config["intensity"]["ct_window"], config["intensity"]["pet"]["suv_clip_max"]
    rs = [float(x) for x in config["baseline"]["resample_spacing"]] if mode == "resampled" else None
    return {"version": CACHE_VERSION, "mode": mode, "axcodes": "RAS", "resample_spacing": rs,
            "ct_window": [w["min"], w["max"]], "suv_clip_max": suv}


def _atomic(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


def entry_valid(cache_dir: Path, pid: str, sig: dict) -> bool:
    m = Path(cache_dir) / f"{pid}.meta.json"
    return m.exists() and json.loads(m.read_text()).get("sig") == sig


def _load_ras(path) -> tuple[np.ndarray, np.ndarray]:
    """Reorient to the closest canonical (RAS) axes -- a flip/permute, never an interpolation."""
    img = nib.as_closest_canonical(load_nifti(path))
    return np.asarray(img.dataobj), np.asarray(img.affine, dtype=float)


def _resample(arr: np.ndarray, affine: np.ndarray, pixdim, mode: str) -> tuple[np.ndarray, np.ndarray]:
    from monai.data import MetaTensor
    from monai.transforms import Spacing

    t = MetaTensor(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None],
                   affine=torch.as_tensor(affine))
    out = Spacing(pixdim=pixdim, mode=mode)(t)
    return out.numpy()[0], np.asarray(out.affine, dtype=float)


def _channel(path, ref_shape, ref_aff, mode: str, config: dict, lo: float, hi: float, name: str, pid: str):
    """Load one modality, check it is on the label's grid, resample if asked, scale to [0,1]."""
    arr, aff = _load_ras(path)
    if arr.shape != ref_shape or not np.allclose(aff, ref_aff, atol=1e-3):
        raise ValueError(f"{pid}: {name} is not on the label's grid ({arr.shape} vs {ref_shape}); "
                         "the shared-grid pipeline requires CT, PET and label on one grid")
    if mode == "resampled":
        arr, _ = _resample(arr, aff, config["baseline"]["resample_spacing"], "bilinear")
    return np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def build_entry(sample: dict, config: dict, cache_dir: str | Path, mode: str,
                dtype: str = "float16", seed: int = 0) -> dict:
    """Peak memory is one float32 channel plus the output, not CT + PET + label at once
    (volumes reach ~2e8 voxels and the target machines have 8 GB)."""
    t0 = time.perf_counter()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pid = sample["patient_id"]
    paths = dict(zip(sample["modalities"], sample["image"]))
    label, aff = _load_ras(sample["label"])
    label = np.rint(label).astype(np.uint8)
    ref_shape, ref_aff = label.shape, aff
    w, suv = config["intensity"]["ct_window"], config["intensity"]["pet"]["suv_clip_max"]
    ct = _channel(paths["CT"], ref_shape, ref_aff, mode, config, w["min"], w["max"], "CT", pid)
    img = np.empty((2,) + ct.shape, dtype=dtype)
    img[0] = ct
    del ct
    img[1] = _channel(paths["PT"], ref_shape, ref_aff, mode, config, 0.0, suv, "PET", pid)
    t_read = time.perf_counter() - t0
    if mode == "resampled":
        label, aff = _resample(label, aff, config["baseline"]["resample_spacing"], "nearest")
        label = np.rint(label).astype(np.uint8)
        if label.shape != img.shape[1:]:
            raise ValueError(f"{pid}: resampled label {label.shape} != resampled image {img.shape[1:]}")
    rng = np.random.default_rng([seed, zlib.crc32(pid.encode())])
    fg = np.flatnonzero(label.ravel())
    if len(fg) > MAX_FG:
        fg = rng.choice(fg, MAX_FG, replace=False)
    fg = np.stack(np.unravel_index(np.sort(fg), label.shape), axis=1).astype(np.int32) if len(fg) else         np.zeros((0, 3), np.int32)
    for name, arr in (("img", img), ("label", label), ("fg", fg)):
        _atomic(cache_dir / f"{pid}.{name}.npy", lambda f, a=arr: np.save(f, a))
    spacing = [float(np.linalg.norm(aff[:3, i])) for i in range(3)]
    meta = {"patient_id": pid, "affine": aff.tolist(), "shape": list(label.shape), "spacing": spacing,
            "orientation": "".join(nib.aff2axcodes(aff)), "n_fg_voxels": int((label > 0).sum()),
            "dtype": dtype, "sig": signature(config, mode),
            "seconds_read": round(t_read, 3), "seconds_total": round(time.perf_counter() - t0, 3)}
    _atomic(cache_dir / f"{pid}.meta.json", lambda f: f.write(json.dumps(meta, indent=1).encode()))
    return meta


def ensure_cache(samples: list[dict], config: dict, cache_dir: str | Path, mode: str) -> None:
    """Build any missing/stale entries; entries are independent so an interrupted build resumes."""
    sig = signature(config, mode)
    dtype = config.get("cache", {}).get("dtype", "float16")
    for s in samples:
        if not entry_valid(Path(cache_dir), s["patient_id"], sig):
            build_entry(s, config, cache_dir, mode, dtype, int(config["seed"]))


def _item_meta(sample: dict, meta: dict) -> dict:
    return {"patient_id": sample["patient_id"], "centre": sample["centre"],
            "cond": torch.as_tensor(sample["cond"]),
            "spacing": torch.tensor(meta["spacing"], dtype=torch.float32)}


def _attach_cond(samples: list[dict], metas: list[dict], stats: dict | None) -> None:
    """Conditioning from the CACHED spacing (the grid the network will actually see: after
    resampling it is the resample spacing), z-scored with the train-only stats."""
    for s, m in zip(samples, metas):
        s["cond"] = make_cond(m["spacing"], stats) if stats is not None else np.zeros(3, np.float32)


class DualCropDataset(torch.utils.data.Dataset):
    """Training: per patient, `num_samples` fixed-size voxel crops (list of dicts; MONAI's
    list_data_collate flattens them). Foreground-centred with prob pos/(pos+neg)."""

    def __init__(self, samples: list[dict], cache_dir, config: dict, stats: dict | None = None):
        self.samples, self.dir = [dict(s) for s in samples], Path(cache_dir)
        self.patch = tuple(config["patch"]["spatial_size"])
        self.pos, self.neg = config["patch"]["pos"], config["patch"]["neg"]
        self.num_samples = config["patch"]["num_samples"]
        self.min_mm = config["patch"].get("min_extent_mm") or DEFAULT_MIN_EXTENT_MM
        self.max_mm = config["patch"].get("max_extent_mm") or DEFAULT_MAX_EXTENT_MM
        self.lm = label_map(config)
        self.seed = int(config["seed"])
        self.metas = [json.loads((self.dir / f"{s['patient_id']}.meta.json").read_text()) for s in samples]
        _attach_cond(self.samples, self.metas, stats)
        for s, m in zip(self.samples, self.metas):  # fail at construction, not mid-epoch
            check_patch_extent(self.patch, m["spacing"], self.min_mm, self.max_mm, context=s["patient_id"])
        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        self._epoch[0] = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> list[dict]:
        s, meta, pid = self.samples[idx], self.metas[idx], self.samples[idx]["patient_id"]
        img = np.load(self.dir / f"{pid}.img.npy", mmap_mode="r")
        lab = np.load(self.dir / f"{pid}.label.npy", mmap_mode="r")
        fg = np.load(self.dir / f"{pid}.fg.npy")
        rng = np.random.default_rng([self.seed, int(self._epoch[0]), idx])
        out = []
        for _ in range(self.num_samples):
            centre = sample_centre_vox(fg, meta["shape"], self.pos, self.neg, rng)
            patches, _ = crop_patch({"image": img, "label": lab}, centre, self.patch, pad=True)
            item = _item_meta(s, meta)
            item["image"] = torch.from_numpy(np.ascontiguousarray(patches["image"]).astype(np.float32))
            item["label"] = torch.from_numpy(apply_label_map(np.asarray(patches["label"]), self.lm))[None]
            out.append(item)
        return out


class DualVolumeDataset(torch.utils.data.Dataset):
    """Validation/test: whole cached volumes, float32, image (2,X,Y,Z), label (1,X,Y,Z)."""

    def __init__(self, samples: list[dict], cache_dir, config: dict, stats: dict | None = None):
        self.samples, self.dir = [dict(s) for s in samples], Path(cache_dir)
        self.lm = label_map(config)
        self.metas = [json.loads((self.dir / f"{s['patient_id']}.meta.json").read_text()) for s in samples]
        _attach_cond(self.samples, self.metas, stats)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s, meta, pid = self.samples[idx], self.metas[idx], self.samples[idx]["patient_id"]
        item = _item_meta(s, meta)
        item["image"] = torch.from_numpy(np.load(self.dir / f"{pid}.img.npy").astype(np.float32))
        item["label"] = torch.from_numpy(apply_label_map(np.load(self.dir / f"{pid}.label.npy"), self.lm))[None]
        item["affine"] = torch.tensor(meta["affine"], dtype=torch.float64)
        return item


def build_dual_datasets(config: dict, mode: str | None = None) -> dict:
    """Splits/conditioning from build_samples, preprocessing from the cache (built on first use)."""
    from src.data.conditioning import load_stats
    from src.data.dataset import build_samples

    mode = mode or config.get("data", {}).get("mode", "native")
    cache_dir = config["cache"]["dir"]
    all_samples = build_samples(config)
    stats = load_stats(config["conditioning"]["stats_path"])
    out = {}
    for name, samples in all_samples.items():
        ensure_cache(samples, config, cache_dir, mode)
        out[name] = (DualCropDataset if name == "train" else DualVolumeDataset)(samples, cache_dir, config, stats)
    return out
