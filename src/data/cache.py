"""Preprocessing cache: resample + normalise each volume once, train from disk.

Why: the baseline pipeline re-reads a gzipped NIfTI and re-runs Spacingd on the
CPU for every volume on every epoch. A stage timing of that pipeline (one
512x512x112 CT, local CPU) put resampling at ~87% of the per-volume cost. The
cache moves load -> orient -> (resample) -> intensity to a one-off script
(scripts/precompute_cache.py); training then only slices a random crop out of a
memory-mapped array.

Layout, one entry per (patient, modality) under cache_dir:
    <patient>__<modality>.image.npy   (D,H,W) float16|float32, intensity-normalised
    <patient>__<modality>.label.npy   (D,H,W) uint8
    <patient>__<modality>.fg.npy      (N,3) int32 sampled foreground voxel coords
    <patient>__<modality>.bg.npy      (N,3) int32 sampled background coords (image > 0,
                                      matching RandCropByPosNegLabeld(image_threshold=0))
    <patient>__<modality>.meta.json   original_spacing, cache_spacing, shapes, preprocess
                                      signature. Written LAST: an entry exists iff this does.
The original (pre-resample) spacing is kept so the native-spacing path and the
conditioning vectors still have it; `--native` builds a cache with no resampling.

Float16 images quantise the [0,1] window in steps of 2^-11 (~4.9e-4), i.e. up to
~0.68 HU at the top of the -1000..400 HU CT window and finer below; use
float32 (--image_dtype float32) where that is not acceptable, at 2x the disk.

Crop randomness is a pure function of (seed, epoch, volume index): resumed runs
redraw the same crops as an uninterrupted run, even with persistent workers
(whose own RNG state is not checkpointed). Call set_epoch() each epoch.
"""

from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np
import torch
from monai.transforms import Compose

from src.data.transforms import build_baseline_load_transforms, build_load_transforms

CACHE_VERSION = 1
MAX_COORDS = 100_000          # sampled foreground / background centres kept per volume
_BG_PROBES = 400_000          # random voxels probed to find background centres
_N_ORIENT_STAGES = 3          # Load, EnsureChannelFirst, Orientation: both transform lists start with these


def entry_name(item: dict) -> str:
    return f"{item['patient_id']}__{item['modality']}"


def preprocess_signature(config: dict, modality: str, native: bool) -> dict:
    """Everything that changes the cached numbers; compared on load to catch stale caches."""
    if modality == "CT":
        w = config["intensity"]["ct_window"]
        intensity = {"ct_window": [w["min"], w["max"]]}
    else:
        intensity = {"pet_suv_clip_max": config["intensity"]["pet"]["suv_clip_max"]}
    spacing = None if native else [float(x) for x in config["baseline"]["resample_spacing"]]
    return {"version": CACHE_VERSION, "axcodes": "RAS", "resample_spacing": spacing, "intensity": intensity}


def _spacing_of(meta_tensor) -> list[float]:
    aff = np.asarray(meta_tensor.affine, dtype=np.float64)
    return [float(x) for x in np.sqrt((aff[:3, :3] ** 2).sum(0))]


def _atomic_write(path: Path, writer) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        writer(f)
    os.replace(tmp, path)


def _np_save(path: Path, arr: np.ndarray) -> None:
    _atomic_write(path, lambda f: np.save(f, arr))


def entry_is_valid(cache_dir: Path, item: dict, signature: dict, image_dtype: str | None = None) -> bool:
    meta_path = Path(cache_dir) / f"{entry_name(item)}.meta.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return meta.get("preprocess") == signature and (image_dtype is None or meta.get("image_dtype") == image_dtype)


def build_cache_entry(
    item: dict, config: dict, cache_dir: str | Path, native: bool = False,
    image_dtype: str = "float16", seed: int = 0,
) -> dict:
    """Preprocesses one {"image": path, "label": path, ...} item and writes its cache entry.
    Returns the meta dict. Peak memory is one full resampled volume (image + label)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    modality = item["modality"]
    stages = (build_load_transforms if native else build_baseline_load_transforms)(modality, config)
    d = Compose(stages[:_N_ORIENT_STAGES])({"image": item["image"], "label": item["label"]})
    original_spacing = _spacing_of(d["image"])
    d = Compose(stages[_N_ORIENT_STAGES:])(d)

    image = np.ascontiguousarray(d["image"].numpy()[0]).astype(image_dtype)
    label = np.rint(d["label"].numpy()[0]).astype(np.uint8)
    name = entry_name(item)
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])

    fg = np.flatnonzero(label.ravel())
    if len(fg) > MAX_COORDS:
        fg = rng.choice(fg, MAX_COORDS, replace=False)
    probes = rng.integers(0, image.size, _BG_PROBES)
    bg = probes[(label.ravel()[probes] == 0) & (image.ravel()[probes] > 0)][:MAX_COORDS]

    def to_coords(flat: np.ndarray) -> np.ndarray:
        return np.stack(np.unravel_index(np.sort(flat), image.shape), axis=1).astype(np.int32)

    _np_save(cache_dir / f"{name}.image.npy", image)
    _np_save(cache_dir / f"{name}.label.npy", label)
    _np_save(cache_dir / f"{name}.fg.npy", to_coords(fg))
    _np_save(cache_dir / f"{name}.bg.npy", to_coords(bg))
    meta = {
        "patient_id": item["patient_id"], "modality": modality, "source_image": str(item["image"]),
        "original_spacing": original_spacing, "cache_spacing": _spacing_of(d["image"]),
        "shape": list(image.shape), "image_dtype": image_dtype,
        "n_fg_voxels": int(label.sum()), "preprocess": preprocess_signature(config, modality, native),
    }
    _atomic_write(cache_dir / f"{name}.meta.json", lambda f: f.write(json.dumps(meta, indent=1).encode()))
    return meta


def _load_meta(cache_dir: Path, item: dict, config: dict) -> dict:
    path = cache_dir / f"{entry_name(item)}.meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no cache entry {path.name} in {cache_dir} -- run scripts/precompute_cache.py first")
    meta = json.loads(path.read_text())
    native = "baseline" not in config
    expected = preprocess_signature(config, item["modality"], native)
    if meta["preprocess"] != expected:
        raise ValueError(
            f"cache entry {path.name} was built with {meta['preprocess']} but this config needs {expected} "
            "-- rebuild with scripts/precompute_cache.py --overwrite")
    return meta


class CachedCropDataset(torch.utils.data.Dataset):
    """Training data: per volume, num_samples random pos/neg crops read from the memory-mapped cache.

    Mirrors RandCropByPosNegLabeld(pos, neg, num_samples, image_threshold=0): a crop centre is a
    foreground voxel with probability pos/(pos+neg), else a background voxel; if one class is
    empty the other is used. Centres come from the sampled coordinate lists saved at build time.
    The crop window is clamped inside the volume; a volume smaller than the patch is zero-padded
    at the high end (MONAI would return a smaller crop and break collation)."""

    def __init__(self, items: list[dict], cache_dir: str | Path, config: dict):
        self.cache_dir = Path(cache_dir)
        self.items = items
        self.patch = np.asarray(config["patch"]["spatial_size"], dtype=int)
        self.pos_prob = config["patch"]["pos"] / (config["patch"]["pos"] + config["patch"]["neg"])
        self.num_samples = config["patch"]["num_samples"]
        self.seed = int(config["seed"])
        self.metas = [_load_meta(self.cache_dir, it, config) for it in items]
        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()  # visible to persistent workers

    def set_epoch(self, epoch: int) -> None:
        self._epoch[0] = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def _crop(self, arr: np.ndarray, start: np.ndarray) -> np.ndarray:
        hi = np.minimum(start + self.patch, arr.shape)
        block = np.asarray(arr[start[0]:hi[0], start[1]:hi[1], start[2]:hi[2]])
        pad = [(0, int(p - s)) for p, s in zip(self.patch, block.shape)]
        return np.pad(block, pad) if any(b for _, b in pad) else block

    def __getitem__(self, idx: int) -> list[dict]:
        item, name = self.items[idx], entry_name(self.items[idx])
        image = np.load(self.cache_dir / f"{name}.image.npy", mmap_mode="r")
        label = np.load(self.cache_dir / f"{name}.label.npy", mmap_mode="r")
        fg = np.load(self.cache_dir / f"{name}.fg.npy")
        bg = np.load(self.cache_dir / f"{name}.bg.npy")
        rng = np.random.default_rng([self.seed, int(self._epoch[0]), idx])
        shape = np.asarray(image.shape)
        out = []
        for _ in range(self.num_samples):
            use_fg = len(fg) > 0 and (len(bg) == 0 or rng.random() < self.pos_prob)
            pool = fg if use_fg else bg
            centre = pool[rng.integers(len(pool))] if len(pool) else rng.integers(0, shape)
            start = np.clip(centre - self.patch // 2, 0, np.maximum(shape - self.patch, 0))
            out.append({
                **{k: v for k, v in item.items() if k not in ("image", "label")},
                "image": torch.from_numpy(self._crop(image, start).astype(np.float32))[None],
                "label": torch.from_numpy(self._crop(label, start).astype(np.float32))[None],
                "crop_start": torch.as_tensor(start, dtype=torch.long),
            })
        return out


class CachedVolumeDataset(torch.utils.data.Dataset):
    """Validation/test data: whole cached volumes as float32 (1,D,H,W), like build_*_val_transforms."""

    def __init__(self, items: list[dict], cache_dir: str | Path, config: dict):
        self.cache_dir, self.items = Path(cache_dir), items
        self.metas = [_load_meta(self.cache_dir, it, config) for it in items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item, name = self.items[idx], entry_name(self.items[idx])
        return {
            **{k: v for k, v in item.items() if k not in ("image", "label")},
            "image": torch.from_numpy(np.load(self.cache_dir / f"{name}.image.npy").astype(np.float32))[None],
            "label": torch.from_numpy(np.load(self.cache_dir / f"{name}.label.npy").astype(np.float32))[None],
        }


def build_cached_datasets(config: dict) -> dict:
    """Same samples/splits/conditioning as build_monai_datasets, read from config["cache"]["dir"]."""
    from src.data.dataset import _expand_to_modality_items, build_samples

    all_samples = build_samples(config)
    modalities = {m for subset in all_samples.values() for s in subset for m in s["modalities"]}
    if len(modalities) > 1:
        raise NotImplementedError("cached datasets are single-modality only, like build_monai_datasets()")
    cache_dir = config["cache"]["dir"]
    out = {}
    for name, samples in all_samples.items():
        items = _expand_to_modality_items(samples)
        out[name] = (CachedCropDataset if name == "train" else CachedVolumeDataset)(items, cache_dir, config)
    return out
