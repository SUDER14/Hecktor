"""Dual-modality (CT + PET) preprocessing cache and datasets.

One cache entry per patient under cache_dir (put it on FAST LOCAL disk, e.g.
Kaggle /tmp, not /kaggle/working -- the latter is synced and slow):

    <pid>.ct.npy      (X,Y,Z) float16|float32, scaled to [0,1], RAS orientation
    <pid>.pet.npy     (X,Y,Z) same, SUV clipped to [0, suv_clip_max] then / suv_clip_max
    <pid>.label.npy   (X,Y,Z) uint8 on the CT grid
    <pid>.fg.npy      (N,3) int32 sampled foreground voxel indices on the CT grid
    <pid>.meta.json   affines, spacings, preprocess signature. Written LAST.

mode "native":    no resampling; CT and PET keep their own grids (ACF).
mode "resampled": CT, PET and label are resampled to data.resample_spacing (B1/B2),
                  which is the only place resampling happens.

Crops are cut by physical_crop (world coordinates, padded near borders), so a
96 mm box is 96^3 CT voxels at 1 mm and 24^3 PET voxels at 4 mm.

Crop randomness is a pure function of (seed, epoch, patient index), so a resumed
run draws the same crops as an uninterrupted one.
"""

from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np
import torch
from monai.transforms import (
    Compose, EnsureChannelFirstd, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd,
)

from src.data.conditioning import to_cond4
from src.data.physical_crop import grid_from_affine, physical_crop

CACHE_VERSION = 2
MAX_FG = 100_000


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


def build_entry(sample: dict, config: dict, cache_dir: str | Path, mode: str,
                dtype: str = "float16", seed: int = 0) -> dict:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pid = sample["patient_id"]
    paths = dict(zip(sample["modalities"], sample["image"]))
    keys = ["ct", "pet", "label"]
    stages = [LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys), Orientationd(keys=keys, axcodes="RAS")]
    if mode == "resampled":
        stages.append(Spacingd(keys=keys, pixdim=config["baseline"]["resample_spacing"],
                               mode=("bilinear", "bilinear", "nearest")))
    w, suv = config["intensity"]["ct_window"], config["intensity"]["pet"]["suv_clip_max"]
    stages += [ScaleIntensityRanged(keys=["ct"], a_min=w["min"], a_max=w["max"], b_min=0.0, b_max=1.0, clip=True),
               ScaleIntensityRanged(keys=["pet"], a_min=0.0, a_max=suv, b_min=0.0, b_max=1.0, clip=True)]
    d = Compose(stages)({"ct": paths["CT"], "pet": paths["PT"], "label": sample["label"]})

    ct_aff, pet_aff = np.asarray(d["ct"].affine, float), np.asarray(d["pet"].affine, float)
    ct, pet = (np.ascontiguousarray(d[k].numpy()[0]).astype(dtype) for k in ("ct", "pet"))
    label = np.rint(d["label"].numpy()[0]).astype(np.uint8)
    rng = np.random.default_rng([seed, zlib.crc32(pid.encode())])
    fg = np.flatnonzero(label.ravel())
    if len(fg) > MAX_FG:
        fg = rng.choice(fg, MAX_FG, replace=False)
    fg = np.stack(np.unravel_index(np.sort(fg), label.shape), axis=1).astype(np.int32) if len(fg) else \
        np.zeros((0, 3), np.int32)
    for name, arr in (("ct", ct), ("pet", pet), ("label", label), ("fg", fg)):
        _atomic(cache_dir / f"{pid}.{name}.npy", lambda f, a=arr: np.save(f, a))
    meta = {"patient_id": pid, "ct_affine": ct_aff.tolist(), "pet_affine": pet_aff.tolist(),
            "ct_shape": list(ct.shape), "pet_shape": list(pet.shape), "n_fg_voxels": int(label.sum()),
            "ct_spacing": [float(x) for x in np.diag(ct_aff)[:3]],
            "pet_spacing": [float(x) for x in np.diag(pet_aff)[:3]],
            "dtype": dtype, "sig": signature(config, mode)}
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
    cond = {m: to_cond4(sample["conditioning"][m]) for m in ("CT", "PT")}
    return {"patient_id": sample["patient_id"], "centre": sample["centre"],
            "cond_ct": torch.as_tensor(cond["CT"]), "cond_pet": torch.as_tensor(cond["PT"]),
            "ct_spacing": torch.tensor(meta["ct_spacing"], dtype=torch.float32),
            "pet_spacing": torch.tensor(meta["pet_spacing"], dtype=torch.float32)}


class DualCropDataset(torch.utils.data.Dataset):
    """Training: per patient, `num_samples` physical crops (list of dicts; MONAI's
    list_data_collate flattens them). Foreground-centred with prob pos/(pos+neg)."""

    def __init__(self, samples: list[dict], cache_dir, config: dict):
        self.samples, self.dir = samples, Path(cache_dir)
        self.patch = tuple(config["patch"]["spatial_size"])
        self.pos_prob = config["patch"]["pos"] / (config["patch"]["pos"] + config["patch"]["neg"])
        self.num_samples = config["patch"]["num_samples"]
        self.tol = config["patch"].get("extent_tol_voxels", 1.0)
        self.seed = int(config["seed"])
        self.metas = [json.loads((self.dir / f"{s['patient_id']}.meta.json").read_text()) for s in samples]
        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        self._epoch[0] = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> list[dict]:
        s, meta, pid = self.samples[idx], self.metas[idx], self.samples[idx]["patient_id"]
        ld = lambda n: np.load(self.dir / f"{pid}.{n}.npy", mmap_mode="r")  # noqa: E731
        ct_aff, pet_aff = np.asarray(meta["ct_affine"]), np.asarray(meta["pet_affine"])
        vols = {"ct": (ld("ct"), ct_aff), "pet": (ld("pet"), pet_aff), "label": (ld("label"), ct_aff)}
        fg = np.load(self.dir / f"{pid}.fg.npy")
        grid = grid_from_affine(meta["ct_shape"], ct_aff, "ct")
        rng = np.random.default_rng([self.seed, int(self._epoch[0]), idx])
        out = []
        for _ in range(self.num_samples):
            if len(fg) and rng.random() < self.pos_prob:
                centre = grid.voxel_to_world(fg[rng.integers(len(fg))])
            else:
                centre = grid.voxel_to_world(rng.random(3) * (np.asarray(grid.shape) - 1))
            patches, info = physical_crop(vols, "ct", self.patch, centre, self.tol, pad=True)
            item = _item_meta(s, meta)
            item.update({k: torch.from_numpy(np.ascontiguousarray(patches[k]).astype(np.float32))[None]
                         for k in ("ct", "pet", "label")})
            item["ct_start_mm"] = torch.as_tensor(info["start_mm"]["ct"], dtype=torch.float32)
            out.append(item)
        return out


class DualVolumeDataset(torch.utils.data.Dataset):
    """Validation/test: whole cached volumes, each on its own grid, float32 with a channel dim."""

    def __init__(self, samples: list[dict], cache_dir, config: dict):
        self.samples, self.dir = samples, Path(cache_dir)
        self.metas = [json.loads((self.dir / f"{s['patient_id']}.meta.json").read_text()) for s in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s, meta, pid = self.samples[idx], self.metas[idx], self.samples[idx]["patient_id"]
        item = _item_meta(s, meta)
        for k in ("ct", "pet", "label"):
            item[k] = torch.from_numpy(np.load(self.dir / f"{pid}.{k}.npy").astype(np.float32))[None]
        item["ct_affine"] = torch.tensor(meta["ct_affine"], dtype=torch.float64)
        item["pet_affine"] = torch.tensor(meta["pet_affine"], dtype=torch.float64)
        return item


def build_dual_datasets(config: dict, mode: str | None = None) -> dict:
    """Splits/conditioning from build_samples, preprocessing from the cache (built on first use)."""
    from src.data.dataset import build_samples

    mode = mode or config.get("data", {}).get("mode", "native")
    cache_dir = config["cache"]["dir"]
    all_samples = build_samples(config)
    out = {}
    for name, samples in all_samples.items():
        ensure_cache(samples, config, cache_dir, mode)
        out[name] = (DualCropDataset if name == "train" else DualVolumeDataset)(samples, cache_dir, config)
    return out
