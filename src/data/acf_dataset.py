"""Datasets that read the cropped `.acf` cache (src/data/crop_cache.py), with per-case normalisation.

Normalisation, identical for every case and every centre (nothing is excluded, nothing is fitted on a split):
  CT   clip to [-1000, 1000] HU, then (x + 1000) / 2000  -> [0, 1]
  PET  divide by the 99.5th percentile of the POSITIVE PET voxels inside the CT body mask (CT > -500 HU),
       computed from that case alone at load time (memoised per process; optional JSON sidecar). The released PET
       is on very different numeric scales in some cases (header PET maxima from ~10 to ~3e5), so a per-case
       scale is used; units are not documented and are not assumed. No upper clip is applied unless
       config["intensity"]["pet"]["clip_max"] is set (it is not by default).
The label is stored raw (0, 1, 2) and mapped to classes at read time by data.label_map (K = 3: identity).

Two preprocessing modes, both computed from the cropped native arrays (no second cache):
  data.mode "native"    patches fixed in voxels at each scan's native spacing (shared-grid ACF; pet_mode shared)
  data.mode "resampled" every scan is linearly resampled (labels nearest) to baseline.resample_spacing, the one
                        fixed spacing chosen from the manifest median, for the B1/B2 baselines. Patches are cut
                        directly in the target grid by sampling the native arrays at the mapped positions (no
                        anti-aliasing, like MONAI Spacing's default).
The re-derived coarse-PET mode (pet_mode rederived) is NOT wired to this cache (it keeps the older
dual_cache path); see docs/DATA_REPORT.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from src.data import crop_cache
from src.data.conditioning import load_stats, make_cond
from src.data.crop import sample_centre_vox
from src.data.geometry import DEFAULT_MAX_EXTENT_MM, DEFAULT_MIN_EXTENT_MM, check_patch_extent
from src.data.shared_cache import apply_label_map, label_map
from src.data.splits import load_split

CT_LO, CT_HI = -1000.0, 1000.0
BODY_HU = -500.0
PET_PERCENTILE = 99.5
CT_FILL = CT_LO  # air outside the volume


def ct_normalise(ct: np.ndarray) -> np.ndarray:
    return ((np.clip(ct.astype(np.float32), CT_LO, CT_HI) - CT_LO) / (CT_HI - CT_LO)).astype(np.float32)


def pet_scale(ct, pet, chunk: int = 16) -> float:
    """99.5th percentile of positive, finite PET voxels inside the CT body mask (CT > -500 HU)."""
    vals = []
    for x0 in range(0, ct.shape[0], chunk):
        p = np.asarray(pet[x0:x0 + chunk]).astype(np.float32)
        m = (np.asarray(ct[x0:x0 + chunk]) > BODY_HU) & (p > 0) & np.isfinite(p)
        vals.append(p[m])
    v = np.concatenate(vals)
    if v.size == 0:
        raise ValueError("no positive PET voxel inside the CT body mask; cannot normalise this case")
    return float(np.percentile(v, PET_PERCENTILE))


_SCALES: dict[tuple[str, str], float] = {}
_SIDECARS: dict[str, dict] = {}
SIDECAR_NAME = "pet_scale.json"


def _rule() -> dict:
    return {"percentile": PET_PERCENTILE, "body_hu": BODY_HU, "positive_only": True}


def load_sidecar(cache_dir) -> dict[str, dict]:
    """{pid: {"scale": float, "bytes": int}} from `<cache_dir>/pet_scale.json` if its rule matches, else {}."""
    p = Path(cache_dir) / SIDECAR_NAME
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return d["scales"] if d.get("rule") == _rule() else {}


def _sidecar_scale(cache_dir, pid: str):
    """Scale from the sidecar, or None if absent or STALE (the .acf file's size changed since it was computed,
    e.g. a case was re-stored)."""
    key = str(cache_dir)
    if key not in _SIDECARS:
        _SIDECARS[key] = load_sidecar(cache_dir)
    e = _SIDECARS[key].get(pid)
    f = Path(cache_dir) / f"{pid}.acf"
    if e is None or not f.exists() or f.stat().st_size != e["bytes"]:
        return None
    return float(e["scale"])


def get_pet_scale(cache_dir, pid: str, ct=None, pet=None) -> float:
    """Per-case PET scale: process memo -> sidecar (valid) -> compute from the arrays (the rule is a pure function
    of the case, so all three agree)."""
    key = (str(cache_dir), pid)
    if key in _SCALES:
        return _SCALES[key]
    sc = _sidecar_scale(cache_dir, pid)
    if sc is None:
        if ct is None:
            _, arr = crop_cache.open_case(Path(cache_dir) / f"{pid}.acf")
            ct, pet = arr["ct"], arr["pet"]
        sc = pet_scale(ct, pet)
    _SCALES[key] = sc
    return sc


def write_scale_sidecar(cache_dir, pids: list[str] | None = None, progress_every: int = 20, log=print) -> dict:
    """Compute every case's scale ONCE and persist it, resumably: entries already valid in an existing sidecar are
    kept, the file is rewritten every `progress_every` new cases, one case at a time. Returns {pid: scale}."""
    cache_dir = Path(cache_dir)
    pids = pids or sorted(f.stem for f in cache_dir.glob("*.acf"))
    entries = dict(load_sidecar(cache_dir))
    out = cache_dir / SIDECAR_NAME
    todo = 0
    for pid in pids:
        f = cache_dir / f"{pid}.acf"
        if pid in entries and entries[pid]["bytes"] == f.stat().st_size:
            continue
        _, arr = crop_cache.open_case(f)
        entries[pid] = {"scale": pet_scale(arr["ct"], arr["pet"]), "bytes": f.stat().st_size}
        del arr
        todo += 1
        if todo % progress_every == 0:
            out.write_text(json.dumps({"rule": _rule(), "scales": entries}))
            log(f"{len(entries)}/{len(pids)} scales")
    out.write_text(json.dumps({"rule": _rule(), "scales": entries}))
    _SIDECARS.pop(str(cache_dir), None)
    _SCALES.clear()
    return {p: entries[p]["scale"] for p in pids}


def _window(arr, start, size, fill):
    """arr[start:start+size] per axis with out-of-volume voxels read as `fill`."""
    out = np.full(tuple(int(n) for n in size), fill, dtype=arr.dtype)
    src, dst = [], []
    for i0, n, dim in zip(start, size, arr.shape):
        a, b = max(int(i0), 0), min(int(i0) + int(n), dim)
        if b <= a:
            return out
        src.append(slice(a, b))
        dst.append(slice(a - int(i0), b - int(i0)))
    out[tuple(dst)] = np.asarray(arr[tuple(src)])
    return out


def _normalise(ct, pet, scale: float, clip_max=None):
    p = pet.astype(np.float32) / scale
    if clip_max is not None:
        p = np.minimum(p, clip_max)
    return ct_normalise(ct), p


def resample_target_shape(shape, spacing, target) -> tuple[int, int, int]:
    return tuple(int(max(1, np.floor(n * s / t + 1e-6))) for n, s, t in zip(shape, spacing, target))


def resample_volume(arrs: dict, spacing, target, orders: dict, fills: dict, chunk: int = 8) -> dict:
    """Linear (nearest for labels) resampling of whole arrays to `target` spacing, in x-slabs for memory.
    Target voxel j sits at native index (j + 0.5) * t / s - 0.5, so both grids tile the same field of view."""
    shape = next(iter(arrs.values())).shape
    out_shape = resample_target_shape(shape, spacing, target)
    ratio = np.asarray(target, float) / np.asarray(spacing, float)
    out = {k: np.empty(out_shape, np.float32) for k in arrs}
    ay = (np.arange(out_shape[1]) + 0.5) * ratio[1] - 0.5
    az = (np.arange(out_shape[2]) + 0.5) * ratio[2] - 0.5
    for x0 in range(0, out_shape[0], chunk):
        ax = (np.arange(x0, min(x0 + chunk, out_shape[0])) + 0.5) * ratio[0] - 0.5
        lo = max(int(np.floor(ax[0])) - 1, 0)
        hi = min(int(np.ceil(ax[-1])) + 2, shape[0])
        coords = np.stack(np.meshgrid(ax - lo, ay, az, indexing="ij"))
        for k, a in arrs.items():
            blk = np.asarray(a[lo:hi]).astype(np.float32)
            out[k][x0:x0 + len(ax)] = ndimage.map_coordinates(blk, coords, order=orders[k], mode="constant", cval=fills[k])
    return out


def resampled_affine(affine, spacing, target) -> np.ndarray:
    aff = np.asarray(affine, float).copy()
    ratio = np.asarray(target, float) / np.asarray(spacing, float)
    sp = np.diag(aff[:3, :3])
    aff[:3, 3] = aff[:3, 3] + (0.5 * ratio - 0.5) * sp
    aff[range(3), range(3)] = np.asarray(target, float)
    return aff


def resample_patch(arrs: dict, spacing, target, centre_vox, patch, orders: dict, fills: dict) -> dict:
    """A `patch`-voxel window of the TARGET grid centred on native voxel `centre_vox`, sampled directly from the
    native arrays (window-limited, so cost scales with the patch, not the volume)."""
    ratio = np.asarray(target, float) / np.asarray(spacing, float)
    patch = np.asarray(patch)
    offs = [(np.arange(n) - (n - 1) / 2.0) * r for n, r in zip(patch, ratio)]
    lo = np.floor(np.asarray(centre_vox) + np.array([o[0] for o in offs])).astype(int) - 1
    hi = np.ceil(np.asarray(centre_vox) + np.array([o[-1] for o in offs])).astype(int) + 2
    coords = np.stack(np.meshgrid(*[c + o - l for c, o, l in zip(centre_vox, offs, lo)], indexing="ij"))
    out = {}
    for k, a in arrs.items():
        win = _window(a, lo, hi - lo, fills[k]).astype(np.float32)
        out[k] = ndimage.map_coordinates(win, coords, order=orders[k], mode="constant", cval=fills[k]).astype(np.float32)
    return out


ORDERS = {"ct": 1, "pet": 1, "label": 0}
FILLS = {"ct": CT_FILL, "pet": 0.0, "label": 0.0}


class _Base(torch.utils.data.Dataset):
    def __init__(self, samples, cache_dir, config, stats):
        self.samples, self.dir = [dict(s) for s in samples], Path(cache_dir)
        self.mode = config.get("data", {}).get("mode", "native")
        self.target = [float(x) for x in config["baseline"]["resample_spacing"]]
        self.lm = label_map(config)
        self.clip_max = config.get("intensity", {}).get("pet", {}).get("clip_max")
        self.heads = [crop_cache.read_header(self.dir / f"{s['patient_id']}.acf") for s in samples]
        for s, h in zip(self.samples, self.heads):
            sp = self.target if self.mode == "resampled" else h["spacing"]
            s["spacing"] = list(sp)
            s["cond"] = make_cond(sp, stats)

    def _item_meta(self, i: int) -> dict:
        s = self.samples[i]
        return {"patient_id": s["patient_id"], "centre": s["centre"], "cond": torch.as_tensor(s["cond"]),
                "spacing": torch.tensor(s["spacing"], dtype=torch.float32)}

    def __len__(self) -> int:
        return len(self.samples)


class AcfCropDataset(_Base):
    """Training: per patient `num_samples` fixed-size voxel patches (list of dicts; MONAI's list_data_collate
    flattens them), foreground-centred with probability pos/(pos+neg). Draws are a pure function of
    (seed, epoch, patient index), so a resumed run reproduces them."""

    def __init__(self, samples, cache_dir, config, stats):
        super().__init__(samples, cache_dir, config, stats)
        self.patch = tuple(config["patch"]["spatial_size"])
        self.pos, self.neg = config["patch"]["pos"], config["patch"]["neg"]
        self.num_samples = config["patch"]["num_samples"]
        self.seed = int(config["seed"])
        lo = config["patch"].get("min_extent_mm") or DEFAULT_MIN_EXTENT_MM
        hi = config["patch"].get("max_extent_mm") or DEFAULT_MAX_EXTENT_MM
        for s in self.samples:   # fail at construction, not mid-epoch
            check_patch_extent(self.patch, s["spacing"], lo, hi, context=s["patient_id"])
        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        self._epoch[0] = int(epoch)

    def __getitem__(self, idx: int) -> list[dict]:
        pid = self.samples[idx]["patient_id"]
        h, arr = crop_cache.open_case(self.dir / f"{pid}.acf")
        scale = get_pet_scale(self.dir, pid, arr["ct"], arr["pet"])
        fg = np.argwhere(np.asarray(arr["label"]) > 0)
        rng = np.random.default_rng([self.seed, int(self._epoch[0]), idx])
        shape, out = h["shape"], []
        for _ in range(self.num_samples):
            centre = sample_centre_vox(fg, shape, self.pos, self.neg, rng)
            if self.mode == "resampled":
                p = resample_patch(arr, h["spacing"], self.target, centre, self.patch, ORDERS, FILLS)
                ct, pet, lab = p["ct"], p["pet"], p["label"].astype(np.int64)
            else:
                start = np.rint(centre - np.asarray(self.patch) / 2.0).astype(int)
                ct = _window(arr["ct"], start, self.patch, CT_FILL)
                pet = _window(arr["pet"], start, self.patch, 0.0)
                lab = _window(arr["label"], start, self.patch, 0).astype(np.int64)
            ct, pet = _normalise(ct, pet, scale, self.clip_max)
            item = self._item_meta(idx)
            item["image"] = torch.from_numpy(np.stack([ct, pet]).astype(np.float32))
            item["label"] = torch.from_numpy(apply_label_map(lab, self.lm))[None]
            out.append(item)
        return out


class AcfVolumeDataset(_Base):
    """Validation / test: the whole cropped volume (resampled to the target spacing in resampled mode), plus the
    native-grid label and affine so every model is scored on the same native label."""

    def __getitem__(self, idx: int) -> dict:
        pid = self.samples[idx]["patient_id"]
        h, arr = crop_cache.open_case(self.dir / f"{pid}.acf")
        scale = get_pet_scale(self.dir, pid, arr["ct"], arr["pet"])
        native_lab = apply_label_map(np.asarray(arr["label"]).astype(np.int64), self.lm).astype(np.uint8)
        if self.mode == "resampled":
            r = resample_volume(arr, h["spacing"], self.target, ORDERS, FILLS)
            ct, pet, lab = r["ct"], r["pet"], r["label"].astype(np.int64)
            aff = resampled_affine(h["affine"], h["spacing"], self.target)
        else:
            ct, pet, lab = np.asarray(arr["ct"]), np.asarray(arr["pet"]), np.asarray(arr["label"]).astype(np.int64)
            aff = np.asarray(h["affine"], float)
        ct, pet = _normalise(ct, pet, scale, self.clip_max)
        item = self._item_meta(idx)
        item["image"] = torch.from_numpy(np.stack([ct, pet]).astype(np.float32))
        item["label"] = torch.from_numpy(apply_label_map(lab, self.lm))[None]
        item["affine"] = torch.tensor(aff, dtype=torch.float64)
        item["native_label"] = torch.from_numpy(native_lab)
        item["native_affine"] = torch.tensor(h["affine"], dtype=torch.float64)
        return item


def build_acf_datasets(config: dict) -> dict:
    """Split from config["split"]["path"], conditioning stats (train-only) from config["conditioning"]["stats_path"],
    samples (id, centre) from the cache headers, so no access to the zip is needed."""
    cd = Path(config["cache"]["dir"])
    heads = {f.stem: crop_cache.read_header(f) for f in sorted(cd.glob("*.acf"))}
    split = load_split(config["split"]["path"])
    stats = load_stats(config["conditioning"]["stats_path"])
    out = {}
    for name, ids in split.items():
        if name.startswith("_"):
            continue
        missing = [i for i in ids if i not in heads]
        if missing:
            raise KeyError(f"split references cases not in the cache: {missing[:5]}")
        samples = [{"patient_id": i, "centre": heads[i]["centre"]} for i in ids]
        out[name] = (AcfCropDataset if name == "train" else AcfVolumeDataset)(samples, cd, config, stats)
    return out
