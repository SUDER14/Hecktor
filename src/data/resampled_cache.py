"""Resampled cache for the B1/B2 baselines: the cropped native `.acf` arrays resampled ONCE to one fixed spacing.

Why: resampling on the fly costs about 1.5 s per training fetch, which would dominate a 100-epoch run. The
cache keeps the same file format as src/data/crop_cache.py (so `crop_cache.open_case` reads it), with
    CT int16 (HU, rounded)   PET float16, or float32 where the resampled maximum exceeds 60,000   label uint8
resampled linearly (labels: nearest) to `target` spacing, which is chosen from the manifest median
(0.9766 x 0.9766 x 3.27 mm). Normalisation (CT clip/scale, per-case PET percentile) is still applied at load time,
exactly as for the native cache. Derived from the native `.acf` file, never from the zip.

No anti-aliasing is applied (same as MONAI Spacing's default); labels stay raw 0/1/2.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np

from src.data import crop_cache
from src.data.acf_dataset import CT_FILL, resample_volume, resampled_affine

VERSION = 1


def signature(target) -> dict:
    return {"version": VERSION, "kind": "resampled", "target_spacing": [float(x) for x in target],
            "source_version": crop_cache.VERSION}


def build_case(src_path: Path, out_dir: Path, target, pet_float32_above: float = crop_cache.PET_FLOAT32_ABOVE) -> dict:
    t0 = time.perf_counter()
    h, arr = crop_cache.open_case(Path(src_path))
    r = resample_volume(arr, h["spacing"], target, {"ct": 1, "pet": 1, "label": 0},
                        {"ct": CT_FILL, "pet": 0.0, "label": 0.0})
    del arr
    ct = np.clip(np.rint(r["ct"]), -32768, 32767).astype(np.int16)
    pet32 = r["pet"]
    pet_max = float(pet32.max())
    pet = pet32 if pet_max > pet_float32_above else pet32.astype(np.float16)
    if not np.isfinite(pet).all():
        raise ValueError(f"{h['patient_id']}: non-finite PET after resampling")
    lab = np.rint(r["label"]).astype(np.uint8)
    del r
    gc.collect()
    header = {"patient_id": h["patient_id"], "centre": h["centre"], "source_box": h["box"], "source_shape": h["shape"],
              "shape": list(ct.shape), "spacing": [float(x) for x in target],
              "affine": resampled_affine(h["affine"], h["spacing"], target).tolist(), "orientation": h["orientation"],
              "sig": signature(target), "pet_max": pet_max, "pet_dtype": str(pet.dtype),
              "label_values": [int(v) for v in np.unique(lab)]}
    size = crop_cache.write_case(Path(out_dir) / f"{h['patient_id']}.acf", header, {"ct": ct, "pet": pet, "label": lab})
    return {"patient_id": h["patient_id"], "centre": h["centre"], "bytes": size, "seconds": time.perf_counter() - t0,
            "shape": list(ct.shape)}
