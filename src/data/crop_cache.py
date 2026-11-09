"""Cropped native-grid cache: ONE memory-mappable file per case.

    <pid>.acf   = magic | header length | JSON header | pad | CT int16 | PET float16 | label uint8

CT (X,Y,Z) int16 HU (rounded; the 4 float32 CTs in the cohort are rounded, clipped counts are recorded),
PET (X,Y,Z) float16 as released (SUV units are NOT assumed here) -- except per case: a case whose PET
maximum in the crop exceeds PET_FLOAT32_ABOVE is stored as float32 instead (float16 overflows to +inf at
65504, and PET is never clipped or saturated). The dtype is read from the header, so readers need no change. Label (X,Y,Z) uint8 with RAW values
(0, 1, 2) so the class definition K can be chosen later. All three share one grid: RAS-reoriented, then the
label-free head-and-neck box of src/data/crop_rule.py. The header stores the cropped volume's affine,
spacing, the box in RAS-volume indices, and the rule parameters; the file is written atomically and last.

Re-derived 4 mm PET and the resampled B1/B2 inputs are computed from these arrays on demand
(src/data/pet_rederive.py and MONAI Spacing); they are not cached.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.crop_rule import BODY_HU, MIN_AREA_MM2, head_neck_box
from src.data.zip_reader import load_nifti

MAGIC = b"ACFCACH1"
VERSION = 1
ALIGN = 64
PET_FLOAT32_ABOVE = 60000.0   # crop PET max above this -> store PET as float32 (float16 would overflow)


def signature(length_mm: float, width_mm: float) -> dict:
    return {"version": VERSION, "length_mm": float(length_mm), "width_mm": float(width_mm),
            "body_hu": BODY_HU, "min_area_mm2": MIN_AREA_MM2, "axcodes": "RAS"}


def _pad(n: int) -> int:
    return (-n) % ALIGN


def _layout(header: dict, arrays: dict[str, np.ndarray]):
    blob = json.dumps(header).encode()
    off = 16 + len(blob)
    off += _pad(off)
    offs = {}
    for name, a in arrays.items():
        offs[name] = off
        off += a.nbytes
        off += _pad(off)
    return blob, offs


def write_case(path: Path, header: dict, arrays: dict[str, np.ndarray]) -> int:
    """Atomic write; returns file size in bytes. Offsets depend on the header length and vice
    versa, so the layout is iterated until it is stable."""
    header = {**header, "arrays": {n: {"dtype": str(a.dtype), "shape": list(a.shape), "offset": 0}
                                   for n, a in arrays.items()}}
    for _ in range(5):
        blob, offs = _layout(header, arrays)
        if all(header["arrays"][n]["offset"] == offs[n] for n in arrays):
            break
        for n in arrays:
            header["arrays"][n]["offset"] = offs[n]
    else:
        raise RuntimeError("cache header layout did not converge")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(MAGIC + len(blob).to_bytes(8, "little") + blob)
        f.write(b"\0" * _pad(f.tell()))
        for name, a in arrays.items():
            if f.tell() != offs[name]:
                raise RuntimeError(f"offset mismatch for {name}")
            f.write(np.ascontiguousarray(a).tobytes())
            f.write(b"\0" * _pad(f.tell()))
    os.replace(tmp, path)
    return path.stat().st_size


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"{path}: not an ACF cache file")
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def open_case(path: Path, mmap: bool = True) -> tuple[dict, dict[str, np.ndarray]]:
    h = read_header(path)
    arrs = {}
    for name, d in h["arrays"].items():
        a = np.memmap(path, dtype=d["dtype"], mode="r", offset=d["offset"], shape=tuple(d["shape"]), order="C")
        arrs[name] = a if mmap else np.array(a)
    return h, arrs


def is_valid(path: Path, sig: dict) -> bool:
    try:
        return path.exists() and read_header(path).get("sig") == sig
    except Exception:
        return False


def build_case(sample: dict, cache_dir: Path, length_mm: float, width_mm: float,
               pet_float32_above: float = PET_FLOAT32_ABOVE) -> dict:
    """One case in memory at a time; returns the stats row (bytes, seconds, shapes)."""
    t0 = time.perf_counter()
    pid = sample["patient_id"]
    img = nib.as_closest_canonical(load_nifti(sample["image"][0]))
    ct = img.get_fdata(dtype=np.float32)
    aff = np.asarray(img.affine, float)
    sp = [float(np.linalg.norm(aff[:3, i])) for i in range(3)]
    box = head_neck_box(ct, sp, length_mm, width_mm)
    sl = tuple(slice(a, b) for a, b in box)
    ct_c = np.rint(ct[sl])
    n_clip = int(((ct_c < -32768) | (ct_c > 32767)).sum())
    ct_c = np.clip(ct_c, -32768, 32767).astype(np.int16)
    full_shape = ct.shape
    del ct, img
    gc.collect()

    def crop(path, what):
        im = nib.as_closest_canonical(load_nifti(path))
        if im.shape != full_shape or not np.allclose(im.affine, aff, atol=1e-3):
            raise ValueError(f"{pid}: {what} is not on the CT grid")
        return np.asarray(im.dataobj)[sl]

    pet32 = crop(sample["image"][1], "PET").astype(np.float32)
    pet_max = float(pet32.max())
    pet = pet32 if pet_max > pet_float32_above else pet32.astype(np.float16)  # never clip, never overflow
    if not np.isfinite(pet).all():
        raise ValueError(f"{pid}: non-finite PET value in the crop")
    del pet32
    gc.collect()
    lab = crop(sample["label"], "label")
    if lab.min() < 0 or lab.max() > 255 or not np.array_equal(lab, np.rint(lab)):
        raise ValueError(f"{pid}: label values outside uint8 / non-integer")
    lab = lab.astype(np.uint8)
    caff = aff.copy()
    caff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ np.array([b[0] for b in box], float)
    header = {"patient_id": pid, "centre": sample["centre"], "orig_shape": list(full_shape),
              "box": [list(b) for b in box], "shape": list(ct_c.shape), "affine": caff.tolist(), "spacing": sp,
              "orientation": "".join(nib.aff2axcodes(aff)), "sig": signature(length_mm, width_mm),
              "ct_clipped_voxels": n_clip, "pet_max": pet_max, "pet_dtype": str(pet.dtype), "label_values": [int(v) for v in np.unique(lab)]}
    size = write_case(Path(cache_dir) / f"{pid}.acf", header, {"ct": ct_c, "pet": pet, "label": lab})
    return {"patient_id": pid, "centre": sample["centre"], "bytes": size, "seconds": time.perf_counter() - t0,
            "shape": list(ct_c.shape), "orig_shape": list(full_shape)}
