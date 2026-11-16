"""Case index, case metadata and the display volume, read from the cropped `.acf` cache.

Everything is read with the repository's own reader (src/data/crop_cache.py, memory-mapped) and the PET is scaled
with the training code's own rule (src/data/acf_dataset.py), so what the Studio shows is what a model is given.
One case is open at a time. Nothing here writes a file.

Display volume, as sent to the browser:
  in-plane stride f = ceil(max(nx, ny) / 256); z is kept native
  CT    int16 HU, every f-th voxel
  PET   float16, (released value / per-case scale) clipped to [0, 20], every f-th voxel
  label uint8 raw values 0, 1, 2; block maximum over f x f so a small label does not vanish
The spacing in the header is the spacing AFTER the stride (sx, sy multiplied by f).
"""

from __future__ import annotations

import csv
import json
import math
import threading
from pathlib import Path

import numpy as np

from app.studio.settings import Settings
from src.data import crop_cache
from src.data.acf_dataset import PET_CLIP_MAX, _normalise, get_pet_scale, load_sidecar

MAX_INPLANE = 256
OUTLIER_PET_MAX = 1000.0   # docs/DATA_REPORT.md section 4: whole-scan PET maximum above 1,000
VOLUME_MAGIC = b"ACFV0001"
ALIGN = 8
SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}
TEST_NOTE = "test set: view only, not for tuning"
PET_NOTE = "SUV-like, units not stated"

_ONE_CASE = threading.Lock()   # one case in memory at a time, whatever the browser asks for


class MissingSource(Exception):
    """A file the Studio needs is absent. The message names it exactly; the UI shows it as an empty state."""


def flips_from_affine(affine) -> list[bool]:
    """Per-axis display flip for the reference viewer: True where the voxel index grows toward patient Right,
    Anterior or Superior, so that screen-left is patient Right and screen-top is Anterior / Superior.

    Only axis-aligned affines are accepted; a rotated one would need resampling, not a flip.
    """
    a = np.asarray(affine, float)[:3, :3]
    off = a - np.diag(np.diag(a))
    if not np.allclose(off, 0.0, atol=1e-6) or np.any(np.diag(a) == 0):
        raise ValueError("affine is not axis-aligned; display flips are undefined")
    return [bool(v > 0) for v in np.diag(a)]


def _read_split(path: Path) -> dict[str, str]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {pid: name for name in SPLIT_NAMES for pid in d.get(name, [])}


def _read_meta_csv(path: Path) -> dict[str, float]:
    """{patient_id: whole-scan PET maximum}. Only that column is used."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["patient_id"]: float(r["pet_max"]) for r in csv.DictReader(f)}


def _valid_scale(sidecar: dict, cache_dir: Path, pid: str) -> float | None:
    """The sidecar's scale, or None when the entry is absent or stale (same rule as the training loader)."""
    e = sidecar.get(pid)
    f = cache_dir / f"{pid}.acf"
    if e is None or not f.exists() or f.stat().st_size != e["bytes"]:
        return None
    return float(e["scale"])


def build_index(s: Settings) -> dict:
    """All cases in the cache with centre, spacing, split, PET scale and the outlier flag.

    Sources that are absent are listed under "missing" with what the index loses without them; the fields they
    feed are null, never guessed.
    """
    cache_dir = Path(s.cache_dir)
    files = sorted(cache_dir.glob("*.acf")) if cache_dir.is_dir() else []
    missing = []
    if not files:
        missing.append({"file": str(cache_dir / "<case>.acf"), "effect": "no case can be listed or opened"})
    split = {}
    if Path(s.split_path).exists():
        split = _read_split(s.split_path)
    else:
        missing.append({"file": str(s.split_path), "effect": "split column is empty; test cases cannot be marked"})
    sidecar = load_sidecar(cache_dir) if files else {}
    if files and not sidecar:
        missing.append({"file": str(cache_dir / "pet_scale.json"),
                        "effect": "PET scale column is empty (the scale is still computed when a case is opened)"})
    pet_max = None
    if Path(s.meta_csv).exists():
        pet_max = _read_meta_csv(s.meta_csv)
    else:
        missing.append({"file": str(s.meta_csv), "effect": "PET-scale outlier flag is unknown for every case"})
    cases = []
    for f in files:
        h = crop_cache.read_header(f)
        pid = h["patient_id"]
        sp = [float(v) for v in h["spacing"]]
        sname = split.get(pid)
        whole = pet_max.get(pid) if pet_max is not None else None
        cases.append({
            "patient_id": pid, "centre": h["centre"], "shape": [int(v) for v in h["shape"]],
            "sx": sp[0], "sy": sp[1], "sz": sp[2],
            "split": SPLIT_NAMES.get(sname), "is_test": sname == "test",
            "pet_scale": _valid_scale(sidecar, cache_dir, pid),
            "pet_max_whole_scan": whole,
            "pet_outlier": None if whole is None else bool(whole > OUTLIER_PET_MAX),
        })
    counts = {n: sum(c["split"] == n for c in cases) for n in SPLIT_NAMES.values()}
    return {"cases": cases, "n_cases": len(cases), "split_counts": counts,
            "n_pet_outliers": None if pet_max is None else sum(bool(c["pet_outlier"]) for c in cases),
            "outlier_rule": f"whole-scan PET maximum above {OUTLIER_PET_MAX:g} (docs/DATA_REPORT.md)",
            "centres": sorted({c["centre"] for c in cases}), "test_note": TEST_NOTE, "pet_note": PET_NOTE,
            "missing": missing}


def _case_path(s: Settings, pid: str) -> Path:
    """Only a name that is exactly one cache file is accepted, so a request can never reach another path."""
    cache_dir = Path(s.cache_dir)
    if not pid or any(ch in pid for ch in "/\\.:") or not (cache_dir / f"{pid}.acf").is_file():
        raise MissingSource(f"no cache file {cache_dir / (pid + '.acf')}")
    return cache_dir / f"{pid}.acf"


def case_meta(s: Settings, pid: str) -> dict:
    """Header facts of one case plus its split and PET scale. Reads the file header only."""
    path = _case_path(s, pid)
    h = crop_cache.read_header(path)
    split = _read_split(s.split_path).get(pid) if Path(s.split_path).exists() else None
    scale = _valid_scale(load_sidecar(s.cache_dir), Path(s.cache_dir), pid)
    return {"patient_id": pid, "centre": h["centre"], "shape": h["shape"], "spacing": h["spacing"],
            "affine": h["affine"], "orientation": h["orientation"], "orig_shape": h["orig_shape"], "box": h["box"],
            "flip": flips_from_affine(h["affine"]), "label_values": h["label_values"],
            "split": SPLIT_NAMES.get(split), "is_test": split == "test", "test_note": TEST_NOTE if split == "test" else None,
            "pet_scale": scale, "pet_scale_source": "pet_scale.json" if scale is not None else None,
            "pet_clip_max": PET_CLIP_MAX, "pet_note": PET_NOTE, "pet_dtype": h["arrays"]["pet"]["dtype"]}


def reduction_factor(nx: int, ny: int, max_inplane: int = MAX_INPLANE) -> int:
    return max(1, math.ceil(max(nx, ny) / max_inplane))


def _block_max(lab: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return np.ascontiguousarray(lab)
    nx, ny, nz = lab.shape
    ox, oy = math.ceil(nx / f), math.ceil(ny / f)
    pad = np.zeros((ox * f, oy * f, nz), lab.dtype)
    pad[:nx, :ny] = lab
    return pad.reshape(ox, f, oy, f, nz).max(axis=(1, 3))


def display_volume(s: Settings, pid: str) -> tuple[dict, dict[str, np.ndarray]]:
    """Header and the three arrays (X, Y, Z) for the browser; see the module docstring for the encoding."""
    path = _case_path(s, pid)
    with _ONE_CASE:
        h, arr = crop_cache.open_case(path)   # memory-mapped
        nx, ny, nz = (int(v) for v in h["shape"])
        f = reduction_factor(nx, ny)
        sidecar_scale = _valid_scale(load_sidecar(s.cache_dir), Path(s.cache_dir), pid)
        scale = get_pet_scale(s.cache_dir, pid, arr["ct"], arr["pet"])
        ct = np.ascontiguousarray(arr["ct"][::f, ::f, :])
        _, pet = _normalise(ct, np.asarray(arr["pet"][::f, ::f, :]), scale, PET_CLIP_MAX)
        lab = _block_max(np.asarray(arr["label"]), f)
        del arr
    split = _read_split(s.split_path).get(pid) if Path(s.split_path).exists() else None
    sp = [float(v) for v in h["spacing"]]
    header = {
        "patient_id": pid, "centre": h["centre"], "name": pid,
        "nx": int(ct.shape[0]), "ny": int(ct.shape[1]), "nz": int(ct.shape[2]),
        "sx": sp[0] * f, "sy": sp[1] * f, "sz": sp[2],
        "native_shape": [nx, ny, nz], "native_spacing": sp, "reduction": f,
        "affine": h["affine"], "flip": flips_from_affine(h["affine"]),
        "split": SPLIT_NAMES.get(split), "is_test": split == "test", "test_note": TEST_NOTE if split == "test" else None,
        "pet": {"scale": float(scale), "clip_max": PET_CLIP_MAX, "note": PET_NOTE,
                "scale_source": "pet_scale.json" if sidecar_scale is not None else "computed from the case (no valid sidecar entry)",
                "encoding": "released PET divided by scale, clipped to [0, clip_max]"},
        "labels": {"0": "background", "1": "label 1", "2": "label 2"},
        "order": "x fastest: index = i + nx * (j + ny * k)",
    }
    return header, {"ct": ct.astype(np.int16, copy=False), "pet": pet.astype(np.float16), "label": lab.astype(np.uint8, copy=False)}


def pack_volume(header: dict, arrays: dict[str, np.ndarray]) -> bytes:
    """magic | uint32 header length | JSON header | pad | arrays, each 8-byte aligned and x-fastest."""
    header = {**header, "arrays": {n: {"dtype": str(a.dtype), "offset": 0, "nbytes": int(a.nbytes)} for n, a in arrays.items()}}

    def layout(h):
        blob = json.dumps(h).encode("utf-8")
        off = len(VOLUME_MAGIC) + 4 + len(blob)
        off += (-off) % ALIGN
        offs = {}
        for n, a in arrays.items():
            offs[n] = off
            off += a.nbytes
            off += (-off) % ALIGN
        return blob, offs, off

    for _ in range(5):   # offsets are written into the header they depend on; iterate until stable
        blob, offs, total = layout(header)
        if all(header["arrays"][n]["offset"] == offs[n] for n in arrays):
            break
        for n in arrays:
            header["arrays"][n]["offset"] = offs[n]
    else:
        raise RuntimeError("display volume header layout did not converge")
    out = bytearray(total)
    out[:8] = VOLUME_MAGIC
    out[8:12] = len(blob).to_bytes(4, "little")
    out[12:12 + len(blob)] = blob
    for n, a in arrays.items():
        out[offs[n]:offs[n] + a.nbytes] = a.tobytes(order="F")
    return bytes(out)


def unpack_volume(body: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    """Inverse of pack_volume (used by the tests, and a reference for the browser-side reader)."""
    if body[:8] != VOLUME_MAGIC:
        raise ValueError("not a Studio display volume")
    n = int.from_bytes(body[8:12], "little")
    header = json.loads(body[12:12 + n].decode("utf-8"))
    shape = (header["nx"], header["ny"], header["nz"])
    arrays = {name: np.frombuffer(body, dtype=d["dtype"], count=int(np.prod(shape)), offset=d["offset"]).reshape(shape, order="F")
              for name, d in header["arrays"].items()}
    return header, arrays
