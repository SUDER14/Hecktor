"""Read-only verification of the cropped cache (one case at a time, no writes into the cache or the zip).

    python scripts/verify_cache.py --data_dir "<zip>" --cache_dir D:/hecktor_cache --out results/crop/verify_cache.json

1. file count, headers valid for the expected rule, no leftover .tmp
2. float16 overflow: for every case whose pre-cast PET maximum (stored in the header) exceeds 65504, count the
   voxels stored as +inf (whole crop and inside the label)
3. spot check N cases against the zip: shape, affine, and voxel-for-voxel values (CT rounded to int16, PET cast to
   float16, label raw), using the box stored in the header
4. PET scale statistics for named cases: whole-volume maximum (from the cache crop and from the scan CSV), the 99.5th
   percentile of positive PET voxels inside the CT body mask (CT > -500 HU), and PET mean inside the label
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import crop_cache  # noqa: E402
from src.data.hecktor2026 import Hecktor2026Adapter  # noqa: E402
from src.data.zip_reader import load_nifti  # noqa: E402

F16_MAX = 65504.0
OUTLIERS = ["MDA-052", "MDA-300", "MDA-310", "CHUP-014", "CHUP-031", "CHUP-045"]
SPOT = ["CHUM-001", "CHUS-046", "HGJ-043", "MDA-439", "USZ-003"]


def pet_scale_stats(pet16: np.ndarray, ct: np.ndarray, lab: np.ndarray) -> dict:
    pet = pet16.astype(np.float32)
    finite = np.isfinite(pet)
    body = (ct > -500) & finite & (pet > 0)
    in_lab = (lab > 0) & finite
    out = {"n_inf_voxels": int((~finite).sum()), "pet_max_finite_crop": float(pet[finite].max()),
           "p99_5_positive_in_body": float(np.percentile(pet[body], 99.5)) if body.any() else float("nan"),
           "n_body_positive_voxels": int(body.sum()),
           "label_mean": float(pet[in_lab].mean()) if in_lab.any() else float("nan"),
           "label_max": float(pet[in_lab].max()) if in_lab.any() else float("nan")}
    out["label_mean_over_p99_5"] = out["label_mean"] / out["p99_5_positive_in_body"] if out["p99_5_positive_in_body"] else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--length", type=float, default=350.0)
    ap.add_argument("--width", type=float, default=300.0)
    ap.add_argument("--scan", default="results/crop/crop_scan.csv")
    ap.add_argument("--out", default="results/crop/verify_cache.json")
    a = ap.parse_args()
    cd = Path(a.cache_dir)
    sig = crop_cache.signature(a.length, a.width)
    samples = {s["patient_id"]: s for s in Hecktor2026Adapter(a.data_dir).training_samples()}
    res: dict = {}

    files = sorted(cd.glob("*.acf"))
    ids = {f.stem for f in files}
    res["n_files"] = len(files)
    res["missing_vs_zip"] = sorted(set(samples) - ids)
    res["extra_vs_zip"] = sorted(ids - set(samples))
    res["tmp_files"] = [p.name for p in cd.glob("*.tmp")]
    heads = {f.stem: crop_cache.read_header(f) for f in files}
    res["bad_signature"] = [k for k, h in heads.items() if h["sig"] != sig]
    res["total_gb"] = sum(f.stat().st_size for f in files) / 2**30
    res["label_values_seen"] = sorted({v for h in heads.values() for v in h["label_values"]})
    res["ct_clipped_voxels_total"] = int(sum(h["ct_clipped_voxels"] for h in heads.values()))

    # no inf / NaN anywhere: stream every PET array in x-slabs (memory-mapped, one case at a time)
    nonfinite, dtypes = [], {}
    for f in files:
        _, arr = crop_cache.open_case(f)
        pet = arr["pet"]
        dtypes[str(pet.dtype)] = dtypes.get(str(pet.dtype), 0) + 1
        if not all(np.isfinite(pet[x0:x0 + 16]).all() for x0 in range(0, pet.shape[0], 16)):
            nonfinite.append(f.stem)
        del pet, arr
    res["pet_dtype_counts"] = dtypes
    res["cases_with_inf_or_nan"] = nonfinite
    over = {k: h["pet_max"] for k, h in heads.items() if h["pet_max"] > F16_MAX}
    rows = []
    for k in sorted(over):
        _, arr = crop_cache.open_case(cd / f"{k}.acf")
        pet, lab = np.asarray(arr["pet"]), np.asarray(arr["label"])
        inf = ~np.isfinite(pet)
        rows.append({"patient_id": k, "pet_max_pre_cast": over[k], "n_inf": int(inf.sum()),
                     "n_inf_in_label": int((inf & (lab > 0)).sum()), "n_label": int((lab > 0).sum())})
        del pet, lab, inf
        gc.collect()
    res["float16_overflow_cases"] = rows
    res["n_float16_overflow_cases"] = len(rows)

    spots = []
    for pid in SPOT:
        s, h = samples[pid], heads[pid]
        _, arr = crop_cache.open_case(cd / f"{pid}.acf")
        sl = tuple(slice(b0, b1) for b0, b1 in h["box"])
        img = nib.as_closest_canonical(load_nifti(s["image"][0]))
        ct = np.rint(img.get_fdata(dtype=np.float32)[sl]).astype(np.int16)
        aff_ok = np.allclose(img.affine[:3, :3], np.asarray(h["affine"])[:3, :3]) and np.allclose(
            img.affine @ np.append([b[0] for b in h["box"]], 1.0), np.asarray(h["affine"]) @ [0, 0, 0, 1])
        full_shape = list(img.shape)
        del img
        pet = np.asarray(nib.as_closest_canonical(load_nifti(s["image"][1])).dataobj)[sl]
        lab = np.asarray(nib.as_closest_canonical(load_nifti(s["label"])).dataobj)[sl].astype(np.uint8)
        c_ct, c_pet, c_lab = (np.asarray(arr[n]) for n in ("ct", "pet", "label"))
        pet = pet.astype(c_pet.dtype)   # float16 for ordinary cases, float32 where the header says so
        spots.append({"patient_id": pid, "shape_cache": list(c_ct.shape), "shape_from_zip_box": list(ct.shape),
                      "orig_shape_matches": full_shape == h["orig_shape"], "affine_matches_zip": bool(aff_ok),
                      "ct_equal": bool(np.array_equal(c_ct, ct)), "label_equal": bool(np.array_equal(c_lab, lab)),
                      "pet_dtype": str(c_pet.dtype), "pet_equal": bool(np.array_equal(c_pet, pet)),
                      "n_voxels_compared": int(ct.size)})
        del ct, pet, lab, c_ct, c_pet, c_lab
        gc.collect()
    res["spot_checks"] = spots

    scan = pd.read_csv(a.scan).set_index("patient_id")
    out_rows = []
    for pid in OUTLIERS:
        _, arr = crop_cache.open_case(cd / f"{pid}.acf")
        st = pet_scale_stats(np.asarray(arr["pet"]), np.asarray(arr["ct"]), np.asarray(arr["label"]))
        out_rows.append({"patient_id": pid, "pet_max_full_volume": float(scan.loc[pid, "pet_max"]),
                         "pet_dtype": str(arr["pet"].dtype), "pet_max_in_label_full_volume": float(scan.loc[pid, "pet_max_in_label"]), **st})
        gc.collect()
    res["outlier_stats"] = out_rows
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("float16_overflow_cases", "outlier_stats", "spot_checks")}, indent=1))
    print("\nspot checks:")
    for s in spots:
        print(s)
    print("\nfloat16 overflow cases:", len(rows))
    print(pd.DataFrame(rows).to_string(index=False) if rows else "none")
    print("\nnon-block outliers:")
    print(pd.DataFrame(out_rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
