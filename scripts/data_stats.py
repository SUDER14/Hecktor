"""Per-case voxel statistics (intensity ranges, label counts) read through the zip.

Why: whether PET is in SUV, and what the label values mean, cannot be settled from
headers. One row per case; every number is measured from the file, none assumed.

    python scripts/data_stats.py --data_dir "<zip or dir>" --out results/manifest_hecktor2026/voxel_stats.csv

Resumable: cases already in the output CSV are skipped. Reads each case once.
"""

from __future__ import annotations

import argparse
import gc
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.hecktor2026 import Hecktor2026Adapter  # noqa: E402
from src.data.zip_reader import load_nifti  # noqa: E402


def case_stats(sample: dict) -> dict:
    """One volume in memory at a time: 8 GB machines thrash if CT, PET and label (up to ~2e8
    voxels each) are all resident as float64."""
    ct_p, pt_p = sample["image"]
    ct_img = load_nifti(ct_p)
    ct = ct_img.get_fdata(dtype=np.float32)
    row = {"patient_id": sample["patient_id"], "centre": sample["centre"], "ct_min": float(ct.min()),
           "ct_max": float(ct.max()), "ct_mean": float(ct.mean(dtype=np.float64)),
           "ct_dtype": str(ct_img.get_data_dtype())}
    del ct, ct_img
    pt_img = load_nifti(pt_p)
    pt = pt_img.get_fdata(dtype=np.float32)
    lab_img = load_nifti(sample["label"])
    lab = np.asarray(lab_img.dataobj)
    row.update({
        "pt_min": float(pt.min()), "pt_max": float(pt.max()), "pt_mean": float(pt.mean(dtype=np.float64)),
        # percentiles of a deterministic every-3rd-voxel subsample (full-volume percentiles are slow)
        "pt_p99_sub": float(np.percentile(pt[::3, ::3, ::3], 99)),
        "pt_p999_sub": float(np.percentile(pt[::3, ::3, ::3], 99.9)),
        "pt_nan": int(np.isnan(pt).sum()), "pt_dtype": str(pt_img.get_data_dtype()),
        "lab_dtype": str(lab.dtype)})
    vals, cnts = np.unique(lab, return_counts=True)
    row["lab_values"] = "|".join(str(v) for v in vals.tolist())
    row["voxel_ml"] = float(np.prod(pt_img.header.get_zooms()[:3])) / 1000.0
    row["n_voxels"] = int(lab.size)
    for k in (0, 1, 2):
        row[f"n_label{k}"] = int(cnts[vals == k].sum()) if k in vals else 0
    row["n_label_other"] = int(sum(c for v, c in zip(vals, cnts) if v not in (0, 1, 2)))
    # PET inside each labelled structure (max uptake; whole-volume max can be bladder/brain)
    for k in (1, 2):
        m = lab == k
        row[f"pt_max_in_label{k}"] = float(pt[m].max()) if m.any() else float("nan")
        row[f"pt_mean_in_label{k}"] = float(pt[m].mean(dtype=np.float64)) if m.any() else float("nan")
    gc.collect()
    return row


def label_stats(sample: dict) -> dict:
    """Labels only (tiny to read): used for ALL cases, since PET/CT reads are slow."""
    img = load_nifti(sample["label"])
    lab = np.asarray(img.dataobj)
    vals, cnts = np.unique(lab, return_counts=True)
    row = {"patient_id": sample["patient_id"], "centre": sample["centre"], "lab_dtype": str(lab.dtype),
           "lab_values": "|".join(str(v) for v in vals.tolist()), "n_voxels": int(lab.size),
           "voxel_ml": float(np.prod(img.header.get_zooms()[:3])) / 1000.0}
    for k in (0, 1, 2):
        row[f"n_label{k}"] = int(cnts[vals == k].sum()) if k in vals else 0
    row["n_label_other"] = int(sum(c for v, c in zip(vals, cnts) if v not in (0, 1, 2)))
    return row


def stratified_per_centre(samples: list[dict], n: int, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_c: dict[str, list[dict]] = {}
    for s in samples:
        by_c.setdefault(s["centre"], []).append(s)
    out = []
    for c in sorted(by_c):
        idx = rng.choice(len(by_c[c]), size=min(n, len(by_c[c])), replace=False)
        out += [by_c[c][i] for i in sorted(idx)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label_only", action="store_true", help="label counts for every case (fast)")
    ap.add_argument("--per_centre", type=int, default=None, help="PET/CT stats on N seeded random cases per centre")
    ap.add_argument("--workers", type=int, default=max(1, 1))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    seen = set(done.get("patient_id", []))
    rows = done.to_dict("records")
    samples = Hecktor2026Adapter(a.data_dir).training_samples()
    if a.per_centre:
        samples = stratified_per_centre(samples, a.per_centre)
    todo = [s for s in samples if s["patient_id"] not in seen]
    fn = label_stats if a.label_only else case_stats
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(fn, todo), 1):
            rows.append(r)
            gc.collect()
            if i % 20 == 0:
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{len(rows)}/{len(samples)}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("done", len(rows))


if __name__ == "__main__":
    main()
