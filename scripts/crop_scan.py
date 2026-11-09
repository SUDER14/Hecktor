"""One pass over all cases: label-free crop-rule statistics from the CT, label bounding boxes, and PET/CT extremes.

    python scripts/crop_scan.py --data_dir "<zip>" --out results/crop/crop_scan.csv

The rule statistics (src/data/crop_rule.py) use ONLY the CT. The label bounding box is stored
alongside purely so scripts/crop_verify.py can check, for any candidate (length, width), that no
labelled voxel falls outside the box. Sequential (one case at a time), resumable (progress file
rewritten every 20 cases; finished cases are skipped on restart), float32 + gc.collect().
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.crop_rule import body_slice_stats  # noqa: E402
from src.data.hecktor2026 import Hecktor2026Adapter  # noqa: E402
from src.data.zip_reader import load_nifti  # noqa: E402

LENGTHS = (200.0, 250.0, 300.0, 350.0, 400.0)


def scan_case(s: dict) -> dict:
    """Labels first (uint8, small), then CT, then PET, one volume at a time."""
    lab = np.asarray(nib.as_closest_canonical(load_nifti(s["label"])).dataobj)
    fg = np.argwhere(lab > 0)
    row = {"patient_id": s["patient_id"], "centre": s["centre"], "n_label_voxels": int(len(fg))}
    if len(fg):
        lo, hi = fg.min(0), fg.max(0) + 1
        row.update({f"lab_{a}{b}": int(v) for a, lo_, hi_ in zip("xyz", lo, hi) for b, v in (("0", lo_), ("1", hi_))})
    del fg
    gc.collect()
    img = nib.as_closest_canonical(load_nifti(s["image"][0]))
    ct = img.get_fdata(dtype=np.float32)
    sp = [float(np.linalg.norm(img.affine[:3, i])) for i in range(3)]
    row.update({"nx": ct.shape[0], "ny": ct.shape[1], "nz": ct.shape[2], "sx": sp[0], "sy": sp[1], "sz": sp[2],
                "orientation": "".join(nib.aff2axcodes(img.affine)), "ct_min": float(ct.min()), "ct_max": float(ct.max()),
                **body_slice_stats(ct, sp, LENGTHS)})
    del ct, img
    gc.collect()
    pet = nib.as_closest_canonical(load_nifti(s["image"][1])).get_fdata(dtype=np.float32)
    row["pet_max"] = float(pet.max())
    row["pet_min"] = float(pet.min())
    m = lab > 0
    row["pet_max_in_label"] = float(pet[m].max()) if m.any() else float("nan")
    del pet, lab, m
    gc.collect()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    seen = set(done.get("patient_id", []))
    samples = Hecktor2026Adapter(a.data_dir).training_samples()
    t0, n_new = time.time(), 0
    for s in samples:
        if s["patient_id"] in seen:
            continue
        rows.append(scan_case(s))
        n_new += 1
        if n_new % 20 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{len(rows)}/{len(samples)}  {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
