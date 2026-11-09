"""Verify candidate head-and-neck crop rules against the labels (read-only, uses crop_scan.csv).

    python scripts/crop_verify.py --scan results/crop/crop_scan.csv [--length 300 --width 300]

For every (length, width) candidate: number of cases whose label bounding box is NOT fully inside the
rule's box, the worst margins (mm; negative = outside), and the cropped voxel total (a size proxy). The
rule itself never sees a label; if a candidate fails, pick different rule parameters for everyone --
never a per-case box. Cases with an empty label are counted and skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.crop_rule import box_from_stats, margin_mm  # noqa: E402

LENGTHS = (200, 250, 300, 350, 400)
WIDTHS = (240, 280, 300, 320, 350, 400)


def evaluate(df: pd.DataFrame, length: int, width: int) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        shape, sp = (r.nx, r.ny, r.nz), (r.sx, r.sy, r.sz)
        box = box_from_stats(shape, sp, int(r.top_k), getattr(r, f"cx_{length}"), getattr(r, f"cy_{length}"), length, width)
        nvox = 1
        for b0, b1 in box:
            nvox *= b1 - b0
        row = {"patient_id": r.patient_id, "centre": r.centre, "box_voxels": nvox}
        if r.n_label_voxels > 0:
            lab = ((r.lab_x0, r.lab_x1), (r.lab_y0, r.lab_y1), (r.lab_z0, r.lab_z1))
            row["margin_mm"] = margin_mm(box, sp, lab)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--length", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--worst", type=int, default=5)
    a = ap.parse_args()
    df = pd.read_csv(a.scan)
    n_empty = int((df.n_label_voxels == 0).sum())
    print(f"{len(df)} cases scanned, {n_empty} with an empty label (skipped in the check)\n")
    combos = [(a.length, a.width)] if a.length and a.width else [(l, w) for l in LENGTHS for w in WIDTHS]
    summary = []
    for length, width in combos:
        e = evaluate(df, length, width)
        lab = e.dropna(subset=["margin_mm"])
        fails = lab[lab.margin_mm < 0].sort_values("margin_mm")
        summary.append({"length_mm": length, "width_mm": width, "n_labelled": len(lab), "n_fail": len(fails),
                        "min_margin_mm": lab.margin_mm.min(), "p1_margin_mm": lab.margin_mm.quantile(0.01),
                        "box_voxels_total_G": e.box_voxels.sum() / 1e9,
                        "uncompressed_GB_at_5B": e.box_voxels.sum() * 5 / 1e9})
        if len(combos) == 1 and len(fails):
            print("failing cases (worst first):\n", fails.head(a.worst * 4).to_string(index=False))
        if len(combos) == 1:
            print("\nsmallest margins (mm):\n", lab.sort_values("margin_mm").head(a.worst).to_string(index=False))
            print("\nper-centre box voxels and failures:\n",
                  e.assign(fail=e.margin_mm < 0).groupby("centre").agg(n=("patient_id", "size"), fails=("fail", "sum"),
                                                                     mean_box_Mvox=("box_voxels", lambda x: x.mean() / 1e6)).round(2).to_string())
    print(pd.DataFrame(summary).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
