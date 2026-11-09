"""Print the measured tables used in docs/DATA_REPORT.md as markdown (read-only; reads CSV/JSON only).

    python scripts/make_data_report_tables.py [--results results]

Inputs (all produced by scripts in this repo, nothing typed in by hand):
    results/manifest_hecktor2026/spacing_manifest.csv   scripts/spacing_manifest.py   (all 782 cases, headers)
    results/manifest_hecktor2026/label_stats.csv        scripts/data_stats.py --label_only (all cases)
    results/manifest_hecktor2026/voxel_stats.csv        scripts/data_stats.py --per_centre (SAMPLE of cases)
    results/pet_roundtrip/pet_roundtrip.csv             scripts/pet_roundtrip.py   (20 cases)
    results/cache_probe/cache_probe.json                scripts/measure_cache.py   (20 cases)
Missing inputs are skipped with a note, never filled in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def md(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    """Markdown table (needs `tabulate`); falls back to a fenced plain table without it."""
    try:
        return df.to_markdown(floatfmt=floatfmt)
    except ImportError:
        return "\n".join(["```", df.to_string(), "```"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    r = Path(ap.parse_args().results)
    m = r / "manifest_hecktor2026"

    if (m / "spacing_manifest.csv").exists():
        d = pd.read_csv(m / "spacing_manifest.csv")
        ct = d[d.modality == "CT"]
        g = ct.groupby("centre")
        t = pd.DataFrame({
            "n": g.size(), "sx_min": g.sx.min(), "sx_med": g.sx.median(), "sx_max": g.sx.max(),
            "sz_min": g.sz.min(), "sz_med": g.sz.median(), "sz_max": g.sz.max(),
            "nxy": g.apply(lambda x: "/".join(map(str, sorted(set(x.nx))))),
            "nz_min": g.nz.min(), "nz_max": g.nz.max(),
            "orient": g.orientation.agg(lambda x: ",".join(f"{k}:{v}" for k, v in x.value_counts().items())),
            "oblique": g.oblique.sum()})
        print("## Header manifest per centre (CT; PET and label identical, see below)\n\n" + md(t) + "\n")
        same = d.groupby("patient_id")[["nx", "ny", "nz", "affine"]].nunique().max().max()
        print(f"Cases whose CT/PET/label differ in shape or affine: {int((d.groupby('patient_id')[['nx','ny','nz','affine']].nunique() > 1).any(axis=1).sum())} (max distinct values per case {same})\n")
        print("Distinct (sx,sy,sz) combinations per centre:\n")
        print(md(ct.assign(sp=ct.sx.round(3).astype(str) + "x" + ct.sy.round(3).astype(str) + "x" + ct.sz.round(2).astype(str))
                 .groupby("centre").sp.nunique().to_frame("n_distinct_spacings")) + "\n")

    if (m / "label_stats.csv").exists():
        d = pd.read_csv(m / "label_stats.csv")
        g = d.groupby("centre")
        t = pd.DataFrame({"n_cases": g.size(), "values": g.lab_values.agg(lambda x: ",".join(sorted(set("|".join(x).split("|"))))),
                          "voxels_label1": g.n_label1.sum(), "voxels_label2": g.n_label2.sum(),
                          "cases_with_1": g.n_label1.agg(lambda x: int((x > 0).sum())),
                          "cases_with_2": g.n_label2.agg(lambda x: int((x > 0).sum())),
                          "cases_empty": g.apply(lambda x: int(((x.n_label1 + x.n_label2) == 0).sum()))})
        print(f"## Label values per centre (ALL {len(d)} cases)\n\n" + md(t) + "\n")
        d["ml1"] = d.n_label1 * d.voxel_ml
        d["ml2"] = d.n_label2 * d.voxel_ml
        print("Volume per case with the label present (mL): median / max\n")
        print(md(pd.DataFrame({"label1_median_ml": d[d.n_label1 > 0].groupby("centre").ml1.median(),
                               "label1_max_ml": d[d.n_label1 > 0].groupby("centre").ml1.max(),
                               "label2_median_ml": d[d.n_label2 > 0].groupby("centre").ml2.median(),
                               "label2_max_ml": d[d.n_label2 > 0].groupby("centre").ml2.max()})) + "\n")

    if (m / "voxel_stats.csv").exists():
        d = pd.read_csv(m / "voxel_stats.csv")
        g = d.groupby("centre")
        t = pd.DataFrame({"n_sampled": g.size(), "ct_min": g.ct_min.min(), "ct_max": g.ct_max.max(),
                          "ct_dtypes": g.ct_dtype.agg(lambda x: ",".join(sorted(set(x)))),
                          "pt_min": g.pt_min.min(), "pt_max_median": g.pt_max.median(), "pt_max_max": g.pt_max.max(),
                          "pt_p999_sub_median": g.pt_p999_sub.median(), "pt_nan": g.pt_nan.sum(),
                          "pt_mean_in_label1_median": g.pt_mean_in_label1.median(),
                          "pt_mean_in_label2_median": g.pt_mean_in_label2.median()})
        print(f"## PET / CT value ranges per centre (SAMPLE: {len(d)} of the cases, seeded)\n\n" + md(t) + "\n")

    if (r / "pet_roundtrip" / "pet_roundtrip.csv").exists():
        d = pd.read_csv(r / "pet_roundtrip" / "pet_roundtrip.csv")
        cols = [c for c in ("pearson_r", "rmse", "max_abs_err", "max_abs_err_rel_pet_max", "pearson_r_gtv", "rmse_gtv",
                            "gtv_max_rel_change", "gtv_mean_rel_change") if c in d]
        print(f"## PET round trip (down to coarse grid, back up), {len(d)} cases\n\n" + md(d[cols].agg(["mean", "min", "max"]).T) + "\n")

    if (r / "cache_probe" / "cache_probe.json").exists():
        j = json.loads((r / "cache_probe" / "cache_probe.json").read_text())
        t = pd.DataFrame({k: {x: v[x] for x in ("mean_s_per_case", "mean_mib_per_case", "projected_full_gb", "projected_full_hours")}
                          for k, v in j["variants"].items()}).T
        print(f"## Cache measurement ({j['n_cases']} cases) and projection to all cases\n\n" + md(t) + "\n")
        print("Gate:", json.dumps(j["gate"]), "| free GB before:", j["free_gb_before"])


if __name__ == "__main__":
    main()
