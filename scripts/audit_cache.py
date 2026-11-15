"""Read-only audit of the cropped .acf cache against the committed split and conditioning statistics.

    python scripts/audit_cache.py --cache_dir D:/hecktor_cache --out results/fable_run/audit_cache.json
    python scripts/audit_cache.py --cache_dir D:/hecktor_cache --out results/fable_run/audit_cache.json --scan

Without --scan only headers are read (seconds). With --scan every voxel of every case is read once to count
non-finite CT / PET values and the label values; the scan is resumable (progress is saved every 25 cases to
<out>.progress.json) and holds one x-slab of one case in memory at a time. Nothing is written to the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import crop_cache  # noqa: E402
from src.data.acf_dataset import SIDECAR_NAME, _rule  # noqa: E402

FEATURES = ("spacing_x", "spacing_y", "slice_thickness")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header_audit(cache_dir: Path, split_path: Path, stats_path: Path) -> dict:
    files = sorted(cache_dir.glob("*.acf"))
    heads = {f.stem: crop_cache.read_header(f) for f in files}
    split = json.loads(split_path.read_text())
    parts = {k: v for k, v in split.items() if not k.startswith("_")}
    ids = [i for v in parts.values() for i in v]
    out = {"n_acf_files": len(files), "n_tmp_files": len(list(cache_dir.glob("*.tmp"))),
           "split_counts": {k: len(v) for k, v in parts.items()},
           "split_duplicates": len(ids) - len(set(ids)),
           "split_overlap": {f"{a}&{b}": len(set(parts[a]) & set(parts[b]))
                             for a in parts for b in parts if a < b},
           "split_ids_missing_from_cache": sorted(set(ids) - set(heads)),
           "cache_ids_missing_from_split": sorted(set(heads) - set(ids)),
           "centres_per_split": {k: dict(sorted(Counter(heads[i]["centre"] for i in v if i in heads).items()))
                                 for k, v in parts.items()},
           "test_centres_in_train_or_val": sorted({heads[i]["centre"] for i in parts.get("test", []) if i in heads}
                                                  & {heads[i]["centre"] for k in ("train", "val") for i in parts.get(k, []) if i in heads}),
           "pet_dtypes": dict(Counter(h["arrays"]["pet"]["dtype"] for h in heads.values())),
           "label_values_in_headers": sorted({v for h in heads.values() for v in h["label_values"]}),
           "split_sha256": sha256(split_path), "stats_sha256": sha256(stats_path)}
    # conditioning statistics recomputed from the headers of the TRAIN ids only
    sp = np.array([heads[i]["spacing"] for i in parts["train"]], float)
    sp_all = np.array([heads[i]["spacing"] for i in ids if i in heads], float)
    stats = json.loads(stats_path.read_text())
    rec = {}
    for j, f in enumerate(FEATURES):
        rec[f] = {"mean_train": float(sp[:, j].mean()), "std_train_ddof0": float(sp[:, j].std(ddof=0)),
                  "std_train_ddof1": float(sp[:, j].std(ddof=1)), "mean_all_782": float(sp_all[:, j].mean()),
                  "file_CT": stats["CT"][f], "file_PT": stats.get("PT", {}).get(f)}
    out["conditioning"] = rec
    # The committed file was fitted from the NIfTI header zooms; the cache header stores the affine column norms.
    # The two differ by float32 rounding, so the comparison uses a tolerance and reports the largest difference.
    tol = 1e-4
    out["conditioning_max_abs_diff_vs_train_recompute"] = max(
        max(abs(r["mean_train"] - r["file_CT"]["mean"]), abs(r["std_train_ddof0"] - r["file_CT"]["std"]))
        for r in rec.values())
    out["conditioning_tolerance"] = tol
    out["conditioning_matches_train_only"] = out["conditioning_max_abs_diff_vs_train_recompute"] < tol
    out["conditioning_differs_from_all_782"] = any(abs(r["mean_all_782"] - r["file_CT"]["mean"]) > 1e-6 for r in rec.values())
    out["train_median_spacing"] = [float(x) for x in np.median(sp, axis=0)]
    out["all_median_spacing"] = [float(x) for x in np.median(sp_all, axis=0)]
    # PET-scale sidecar
    side = cache_dir / SIDECAR_NAME
    if side.exists():
        d = json.loads(side.read_text())
        sc = d.get("scales", {})
        stale = [p for p, e in sc.items() if p in heads and e["bytes"] != (cache_dir / f"{p}.acf").stat().st_size]
        vals = np.array([e["scale"] for e in sc.values()], float)
        out["sidecar"] = {"present": True, "rule_matches_code": d.get("rule") == _rule(), "n_entries": len(sc),
                          "missing": sorted(set(heads) - set(sc)), "stale": stale,
                          "non_finite_or_non_positive": int((~np.isfinite(vals) | (vals <= 0)).sum()),
                          "min": float(vals.min()), "median": float(np.median(vals)), "max": float(vals.max()),
                          "sha256": sha256(side)}
    else:
        out["sidecar"] = {"present": False}
    return out


def voxel_scan(cache_dir: Path, progress_path: Path, chunk: int = 32, log=print) -> dict:
    done = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    files = sorted(cache_dir.glob("*.acf"))
    t0 = time.perf_counter()
    for n, f in enumerate(files):
        if f.stem in done:
            continue
        _, arr = crop_cache.open_case(f)
        bad_pet = bad_ct = 0
        labs: set[int] = set()
        pmin, pmax = np.inf, -np.inf
        for x0 in range(0, arr["pet"].shape[0], chunk):
            p = np.asarray(arr["pet"][x0:x0 + chunk])
            bad_pet += int((~np.isfinite(p)).sum())
            if np.isfinite(p).any():
                pmin, pmax = min(pmin, float(p[np.isfinite(p)].min())), max(pmax, float(p[np.isfinite(p)].max()))
            c = np.asarray(arr["ct"][x0:x0 + chunk])
            if c.dtype.kind == "f":
                bad_ct += int((~np.isfinite(c)).sum())
            labs.update(int(v) for v in np.unique(np.asarray(arr["label"][x0:x0 + chunk])))
        done[f.stem] = {"pet_nonfinite": bad_pet, "ct_nonfinite": bad_ct, "labels": sorted(labs),
                        "pet_min": pmin, "pet_max": pmax}
        del arr
        if (n + 1) % 25 == 0:
            progress_path.write_text(json.dumps(done))
            log(f"{len(done)}/{len(files)} scanned, {time.perf_counter() - t0:.0f} s")
    progress_path.write_text(json.dumps(done))
    return {"n_scanned": len(done), "cases_with_nonfinite_pet": sorted(k for k, v in done.items() if v["pet_nonfinite"]),
            "cases_with_nonfinite_ct": sorted(k for k, v in done.items() if v["ct_nonfinite"]),
            "label_values": sorted({x for v in done.values() for x in v["labels"]}),
            "cases_with_negative_pet": sorted(k for k, v in done.items() if v["pet_min"] < 0),
            "pet_min_overall": min(v["pet_min"] for v in done.values()),
            "pet_max_overall": max(v["pet_max"] for v in done.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True, type=Path)
    ap.add_argument("--split", type=Path, default=ROOT / "configs/splits/hecktor2026_seed42.json")
    ap.add_argument("--stats", type=Path, default=ROOT / "configs/splits/hecktor2026_seed42_conditioning_stats.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    res["headers"] = header_audit(a.cache_dir, a.split, a.stats)
    if a.scan:
        res["scan"] = voxel_scan(a.cache_dir, a.out.with_suffix(".progress.json"))
    a.out.write_text(json.dumps(res, indent=1))
    show = {k: v for k, v in res["headers"].items()}
    print(json.dumps(show, indent=1))
    if "scan" in res:
        print(json.dumps(res["scan"], indent=1))


if __name__ == "__main__":
    main()
