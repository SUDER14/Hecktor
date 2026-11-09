"""Measure preprocessing time and cached size on N stratified cases, then project the full cache.

    python scripts/measure_cache.py --data_dir "<zip>" --cache_root D:/hecktor_cache/_probe --n 20

Builds the cache entries for the SAME cases under each variant (shared native, rederived native,
resampled) with the real production code (src/data/shared_cache.py, src/data/dual_cache.py), reads back
the bytes actually written, and scales per centre: projection = sum over centres of
(mean bytes of sampled cases in that centre) x (cases in that centre). Prints the gate decision
(> 80 GB, or free space left < 20 GB). Nothing is estimated from voxel counts here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from scripts.pet_roundtrip import stratified_sample  # noqa: E402
from src.data import dual_cache, shared_cache  # noqa: E402
from src.data.hecktor2026 import Hecktor2026Adapter  # noqa: E402

GATE_GB, MIN_FREE_GB = 80.0, 20.0


def entry_bytes(d: Path, pid: str) -> int:
    return sum(f.stat().st_size for f in d.glob(f"{pid}.*"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/cache_probe/cache_probe.json")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    root = Path(a.cache_root)
    allsamples = Hecktor2026Adapter(a.data_dir).training_samples()
    n_by_centre = {}
    for s in allsamples:
        n_by_centre[s["centre"]] = n_by_centre.get(s["centre"], 0) + 1
    picked = stratified_sample(allsamples, a.n, 42)
    base = yaml.safe_load(Path("configs/b4_acf_cond.yaml").read_text())
    variants = {
        "shared_native": (shared_cache.build_entry, {"data": {"mode": "native", "pet_mode": "shared"}}, "native"),
        "rederived_native": (dual_cache.build_entry, {"data": {"mode": "native", "pet_mode": "rederived",
                                                              "pet_spacing": [4.0, 4.0, 4.0]}}, "native"),
        "resampled": (shared_cache.build_entry, {"data": {"mode": "resampled", "pet_mode": "shared"}}, "resampled"),
    }
    free0 = shutil.disk_usage(root.anchor or ".").free / 2**30
    res = {"n_cases": len(picked), "cases": [s["patient_id"] for s in picked], "n_by_centre": n_by_centre,
           "free_gb_before": round(free0, 1), "variants": {}}
    for name, (fn, cfg_over, mode) in variants.items():
        cfg = {**base, **cfg_over, "seed": 42}
        d = root / name
        rows = []
        for s in picked:
            t0 = time.perf_counter()
            fn(s, cfg, d, mode, "float16", 42)
            rows.append({"patient_id": s["patient_id"], "centre": s["centre"], "seconds": time.perf_counter() - t0,
                         "bytes": entry_bytes(d, s["patient_id"])})
            print(name, rows[-1]["patient_id"], round(rows[-1]["seconds"], 1), "s", round(rows[-1]["bytes"] / 2**20), "MiB", flush=True)
        by_c: dict[str, list[dict]] = {}
        for r in rows:
            by_c.setdefault(r["centre"], []).append(r)
        proj_b = sum(sum(x["bytes"] for x in v) / len(v) * n_by_centre[c] for c, v in by_c.items())
        proj_s = sum(sum(x["seconds"] for x in v) / len(v) * n_by_centre[c] for c, v in by_c.items())
        res["variants"][name] = {"mean_s_per_case": sum(r["seconds"] for r in rows) / len(rows),
                                 "mean_mib_per_case": sum(r["bytes"] for r in rows) / len(rows) / 2**20,
                                 "projected_full_gb": proj_b / 2**30, "projected_full_hours": proj_s / 3600, "rows": rows}
    tot = sum(v["projected_full_gb"] for v in res["variants"].values())
    res["projected_all_variants_gb"] = tot
    res["gate"] = {"over_80gb_any_variant": any(v["projected_full_gb"] > GATE_GB for v in res["variants"].values()),
                   "free_after_largest_gb": free0 - max(v["projected_full_gb"] for v in res["variants"].values()),
                   "free_after_all_gb": free0 - tot, "min_free_gb": MIN_FREE_GB}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("variants", "cases")}, indent=1))
    for k, v in res["variants"].items():
        print(k, {x: round(y, 2) for x, y in v.items() if x != "rows"})
    if not a.keep:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
