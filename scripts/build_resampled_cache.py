"""Measure (default) or build the resampled B1/B2 cache from the native `.acf` cache.

    # measure on 20 stratified seeded cases, print size/time/projection, delete the probe (does NOT build):
    python scripts/build_resampled_cache.py --src D:/hecktor_cache --out D:/hecktor_cache_resampled --measure 20

    # full build: only after approval; refuses past the gates (> 80 GB projected, or < 20 GB free afterwards)
    python scripts/build_resampled_cache.py --src D:/hecktor_cache --out D:/hecktor_cache_resampled

One case at a time, resumable (valid files are skipped; progress file rewritten every 20 cases), log in <out>/build.log.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from scripts.pet_roundtrip import stratified_sample  # noqa: E402
from src.data import crop_cache, resampled_cache  # noqa: E402

GATE_GB, MIN_FREE_GB = 80.0, 20.0
TARGET = (0.9766, 0.9766, 3.27)   # manifest median over all 782 scans (VERIFIED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=float, nargs=3, default=list(TARGET))
    ap.add_argument("--measure", type=int, default=0)
    ap.add_argument("--i_know_the_gate", action="store_true")
    ap.add_argument("--report", default="results/timing/resampled_cache_measure.json")
    a = ap.parse_args()
    src, root = Path(a.src), Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    d = root / "_probe" if a.measure else root
    d.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.acf"))
    heads = {f.stem: crop_cache.read_header(f) for f in files}
    n_by_c: dict[str, int] = {}
    for h in heads.values():
        n_by_c[h["centre"]] = n_by_c.get(h["centre"], 0) + 1
    samples = [{"patient_id": k, "centre": h["centre"]} for k, h in heads.items()]
    todo = stratified_sample(samples, a.measure, 42) if a.measure else samples
    sig = resampled_cache.signature(a.target)
    if not a.measure and not a.i_know_the_gate:
        m = Path(a.report)
        if not m.exists():
            raise SystemExit("no measurement found: run with --measure 20 first")
        r = json.loads(m.read_text())
        if r["gate_over_80gb"] or r["gate_free_below_20gb"]:
            raise SystemExit(f"gate: projected {r['projected_total_gb']:.1f} GB, D: free after {r['free_gb_after_full_build']:.1f} GB; ask first")
    rows, t0 = [], time.time()
    with open(root / "build.log", "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} start {len(todo)} cases target={a.target} dir={d}\n")
        for i, s in enumerate(todo, 1):
            out = d / f"{s['patient_id']}.acf"
            if out.exists() and crop_cache.read_header(out).get("sig") == sig:
                rows.append({"patient_id": s["patient_id"], "centre": s["centre"], "bytes": out.stat().st_size, "seconds": float("nan")})
                continue
            rows.append(resampled_cache.build_case(src / f"{s['patient_id']}.acf", d, a.target))
            if i % 20 == 0 or i == len(todo):
                (d / "_progress.json").write_text(json.dumps({"done": len(rows), "total": len(todo), "elapsed_s": time.time() - t0}))
                fh.write(f"{time.strftime('%H:%M:%S')} {i}/{len(todo)} {s['patient_id']} {rows[-1]['bytes'] / 2**20:.0f} MiB\n")
                fh.flush()
        if a.measure:
            df = pd.DataFrame(rows)
            g = df.groupby("centre")
            proj_gb = sum(g.bytes.mean()[c] * n_by_c[c] for c in g.groups) / 2**30
            proj_h = sum(g.seconds.mean()[c] * n_by_c[c] for c in g.groups) / 3600
            free = shutil.disk_usage(str(root)).free / 2**30
            res = {"n": len(df), "target_spacing": a.target, "mean_mib_per_case": float(df.bytes.mean() / 2**20),
                   "mean_s_per_case": float(df.seconds.mean()), "projected_total_gb": float(proj_gb),
                   "projected_hours": float(proj_h), "free_gb_now": float(free), "free_gb_after_full_build": float(free - proj_gb),
                   "gate_over_80gb": bool(proj_gb > GATE_GB), "gate_free_below_20gb": bool(free - proj_gb < MIN_FREE_GB),
                   "per_centre_mib": {c: float(g.bytes.mean()[c] / 2**20) for c in g.groups}, "cases": list(df.patient_id)}
            Path(a.report).parent.mkdir(parents=True, exist_ok=True)
            Path(a.report).write_text(json.dumps(res, indent=1))
            print(json.dumps({k: v for k, v in res.items() if k not in ("cases", "per_centre_mib")}, indent=1))
            shutil.rmtree(d, ignore_errors=True)
            print(f"probe deleted; free now {shutil.disk_usage(str(root)).free / 2**30:.1f} GB")
        else:
            print(f"done {len(rows)} cases, {sum(r['bytes'] for r in rows) / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
