"""Build (or measure) the cropped native-grid cache. One case at a time, resumable, logged.

    # measure on 20 stratified cases, print projections, delete the probe:
    python scripts/build_hecktor_cache.py --data_dir "<zip>" --cache_dir D:/hecktor_cache --length 300 --width 300 --measure 20
    # full build (only after the size decision):
    python scripts/build_hecktor_cache.py --data_dir "<zip>" --cache_dir D:/hecktor_cache --length 300 --width 300

Progress: <dir>/_progress.json (rewritten every 20 cases) and <cache_dir>/build.log. Finished, valid files are
skipped on restart. A full build refuses to start if the last measurement projected more than 80 GB or
less than 20 GB free on the cache drive, unless --i_know_the_gate is given.
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
from src.data import crop_cache  # noqa: E402
from src.data.hecktor2026 import Hecktor2026Adapter  # noqa: E402

GATE_GB, MIN_FREE_GB = 80.0, 20.0


def log(fh, msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--length", type=float, required=True)
    ap.add_argument("--width", type=float, required=True)
    ap.add_argument("--measure", type=int, default=0, help="stratified probe of N cases, then delete it")
    ap.add_argument("--i_know_the_gate", action="store_true")
    ap.add_argument("--out", default="results/crop/cache_measure.json")
    a = ap.parse_args()
    root = Path(a.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    d = root / "_probe_crop" if a.measure else root
    d.mkdir(parents=True, exist_ok=True)
    if not a.measure and not a.i_know_the_gate:
        m = Path(a.out)
        if not m.exists():
            raise SystemExit("no measurement found: run with --measure 20 first")
        r = json.loads(m.read_text())
        if r["gate_over_80gb"] or r["gate_free_below_20gb"]:
            raise SystemExit(f"gate: last measurement projected {r['projected_total_gb']:.1f} GB "
                             f"(D: free after {r['d_free_gb_after_full_build']:.1f} GB); ask before building")
    samples = Hecktor2026Adapter(a.data_dir).training_samples()
    n_by_c: dict[str, int] = {}
    for s in samples:
        n_by_c[s["centre"]] = n_by_c.get(s["centre"], 0) + 1
    todo = stratified_sample(samples, a.measure, 42) if a.measure else samples
    sig = crop_cache.signature(a.length, a.width)
    rows, t0 = [], time.time()
    with open(root / "build.log", "a") as fh:
        log(fh, f"start: {len(todo)} cases, rule length={a.length} width={a.width}, dir={d}")
        for i, s in enumerate(todo, 1):
            p = d / f"{s['patient_id']}.acf"
            if crop_cache.is_valid(p, sig):
                h = crop_cache.read_header(p)
                rows.append({"patient_id": s["patient_id"], "centre": s["centre"], "bytes": p.stat().st_size,
                             "seconds": float("nan"), "shape": h["shape"], "orig_shape": h["orig_shape"]})
                continue
            rows.append(crop_cache.build_case(s, d, a.length, a.width))
            if i % 20 == 0 or i == len(todo):
                (d / "_progress.json").write_text(json.dumps({"done": len(rows), "total": len(todo),
                                                              "elapsed_s": time.time() - t0}))
                log(fh, f"{i}/{len(todo)} last={s['patient_id']} {rows[-1]['bytes'] / 2**20:.0f} MiB "
                        f"{rows[-1]['seconds']:.1f}s")
        if a.measure:
            df = pd.DataFrame(rows)
            g = df.groupby("centre")
            proj_gb = sum(g.bytes.mean()[c] * n_by_c[c] for c in g.groups) / 2**30
            proj_h = sum(g.seconds.mean()[c] * n_by_c[c] for c in g.groups) / 3600
            free_now = shutil.disk_usage(str(root)).free / 2**30
            res = {"n": len(df), "length_mm": a.length, "width_mm": a.width,
                   "mean_mib_per_case": float(df.bytes.mean() / 2**20), "mean_s_per_case": float(df.seconds.mean()),
                   "projected_total_gb": float(proj_gb), "projected_hours": float(proj_h), "d_free_gb_now": float(free_now),
                   "d_free_gb_after_full_build": float(free_now - proj_gb), "gate_over_80gb": bool(proj_gb > GATE_GB),
                   "gate_free_below_20gb": bool(free_now - proj_gb < MIN_FREE_GB),
                   "per_centre_mib": {c: float(g.bytes.mean()[c] / 2**20) for c in g.groups}, "cases": list(df.patient_id)}
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
            log(fh, json.dumps({k: v for k, v in res.items() if k not in ("cases", "per_centre_mib")}))
            shutil.rmtree(d, ignore_errors=True)
            log(fh, f"probe deleted; D: free now {shutil.disk_usage(str(root)).free / 2**30:.1f} GB")
        else:
            log(fh, f"done {len(rows)} cases, {sum(r['bytes'] for r in rows) / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
