"""Re-store, as float32 PET, every cached case whose header PET maximum exceeds a threshold (default 60,000).

    python scripts/restore_pet_float32.py --data_dir "<zip>" --cache_dir D:/hecktor_cache [--threshold 60000]

Reads the header of every .acf (cheap), selects cases with header pet_max > threshold that are still float16, and
rebuilds ONLY those through src/data/crop_cache.build_case (atomic replace, same crop rule). All other files are
left untouched. One case at a time; resumable (cases already float32 are skipped). Nothing is clipped or saturated.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import crop_cache  # noqa: E402
from src.data.hecktor2026 import Hecktor2026Adapter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--threshold", type=float, default=60000.0)
    ap.add_argument("--length", type=float, default=350.0)
    ap.add_argument("--width", type=float, default=300.0)
    a = ap.parse_args()
    cd = Path(a.cache_dir)
    samples = {s["patient_id"]: s for s in Hecktor2026Adapter(a.data_dir).training_samples()}
    todo = []
    for f in sorted(cd.glob("*.acf")):
        h = crop_cache.read_header(f)
        if h["pet_max"] > a.threshold and h["arrays"]["pet"]["dtype"] == "float16":
            todo.append((f.stem, h["pet_max"]))
    print(f"{len(todo)} cases to re-store as float32 (header PET max > {a.threshold:g})", flush=True)
    for i, (pid, pmax) in enumerate(todo, 1):
        t0 = time.time()
        row = crop_cache.build_case(samples[pid], cd, a.length, a.width, pet_float32_above=a.threshold)
        print(f"{i}/{len(todo)} {pid} header_pet_max={pmax:.0f} -> {row['bytes'] / 2**20:.0f} MiB {time.time() - t0:.1f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
