"""Compute every case's PET scale once and store it next to the cache (`<cache_dir>/pet_scale.json`).

    python scripts/precompute_pet_scales.py --cache_dir D:/hecktor_cache

Per case: the 99.5th percentile of positive PET voxels inside the CT body mask (CT > -500 HU); see
src/data/acf_dataset.py. Resumable (valid entries are kept, progress is written every 20 cases), one case at a
time, no worker processes. An entry is treated as stale if the .acf file size changed since it was computed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.acf_dataset import SIDECAR_NAME, write_scale_sidecar  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    a = ap.parse_args()
    t0 = time.time()
    sc = write_scale_sidecar(a.cache_dir, log=lambda m: print(f"{time.time() - t0:.0f}s {m}", flush=True))
    print(f"done: {len(sc)} scales -> {Path(a.cache_dir) / SIDECAR_NAME} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
