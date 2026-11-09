"""Measure training data-loading speed from the real .acf cache: seconds per fetch with 0, 2 and 3 DataLoader workers.

    python scripts/measure_data_rate.py --config configs/shared_b4_acf_cond.yaml --out results/timing/rate_b4.json

A fetch = one patient's `patch.num_samples` patches (= one micro-batch at the configured batch). Warm-up fetches are
discarded. RAM guard: the sum of RSS over this process and its children is polled; if it exceeds --ram_limit_gb
(default 6.5) the run is aborted and that worker count is reported as ABORTED, never retried. Runs one worker setting at
a time. Each fetch is a different patient (sequential order), so page-cache warmth is the same across settings only
after the first pass: the order of settings is 0, 2, 3 and each uses its own slice of cases.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from monai.data import DataLoader  # noqa: E402

from src.data.acf_dataset import build_acf_datasets  # noqa: E402


def tree_rss_gb() -> float:
    p = psutil.Process()
    return (p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))) / 2**30


class Guard(threading.Thread):
    def __init__(self, limit_gb: float):
        super().__init__(daemon=True)
        self.limit, self.peak, self.tripped, self._stop_evt = limit_gb, 0.0, False, threading.Event()

    def run(self):
        while not self._stop_evt.is_set():
            try:
                self.peak = max(self.peak, tree_rss_gb())
            except psutil.Error:
                pass
            if self.peak > self.limit:
                self.tripped = True
                return
            time.sleep(0.2)

    def stop(self):
        self._stop_evt.set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 2, 3])
    ap.add_argument("--fetches", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--ram_limit_gb", type=float, default=6.5)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    torch.set_num_threads(1)
    res = {"config": a.config, "mode": cfg["data"]["mode"], "num_samples": cfg["patch"]["num_samples"],
           "ram_limit_gb": a.ram_limit_gb, "settings": {}}
    n_per = a.fetches + a.warmup
    for k, w in enumerate(a.workers):
        ds = build_acf_datasets(cfg)["train"]
        ds.set_epoch(1)
        lo = k * n_per
        ds.samples, ds.heads = ds.samples[lo:lo + n_per], ds.heads[lo:lo + n_per]   # a distinct slice of cases per setting
        guard = Guard(a.ram_limit_gb)
        guard.start()
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=w, persistent_workers=w > 0, prefetch_factor=2 if w else None)
        t0, n, aborted = None, 0, False
        for i, _ in enumerate(dl):
            if guard.tripped:
                aborted = True
                break
            if i == a.warmup - 1:
                t0 = time.perf_counter()
            elif i >= a.warmup:
                n += 1
        elapsed = time.perf_counter() - t0 if t0 else float("nan")
        guard.stop()
        res["settings"][str(w)] = ({"ABORTED": "RAM above limit", "peak_rss_gb": guard.peak} if aborted or guard.tripped else
                                   {"fetches_timed": n, "seconds_per_fetch": elapsed / max(n, 1), "peak_rss_gb": guard.peak})
        print(w, res["settings"][str(w)], flush=True)
        del dl, ds
        if aborted:
            break
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
