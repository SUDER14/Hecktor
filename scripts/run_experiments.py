"""Run experiments E1-E6 over trained runs.

    python scripts/run_experiments.py e1 --runs B1=runs/b1_unet_resample B4=runs/b4_acf_cond --out results/e1
    python scripts/run_experiments.py e2 --runs ... --out results/e2        # anisotropy 2/3/5/7 mm
    python scripts/run_experiments.py e3 --runs ... --device cuda --out results/e3
    python scripts/run_experiments.py e4 --runs B4=... B5=... --out results/e4
    python scripts/run_experiments.py e5 --per-case results/e1/e1_per_case.csv --pairs B4:B3 B5:B4 --out results/e5
    python scripts/run_experiments.py e6 --per-case results/e1/e1_per_case.csv --reference B4 --out results/e6

E5/E6 read an E1-style per-case CSV (columns: model, patient_id, centre, dice, ...), so they can
be re-run without re-running inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.eval import experiments as ex  # noqa: E402


def parse_runs(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--runs entries are LABEL=run_dir, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", choices=["e1", "e2", "e3", "e4", "e5", "e6"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--per-case", default=None)
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B model-label pairs for e5")
    ap.add_argument("--reference", default=None)
    a = ap.parse_args()
    runs = parse_runs(a.runs)
    if a.exp == "e1":
        print(ex.e1_heldout(runs, a.out, a.split, a.device, a.max_cases).groupby("model")[["dice", "hd95_mm"]].mean())
    elif a.exp == "e2":
        ex.e2_anisotropy(runs, a.out, a.split, device=a.device, max_cases=a.max_cases)
        print(pd.read_csv(Path(a.out) / "e2_summary.csv"))
    elif a.exp == "e3":
        print(ex.e3_resources(runs, a.out, a.device))
    elif a.exp == "e4":
        print(ex.e4_calibration(runs, a.out, a.split, a.device, a.max_cases))
    elif a.exp == "e5":
        pairs = [tuple(p.split(":")) for p in a.pairs]
        print(ex.e5_ablations(pd.read_csv(a.per_case), pairs, a.out))
    else:
        print(ex.e6_statistics(pd.read_csv(a.per_case), a.out, a.reference))


if __name__ == "__main__":
    main()
