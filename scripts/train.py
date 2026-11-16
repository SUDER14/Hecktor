"""Portable training entry point (step-based, resumable at any minute). Logic: src/training/portable.py.

    python scripts/train.py --config configs/shared_b4_acf_cond.yaml --data-root <folder with the .acf cache> \
        --out-root <folder for runs> --resume auto
    python scripts/train.py --config configs/shared_b4_acf_cond.yaml --out-root <folder> --synthetic \
        --set steps.max_steps=40 --set steps.val_every=20

No path is hard-coded: the cache comes from --data-root, everything written goes under --out-root/runs/<name>/.
A real run refuses a dirty git tree, and refuses to start on a hosted notebook machine unless ALLOW_REAL_DATA=1.
--synthetic runs the whole loop (training, validation, checkpoints, resume) on random tensors: no data, no cache.
The test set is never loaded by this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.portable import train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data-root", type=Path, default=None, help="folder holding <case>.acf and pet_scale.json")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--resume", default="auto", help="auto (default), none, or a checkpoint path")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                    help="override an existing config key, e.g. steps.max_steps=40 (recorded in the manifest)")
    ap.add_argument("--name", default=None, help="run name; default <config stem>_seed<seed>")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    a = ap.parse_args()
    run_dir = train(a.config, a.data_root, a.out_root, a.resume, a.synthetic, a.overrides, a.name, a.device)
    print(f"run directory: {run_dir}")


if __name__ == "__main__":
    main()
