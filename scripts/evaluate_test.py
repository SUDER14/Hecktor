"""Locked test evaluation, run ONCE per trained model. Logic and the full rules: src/eval/locked_test.py.

    python scripts/evaluate_test.py --run-dir <out-root>/runs/shared_b4_acf_cond_seed42 \
        --data-root <folder with the .acf cache> --out-root <folder OUTSIDE the repo>

    # a finished test run can be repeated only with a logged reason
    python scripts/evaluate_test.py ... --rerun "HD95 bug fixed in commit abc1234"

    # nnU-Net label maps through the same metric code and the same lock
    python scripts/evaluate_test.py --pred-dir <folder of <case>.nii.gz> --name nnunet_fold0 \
        --data-root <cache> --out-root <folder>

    # no data, no cache, no git gates: proves the script runs end to end
    python scripts/evaluate_test.py --synthetic --config configs/shared_b4_acf_cond.yaml --out-root <folder> \
        --set "patch.spatial_size=[32, 32, 32]" --set "patch.min_extent_mm=[8.0, 8.0, 8.0]"

Refuses to run unless docs/ANALYSIS_PLAN.md is committed and unmodified and the git tree is clean. Loads only
<run-dir>/checkpoints/best.pt. Tables go to results/test/<run>/ (per_case.csv, sweep.csv, wrong_spacing.csv,
calibration.json, uncertainty.json, summary.json, manifest.json, LOCK.json); predictions and uncertainty maps go
under --out-root, never into the repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.locked_test import evaluate, evaluate_predictions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None, help="a finished run folder written by scripts/train.py")
    ap.add_argument("--data-root", type=Path, default=None, help="folder holding <case>.acf and pet_scale.json")
    ap.add_argument("--out-root", required=True, type=Path, help="outside the repo; predictions and uncertainty maps go here")
    ap.add_argument("--rerun", default=None, metavar="REASON", help="repeat a finished test run; the reason is logged")
    ap.add_argument("--no-save-maps", action="store_true", help="do not write prediction / uncertainty NIfTI files")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--pred-dir", type=Path, default=None, help="score external label maps (nnU-Net) instead of a run")
    ap.add_argument("--name", default=None, help="result name for --pred-dir")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--config", type=Path, default=None, help="--synthetic only: model config, weights freshly initialised")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides", help="--synthetic only")
    a = ap.parse_args()
    if a.pred_dir is not None:
        if not a.name:
            raise SystemExit("--pred-dir needs --name")
        res = evaluate_predictions(a.pred_dir, a.name, a.data_root, a.out_root, a.synthetic, a.rerun)
    else:
        res = evaluate(a.run_dir, a.data_root, a.out_root, a.synthetic, a.config, a.overrides, a.rerun, a.device,
                       save_maps=not a.no_save_maps)
    print(f"results: {res}")


if __name__ == "__main__":
    main()
