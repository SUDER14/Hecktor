"""Score nnU-Net predictions with OUR metric code, on the same native label as every other model.

    python scripts/nnunet_evaluate.py --data_dir data/hecktor2026_training --split configs/splits/hecktor2026_seed42.json \
        --pred_dir preds_nnunet --subset test --out results/nnunet/nnunet_per_case.csv

nnU-Net writes predictions back in the input (CT) grid, so they are compared with the native
label directly. Dice, HD95 (mm), Surface Dice use src/metrics; the CSV has the same columns as the
E1 per-case table (model = "nnUNet") so scripts/run_experiments.py e5/e6 can pair it with ACF.
ECE and uncertainty columns are NaN: nnU-Net only provides them if run with --save_probabilities,
and that route is not wired here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.hecktor2026 import Hecktor2026Adapter  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.metrics.segmentation import dice, hd95, surface_dice  # noqa: E402
from src.training.train import get_git_info  # noqa: E402


def evaluate(data_dir: str, split_path: str, pred_dir: str, subset: str = "test", model: str = "nnUNet") -> pd.DataFrame:
    samples = {s["patient_id"]: s for s in Hecktor2026Adapter(data_dir).training_samples()}
    git, rows = get_git_info()["commit"], []
    for pid in load_split(split_path)[subset]:
        pred_img = nib.as_closest_canonical(nib.load(Path(pred_dir) / f"{pid}.nii.gz"))
        lab_img = nib.as_closest_canonical(nib.load(samples[pid]["label"]))
        pred, lab = np.asarray(pred_img.dataobj) > 0, np.asarray(lab_img.dataobj) > 0
        if pred.shape != lab.shape:
            raise ValueError(f"{pid}: prediction {pred.shape} is not on the native CT grid {lab.shape}")
        sp = tuple(float(abs(x)) for x in np.diag(lab_img.affine)[:3])
        h = hd95(pred, lab, sp)
        rows.append({"model": model, "patient_id": pid, "centre": samples[pid]["centre"], "tag": "nnunet",
                     "dice": dice(pred, lab), "hd95_mm": h.value, "hd95_pred_empty": h.pred_empty,
                     "surface_dice_2mm": surface_dice(pred, lab, sp, 2.0), "surface_dice_3mm": surface_dice(pred, lab, sp, 3.0),
                     "ece_all": np.nan, "ece_roi": np.nan, "unc_auroc": np.nan, "unc_overlap_dice": np.nan,
                     "label_voxels": int(lab.sum()), "git": git})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = evaluate(a.data_dir, a.split, a.pred_dir, a.subset)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} cases -> {a.out}; mean dice {df['dice'].mean():.4f} (nan-aware over {df['dice'].notna().sum()} cases)")


if __name__ == "__main__":
    main()
