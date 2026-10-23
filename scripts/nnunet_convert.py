"""Convert our split of the HECKTOR 2026 layout to nnU-Net v2 raw format.

    python scripts/nnunet_convert.py --data_dir data/hecktor2026_training --split configs/splits/hecktor2026_seed42.json \
        --raw_dir $nnUNet_raw --dataset_id 501 --name HECKTOR2026

Writes   $nnUNet_raw/Dataset501_HECKTOR2026/{imagesTr,labelsTr,imagesTs,labelsTs,dataset.json}
and      splits_final.json (our train / val as fold 0) to copy into $nnUNet_preprocessed/Dataset501_*/ .

IMPORTANT, and different from our own pipeline: nnU-Net requires every channel of a case to share
one grid (same shape, spacing, origin). CT and PET do not, so PET is resampled (trilinear) onto the
CT grid here, ONCE, as a necessary pre-step. nnU-Net then resamples everything again to its own
median target spacing during preprocessing. That is a different treatment of the data from ACF's
native-spacing approach and must be stated wherever nnU-Net numbers are reported.

Held-out test patients go to imagesTs / labelsTs only (never into training or the splits file).
Channels: _0000 = CT, _0001 = PT. Labels: 0 background, 1 GTV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from src.data.hecktor2026 import Hecktor2026Adapter  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.eval.infer import resample_to_grid  # noqa: E402


def convert_case(sample: dict, img_dir: Path, lab_dir: Path | None) -> None:
    paths = dict(zip(sample["modalities"], sample["image"]))
    ct = nib.as_closest_canonical(nib.load(paths["CT"]))
    pet = nib.as_closest_canonical(nib.load(paths["PT"]))
    pet_on_ct = resample_to_grid(np.asarray(pet.dataobj, np.float32), pet.affine, ct.affine, ct.shape[:3], order=1)
    pid = sample["patient_id"]
    img_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(ct.dataobj, np.float32), ct.affine), img_dir / f"{pid}_0000.nii.gz")
    nib.save(nib.Nifti1Image(pet_on_ct.astype(np.float32), ct.affine), img_dir / f"{pid}_0001.nii.gz")
    if lab_dir is not None:
        lab = nib.as_closest_canonical(nib.load(sample["label"]))
        lab_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image((np.asarray(lab.dataobj) > 0).astype(np.uint8), ct.affine), lab_dir / f"{pid}.nii.gz")


def convert(data_dir: str, split_path: str, raw_dir: str, dataset_id: int = 501, name: str = "HECKTOR2026") -> Path:
    samples = {s["patient_id"]: s for s in Hecktor2026Adapter(data_dir).training_samples()}
    split = load_split(split_path)
    out = Path(raw_dir) / f"Dataset{dataset_id:03d}_{name}"
    train_ids = sorted(split["train"] + split["val"])
    for pid in train_ids:
        convert_case(samples[pid], out / "imagesTr", out / "labelsTr")
    for pid in split["test"]:
        convert_case(samples[pid], out / "imagesTs", out / "labelsTs")
    (out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT", "1": "PT"},
        "labels": {"background": 0, "GTV": 1},
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz",
    }, indent=2))
    (out / "splits_final.json").write_text(json.dumps(
        [{"train": sorted(split["train"]), "val": sorted(split["val"])}], indent=2))
    print(f"{len(train_ids)} train+val, {len(split['test'])} test cases -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--dataset_id", type=int, default=501)
    ap.add_argument("--name", default="HECKTOR2026")
    a = ap.parse_args()
    convert(a.data_dir, a.split, a.raw_dir, a.dataset_id, a.name)


if __name__ == "__main__":
    main()
