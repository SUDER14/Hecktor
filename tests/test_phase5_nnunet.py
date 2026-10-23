import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.hecktor2026 import Hecktor2026Adapter
from src.data.splits import group_split_by_centre, save_split
from src.data.synthetic import SynthConfig, write_synthetic_hecktor2026


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_convert_puts_both_channels_on_ct_grid_and_keeps_test_out_of_training(tmp_path):
    root = tmp_path / "d"
    write_synthetic_hecktor2026(root, SynthConfig(fov_mm=(32, 32, 32), n_per_centre=2, seed=3))
    samples = Hecktor2026Adapter(root).training_samples()
    split = group_split_by_centre(samples, ["MDA"], 0.25, seed=1)
    save_split(split, tmp_path / "split.json", meta={"seed": 1})
    conv = _load("nnunet_convert")
    out = conv.convert(str(root), str(tmp_path / "split.json"), str(tmp_path / "raw"), 501, "T")
    assert out.name == "Dataset501_T"
    ds = json.loads((out / "dataset.json").read_text())
    assert ds["channel_names"] == {"0": "CT", "1": "PT"} and ds["numTraining"] == len(split["train"]) + len(split["val"])
    pid = split["train"][0]
    ct, pt = (nib.load(out / "imagesTr" / f"{pid}_000{i}.nii.gz") for i in (0, 1))
    assert ct.shape == pt.shape and np.allclose(ct.affine, pt.affine)  # nnU-Net requires identical geometry
    tr_ids = {p.name.split("_0000")[0] for p in (out / "imagesTr").glob("*_0000.nii.gz")}
    assert not (tr_ids & set(split["test"])) and {p.name.split("_0000")[0] for p in (out / "imagesTs").glob("*_0000.nii.gz")} == set(split["test"])
    folds = json.loads((out / "splits_final.json").read_text())
    assert folds[0]["val"] == split["val"] and not set(folds[0]["train"]) & set(split["test"])


def test_evaluate_uses_our_metrics_and_flags_wrong_grid(tmp_path):
    root = tmp_path / "d"
    write_synthetic_hecktor2026(root, SynthConfig(fov_mm=(32, 32, 32), n_per_centre=2, seed=3))
    samples = Hecktor2026Adapter(root).training_samples()
    split = group_split_by_centre(samples, ["MDA"], 0.25, seed=1)
    save_split(split, tmp_path / "split.json")
    pred = tmp_path / "pred"
    pred.mkdir()
    for pid in split["test"]:  # a "perfect" prediction = the label itself
        lab = nib.as_closest_canonical(nib.load(next(s["label"] for s in samples if s["patient_id"] == pid)))
        nib.save(nib.Nifti1Image((np.asarray(lab.dataobj) > 0).astype(np.uint8), lab.affine), pred / f"{pid}.nii.gz")
    ev = _load("nnunet_evaluate")
    df = ev.evaluate(str(root), str(tmp_path / "split.json"), str(pred))
    d = df["dice"].dropna()
    assert len(df) == len(split["test"]) and (d == 1.0).all() and (df["model"] == "nnUNet").all()
