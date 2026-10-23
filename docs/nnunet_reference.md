# nnU-Net v2 reference

nnU-Net (Isensee et al., arXiv:1809.10486; "Revisited", arXiv:2404.09556) is the strong, self-configuring
baseline. **It may win.** Results are not claimed here; this file only gives commands.

## What is different from our pipeline (state this wherever nnU-Net numbers appear)

* nnU-Net needs all channels of a case on **one** grid. PET (about 4 mm) is therefore resampled onto the CT
  grid by `scripts/nnunet_convert.py` (trilinear), then nnU-Net resamples again to **its own median
  target spacing**. ACF never resamples its inputs. Both approaches resample *something*; they do not
  resample the same thing.
* nnU-Net does its own intensity normalisation and patch/batch planning.
* Same patients: the same train/val/test lists (`configs/splits/hecktor2026_seed42.json`) are used. Test
  patients are written to `imagesTs` and never enter nnU-Net's training or its splits file.
* Predictions land on the native CT grid and are scored with **our** metric code
  (`scripts/nnunet_evaluate.py`), so Dice / HD95 / Surface Dice are computed identically for every model.

## Commands (NOT RUN in this repo's development environment: nnU-Net is not installed and there is no GPU/HECKTOR data)

```bash
pip install nnunetv2
export nnUNet_raw=/tmp/nnUNet_raw nnUNet_preprocessed=/tmp/nnUNet_preprocessed nnUNet_results=/tmp/nnUNet_results

# 1. convert (needs the split json produced by the first training run / dataset builder)
python scripts/nnunet_convert.py --data_dir data/hecktor2026_training \
    --split configs/splits/hecktor2026_seed42.json --raw_dir $nnUNet_raw --dataset_id 501

# 2. plan + preprocess
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -c 3d_fullres

# 3. use OUR split as fold 0
cp $nnUNet_raw/Dataset501_HECKTOR2026/splits_final.json $nnUNet_preprocessed/Dataset501_HECKTOR2026/splits_final.json

# 4. train fold 0 (GPU)
nnUNetv2_train 501 3d_fullres 0 --npz

# 5. predict the held-out centres
nnUNetv2_predict -i $nnUNet_raw/Dataset501_HECKTOR2026/imagesTs -o preds_nnunet \
    -d 501 -c 3d_fullres -f 0

# 6. evaluate with our metrics, then compare with ACF in E5/E6
python scripts/nnunet_evaluate.py --data_dir data/hecktor2026_training \
    --split configs/splits/hecktor2026_seed42.json --pred_dir preds_nnunet --subset test \
    --out results/nnunet/nnunet_per_case.csv
```

To include it in the paired statistics, concatenate `results/nnunet/nnunet_per_case.csv` with the E1
per-case CSV (both have `model, patient_id, centre, dice, hd95_mm, ...`) and run
`scripts/run_experiments.py e5|e6 --per-case <merged.csv>`.

`nnUNetv2_predict` flag names and the `splits_final.json` format follow the nnU-Net v2 documentation as
understood when this was written; confirm with `nnUNetv2_predict -h` and the installed version's docs
before the first real run. ECE for nnU-Net would need `--save_probabilities` (not wired).
