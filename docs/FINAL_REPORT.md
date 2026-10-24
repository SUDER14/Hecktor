# Final report

Date: 2026-10-24. Code commit verified below: `d1094ae` (the commit adding this report follows it; the pushed hash is
shown by the `git ls-remote` proof at the end of the session and equals local HEAD).

## What was built

| Phase | Deliverable | Where |
|---|---|---|
| 0 | Audit | `docs/STATE.md` |
| 1 | HECKTOR 2026 adapter, synthetic PET/CT cohort (1x1x1 and 1x1x3 mm CT, 3-4 mm PET, fake centres), 4-d conditioning `[sx,sy,sz,flag]` with train-only stats, centre-grouped splits (seed saved, tiny-centre-only test sets rejected), world-coordinate padded CT/PET crops, dual preprocessing cache, spacing manifest | `src/data/`, `scripts/spacing_manifest.py` |
| 2 | B1 `UNet3D`, B2 `DualEncoderUNet`, ACF (level-matched fusion, per-modality FiLM hypernetworks, physical-extent checks), Tier-2 kernels (off), evidential head | `src/models/` |
| 3 | Dice, Dice+CE, evidential CE, annealed KL; Dice, HD95, Surface Dice, ECE, reliability diagram, uncertainty/error overlap | `src/training/losses.py`, `src/metrics/` |
| 4 | Config-driven trainer (AMP, accumulation, AdamW, cosine, resume, git/env logging), native-spacing sliding-window inference with NIfTI output, per-case evaluation, E1-E6, statistics, smoke and overfit scripts, B1-B5 configs | `src/training/run.py`, `src/eval/`, `scripts/`, `configs/` |
| 5 | nnU-Net v2 convert / evaluate scripts and commands | `scripts/nnunet_*.py`, `docs/nnunet_reference.md` |
| 6 | Kaggle (`.py` cells) and Colab notebooks, optional cu124 cell | `notebooks/kaggle_runner.py`, `colab_runner.ipynb` |
| 7 | Streamlit viewer, screenshot script | `app/`, `scripts/screenshot_app.py` |
| 8 | Glossary, 9 chapters, 40 exercises + answers, 15-paper reading list, 2 executed notebooks | `docs/learn/`, `notebooks/0*.ipynb` |

## VERIFIED / NOT VERIFIED

| Claim | Status | Evidence |
|---|---|---|
| Full test suite passes on CPU | VERIFIED | `python -m pytest -q` -> `121 passed, 31 warnings in 186.99s` |
| Lint clean (pyflakes-class rules) | VERIFIED | `ruff check src scripts tests app --select E9,F` -> `All checks passed!` (style rules E/W were not enforced) |
| Smoke test on CPU < 5 min | VERIFIED | `scripts/smoke_test.py`: PASS for unet3d, dual_unet, acf_softmax, acf_evidential; internal `total 55.8s`, wall 91 s |
| Train -> checkpoint -> resume continues at the next epoch | VERIFIED | smoke test asserts the epoch rows after resume; `tests/test_phase4_train.py` |
| Overfit one batch passes (CPU, tiny) | VERIFIED, wiring check only | `PASS: fg dice 0.9014 >= 0.9 at iter 9` (2 synthetic crops, 16^3). Not a result |
| Spec's hand-computed loss values | VERIFIED | Dice loss 0.1860, evidential CE 0.3333, KL 0.0236 (tests + notebook 02) |
| CT/PET crops cover the same physical box (96 mm = 96^3 CT / 24^3 PET) | VERIFIED on synthetic data | `tests/test_phase1_data.py`, `test_physical_crop.py`, `ACF.level_table` |
| FiLM is identity at init; gradient reaches both hypernetworks; zeroed-conditioning ACF matches B2 structure | VERIFIED | `tests/test_phase2_models.py` |
| Parameter counts: B1 5,730,274; B2 9,481,362; B3/B4/B5 6,229,986 | VERIFIED | instantiated from the committed configs |
| B2b (`configs/b2b_dual_matched.yaml`, widths 12/28/52/104/208) has 6,277,406 params, +0.76% vs ACF (within 3%) | VERIFIED | `tests/test_param_matched.py` (instantiated from configs; fails if drift exceeds 3%) |
| `scripts/make_synthetic_cohort.py --synthetic_production` yields 96^3 CT (1 mm) / 24^3 PET (4 mm) training items | VERIFIED (shapes only) | one item loaded through `build_dual_datasets`; no timing or VRAM measured |
| Streamlit app runs headless on synthetic data | VERIFIED | `tests/test_phase7_app.py` (Streamlit AppTest) and rendered PNGs viewed |
| E1-E6 scripts run end to end | VERIFIED on untrained smoke models only | outputs are meaningless numbers and are not reported |
| Any Dice / HD95 / ECE / speed / VRAM figure for real data | **NOT VERIFIED (none exist)** | no real data, no GPU |
| Performance on HECKTOR; the adapter against real files | NOT VERIFIED | access not granted; layout taken from the task description |
| PET is in SUV units (`suv_clip_max: 20` is a placeholder) | NOT VERIFIED | |
| 96^3 / 1 mm training memory and epoch time | NOT VERIFIED | never run at production scale or on a GPU |
| nnU-Net commands (`nnUNetv2_*` flags, `splits_final.json` format) | NOT VERIFIED | nnU-Net not installed; written from documentation knowledge |
| Playwright full-page screenshot mode | NOT VERIFIED | Playwright not installed; figure-mode screenshots were run |
| Kaggle / Colab notebooks | NOT VERIFIED | not run on either platform; JSON validity of the notebook checked |
| Evidential head improves calibration; ACF beats baselines or nnU-Net | NOT VERIFIED | untested hypotheses |

## NOT RUN

GPU training of B1-B5 and B2b at 96^3; E1-E6 on real checkpoints; nnU-Net training/prediction; Kaggle and Colab runners;
HECKTOR spacing manifest; peak-VRAM measurement (E3 writes "not measured" without CUDA).

## Exact commands to run next (Kaggle or Colab GPU)

```bash
git clone https://github.com/SUDER14/Hecktor && cd Hecktor
pip install monai nibabel SimpleITK scipy pandas scikit-image matplotlib pyyaml
python -c "import torch; print(torch.cuda.is_available())"              # must print True (else the cu124 cell)
python scripts/spacing_manifest.py --data_dir /tmp/hecktor2026_training --out results/manifest
# 2-epoch timing probe (read time_data_s vs time_gpu_s, peak_vram_mb):
python -m src.training.run --config configs/b4_acf_cond.yaml --override data_dir=/tmp/hecktor2026_training \
    --override run.dir=runs/probe --override train.n_epochs=2 --override train.val_every=1
# then each config, resumable:
for c in b1_unet_resample b2_dual_resample b2b_dual_matched b3_acf_zeroed b4_acf_cond b5_acf_evidential; do
  python -m src.training.run --config configs/$c.yaml --override data_dir=/tmp/hecktor2026_training --resume; done
python scripts/run_experiments.py e1 --runs B1=runs/b1_unet_resample B2=runs/b2_dual_resample B2b=runs/b2b_dual_matched \
    B3=runs/b3_acf_zeroed B4=runs/b4_acf_cond B5=runs/b5_acf_evidential --out results/e1 --device cuda
```
Before the first real run: confirm `split.test_centres` against the manifest (do not use only HMR/USZ), confirm PET
units, and check that no HECKTOR affine is oblique (the crop code rejects those).

## Open risks

1. **HECKTOR not yet available.** Everything above is validated on synthetic data and (earlier) Decathlon Lung.
   Real spacing statistics, PET units, orientations and centre sizes are unknown.
2. **nnU-Net may win.** It is a strong, self-configuring baseline; the thesis must be framed so that a loss to
   nnU-Net, with a gain under spacing shift or in calibration, is still a reportable finding.
3. **Prior-art search is not done.** Known and cited: HyperSpace (arXiv:2407.03681). Found this session while
   verifying the reading list: Consispace (arXiv:2606.31839, spacing-consistent resampling) and Diaz et al.
   (arXiv:2503.21829, resolution-adaptive convolutions, no resampling). A systematic search is still required
   before any novelty statement. No "first"/"novel" claims were written in code or docs.
4. **Capacity confound.** B2 has about 52% more parameters than ACF (full-depth PET encoder at 1 mm); report counts.
   B2b (6,277,406 params, +0.76% vs ACF) is the parameter-matched control, but it has never been trained, and it
   keeps B2's `pet_depth: 4` (deeper PET path than ACF), so matched total parameters do not mean matched architecture.
   `run_experiments.py` takes free-form `LABEL=run_dir` pairs (read from the source, not run with B2b).
5. **Single seed, 100-epoch plan unmeasured.** Differences below seed variance mean nothing.
6. Synthetic anisotropy experiment (E2) is block averaging, not a model of real thick-slice scanners.
7. Evidential training stability (loss weights, KL annealing length) is untested on real data.
8. `a.py` and an unrelated `.pptx` in the repo root remain untracked and were not touched.

## Pushed commit

See the `git ls-remote origin main` output at the end of the session transcript; it equals local HEAD.
