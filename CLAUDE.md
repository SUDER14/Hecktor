# CLAUDE.md

Project memory for Claude Code. Read this before doing anything in this repo.

---

## What this is

M.Tech dissertation (VIT Vellore, CSE AI&ML). 14 weeks, 10 credits.

**Acquisition-Conditioned Fusion (ACF)** — a dual-encoder 3D segmentation network for
head-and-neck tumour delineation on PET/CT, where a small hypernetwork reads each
scan's voxel spacing and generates FiLM modulation parameters for the encoder blocks
at inference time. Volumes are processed at native resolution; there is no resampling
stage. The decoder outputs Dirichlet concentration parameters, giving a segmentation
mask and a calibrated per-voxel uncertainty map in one forward pass.

Target dataset is **HECKTOR** (PET/CT, multi-centre, GTV masks). **Access has not been
granted yet.** Until it is, everything is built and validated against **Medical
Segmentation Decathlon**, which is freely downloadable and has real spacing variation.

---

## Current status

- [x] Proposal and Form B written
- [x] `spacing_analysis.py` written
- [x] Decathlon Task06_Lung downloaded locally (`data/`, gitignored; 63 training volumes)
- [x] Spacing analysis run on real data (outputs in `results/spacing_task06_lung/`)
- [x] Data loader (`src/data/`; split + conditioning stats committed in `configs/splits/`)
- [ ] Baseline single-modality 3D U-Net — code, train loop, per-epoch checkpointing
      and resume are written; **not yet validated**: the overfit-one-batch test has
      not passed (last seen plateauing on CPU at 2 mm), and nothing has run on a
      GPU. No Colab run, no measured epoch time, no validation Dice yet.
- [ ] Repo has one local commit (splits only), no remote — Colab cannot clone it yet
- [ ] Dual-encoder baseline
- [ ] Hypernetwork + FiLM
- [ ] Evidential head + calibration
- [ ] HECKTOR access

---

## Prior art constraint — important

**HyperSpace** (Joutard, Pietsch & Prevost, ImFusion GmbH, MICCAI 2024,
arXiv:2407.03681, code at github.com/ImFusionGmbH/HyperSpace) already conditions a
segmentation U-Net on voxel spacing via a hypernetwork and processes at native
resolution.

Do not describe spacing-conditioned hypernetworks as novel in code comments, docstrings
or documentation. Cite HyperSpace as the foundation. What is genuinely open here:

1. Multi-modal — HyperSpace is single-modality; independent conditioning per modality
   (handling the 4-5x PET/CT resolution asymmetry) is not something they did
2. Uncertainty — HyperSpace has no uncertainty or calibration component
3. Their reviewers noted the efficiency claims were never actually measured; this
   project measures peak VRAM and preprocessing wall-clock explicitly

---

## Settled decisions — do not revisit

| Question | Answer |
|---|---|
| CT-MRI fusion? | No. Verified: HEAD-NECK-RADIOMICS-HN1 and Head-Neck-CT-Atlas contain no MR series. Project is PET/CT only. |
| Dilation proportional to spacing? | No. Dilation is integer-only; spacing ratios are continuous. Dilated kernels also sample a sparse lattice (gridding). Use FiLM modulation. |
| Uncertainty to CTV margin? | No. CTV is a biological margin (ICRU 50/62/83). Delineation uncertainty belongs to PTV. Any margin work is a decision-support overlay only. |
| Full weight generation? | Tier 2 stretch goal only. FiLM first — it trains stably on limited compute. |

---

## Environment

- **Training runs on Google Colab Pro**, not locally. Local machine is for code, git,
  figures and writing.
- Code must be importable from a Colab notebook — keep `src/` clean and dependency-light.
- **Per-epoch checkpointing to Google Drive is mandatory.** Colab sessions die. A
  training loop without resumable checkpoints is not finished.
- Patch-based training: 128x128x64 or 96^3. Batch 2 with gradient accumulation to an
  effective 8. Mixed precision via `torch.amp`.

**Stack:** python >=3.10, torch >=2.1, monai >=1.3, nibabel, SimpleITK, numpy, scipy,
pandas, scikit-image, matplotlib, wandb.

Prefer MONAI's transforms and network blocks over hand-rolled equivalents. Do not
reimplement what MONAI already provides.

---

## Layout

```
src/
  metadata/     conditioning vector extraction, spacing analysis
  data/         datasets, transforms, loaders
  models/       acf.py, baselines.py, hypernet.py, evidential.py
  training/     train loop, checkpointing, scheduler
  eval/         metrics, calibration, figures
configs/        yaml experiment configs
notebooks/      Colab notebooks
results/        figures, tables, logs (gitignored except figures)
data/           gitignored entirely
docs/           proposal, Form B, thesis drafts
```

---

## Conventions

- Type hints on public functions. Docstrings explain *why*, not *what*.
- Every experiment driven by a config file, never hardcoded constants. Reproducibility
  matters more than convenience — there will be 15-25 training runs to compare.
- Seed everything. Log the seed.
- Normalisation statistics are fitted on train only and saved to disk. Reusing them
  unchanged at val/test is a correctness requirement, not a style preference.
- Write code that works on Decathlon *and* HECKTOR. Dataset-specific logic goes behind
  a small adapter, not scattered through the pipeline.

---

## Working agreement

This project has already been derailed twice by confident, unverified claims — a
proposal built on a dataset that lacked the required modality, and novelty claims
anticipated by a paper nobody had searched for.

So:

- **Do not invent experimental results, metrics, or benchmark numbers.** Ever. If a
  number is needed and not measured, leave it blank and say so.
- If unsure whether an API behaves as assumed, check the docs or write a test — do not
  assert it confidently.
- Flag problems rather than working around them silently.
- If a task seems bigger than one session, say so and propose a split instead of
  producing something half-finished.
