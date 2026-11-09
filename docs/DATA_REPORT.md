# HECKTOR 2026 data report

Every statement is tagged **VERIFIED** (measured by a script in this repo, named in the text) or **NOT VERIFIED**.
Nothing here is a model result: nothing has been trained, and no GPU has been used.
Tables are produced by `scripts/make_data_report_tables.py` from the CSV/JSON outputs in `results/` (gitignored).

## 1. Source and reader

- Source: `E:\HECKTOR\HECKTOR 2026 Training Data.zip` (144 GB), 782 cases under `HECKTOR 2026 Training Data/<CENTRE-NNN>/`
  with `<id>__CT.nii.gz`, `<id>__PT.nii.gz`, `<id>.nii.gz` (label) and `HECKTOR_2026_training_data.csv`. VERIFIED.
- The zip contains macOS junk (`__MACOSX/`, `.DS_Store`, `._*`: 3,140 entries); it is skipped. VERIFIED.
- Cases are read in place by `src/data/zip_reader.py` (member bytes decompressed in memory, nibabel in-memory parsing; headers by
  reading the first ~1 kB of the stream). Nothing is extracted, nothing is written next to the zip. VERIFIED
  (`tests/test_adapters.py`: zip reads equal disk reads voxel for voxel on a synthetic zip with the same junk entries).
- Centre = the id prefix. Centres and case counts: CHUM 56, CHUP 75, CHUS 72, CHUV 51, HGJ 55, HMR 18, MDA 444, USZ 11. VERIFIED.

## 2. Geometry (headers of all 782 cases; `scripts/spacing_manifest.py`)

- **CT, PET and label share one shape, spacing and affine in all 782 cases**: the released PET is already resampled onto the CT
  grid, so the acquisition-grid PET is not available. VERIFIED.
- Orientation: LPS in 781 cases, RAS in 1 (MDA-075). No oblique affines (tolerance 1e-3). VERIFIED.
- In-plane spacing is isotropic in every case (sx = sy); in-plane matrix 512 x 512 except some MDA cases at 128 x 128. VERIFIED.

| centre | n | in-plane mm (min / median / max) | z mm (min / median / max) | nz (min-max) | distinct (sx,sy,sz) |
|---|---|---|---|---|---|
| CHUM | 56 | 0.977 / 0.977 / 1.367 | 1.50 / 1.50 / 3.31 | 90-348 | 5 |
| CHUP | 75 | 0.977 / 0.977 / 0.977 | 1.50 / 1.50 / 2.00 | 485-736 | 2 |
| CHUS | 72 | 0.684 / 1.172 / 1.172 | 2.00 / 3.00 / 5.00 | 109-218 | 4 |
| CHUV | 51 | 0.977 / 1.367 / 1.367 | 1.00 / 3.27 / 4.25 | 67-392 | 5 |
| HGJ | 55 | 0.977 / 0.977 / 0.977 | 3.27 / 3.27 / 3.27 | 91-135 | 1 |
| HMR | 18 | 0.977 / 0.977 / 1.367 | 3.27 / 3.27 / 3.27 | 83-267 | 2 |
| MDA | 444 | 0.488 / 0.977 / 2.734 | 1.25 / 3.27 / 5.00 | 87-769 | 23 |
| USZ | 11 | 0.977 / 0.977 / 1.367 | 3.27 / 3.27 / 3.27 | 299-351 | 3 |

Median over all scans: 0.9766 x 0.9766 x 3.27 mm. This is the single spacing the resampled baselines (B1, B2) use
(`baseline.resample_spacing`). VERIFIED. Whole-body-length scans exist (CHUP up to 736 slices at 1.5 mm); largest volume
about 2.0e8 voxels.

## 3. Intensities

- PET dtype float32, CT int16 / int32 (float32 in 4 cases), label uint8, in every case read. VERIFIED.
- **PET scale is not uniform.** Over all 782 cases (`scripts/crop_scan.py`): the median whole-volume PET maximum is 19.1, but
  **52 cases have a PET maximum above 1,000**: MDA-402 to MDA-447 (46 cases, contiguous; MDA-400 and MDA-401 are not affected),
  plus MDA-052, MDA-300, MDA-310, CHUP-014, CHUP-031, CHUP-045. VERIFIED.
  - In the MDA-402..447 block the in-label PET maximum is 14,081 to 78,525 (median 33,515); everywhere else the median in-label
    maximum is 13.9. VERIFIED.
  - Of the six non-block cases: MDA-052 has one hot spot outside the head-and-neck crop (crop 99.5th percentile 8.5, label mean 4.5:
    normal scale); MDA-300, MDA-310, CHUP-014, CHUP-031, CHUP-045 are on a whole-case scale of about 1,000 to 3,000
    (99.5th percentiles 1,293 / 3,052 / 1,145 / 1,072 / 1,294). VERIFIED (`scripts/verify_cache.py`).
- The headers carry no units or description (`descrip` empty, `scl_slope` 1, `scl_inter` 0). **Whether any PET is SUV is NOT VERIFIED.**
  In the 64-case sample (8 per centre, seeded) outside the outliers the PET looks SUV-like (tumour mean about 3 to 14, whole-volume
  mean 0.1 to 0.3) but that is consistent with SUV, not proof. `suv_clip_max: 20` is therefore not used anywhere in the new path.
- CT ranges (64-case sample): minimum -1024 in some centres and -3024 in others; HGJ has cases from -7029 to 23,152; MDA has -2048.
  Stored CT is clipped to [-1000, 1000] before use, so these ranges do not reach the network.
- PET minimum is 0 in the sample; two cases in the full scan have a negative PET minimum. No NaN in the sample. VERIFIED (partly sample).

## 4. Labels and the CSV

- Raw label values are exactly {0, 1, 2} in all 782 cases (no other value); 664 cases have both 1 and 2, 87 only label 1, 31 only
  label 2, none empty. Total voxels: label 1 3,801,916; label 2 4,815,991. VERIFIED (`scripts/data_stats.py --label_only`).
- CSV columns: PatientID, CenterID, Age, Gender, Tobacco Consumption, Alcohol Consumption, Performance Status, Treatment, HPV Status,
  Relapse, RFS, T-stage, N-stage. **No column says what a label value denotes**; the CSV is silent on it. VERIFIED.
- Observed relations, all 782 cases: label 2 is present in every case with N-stage N1, N2 or N3 (695 of 695) and in none of the
  87 N0 cases; median label-2 volume grows with N-stage (1.7 / 12.1 / 37.2 mL for N1 / N2 / N3) and median label-1 volume with
  T-stage (3.8 / 6.7 / 13.8 / 25.7 mL for T1 / T2 / T3 / T4). 31 cases have label 2 and no label 1 (22 of them T1). VERIFIED as
  correlations. **What label 1 and label 2 denote is an unverified assumption** and the code and docs call them label 1 and label 2.

## 5. PET round trip (re-derived PET, `scripts/pet_roundtrip.py`)

`pet_mode: rederived` anti-aliases the CT-grid PET (Gaussian, sigma = (ratio - 1) / 2 CT voxels) and samples it linearly on a
coarse grid (default 4 x 4 x 4 mm; an axis already coarser keeps its spacing). **It is an approximation of the acquisition grid, never
native PET.** Down then back up to the CT grid, against the released PET, 20 stratified seeded cases (MDA 11, CHUP 2, CHUS 2, others
1 each): Pearson r 0.9977 mean (min 0.9936); inside the label r 0.988 mean (min 0.942); the maximum inside the label falls by 14.0%
on average (range 5.4% to 27.4%; 3 of 20 cases lose more than 20%); largest single-voxel error relative to the PET maximum 21%
mean (up to 37%). Absolute RMSE and max-error columns are dominated by MDA-428 (PET maximum 40,959) and are not meaningful averaged.
VERIFIED on these 20 cases only.

## 6. Crop rule (label-free) and its verification

The full-field-of-view cache would be 209 GB (shared native, projected from 20 cases), over the 80 GB limit, so a fixed-size
head-and-neck box is stored (`src/data/crop_rule.py`). **Rule, CT only, same for every case and split**: body = CT > -300 HU; a slice
counts if its body area is at least 1,500 mm2; top of head = the most superior such slice; the box is 350 mm going down from it in z
and a 300 mm square in x and y centred on the body centroid of that slab; clamped to the volume by shifting. Labels are used only to verify.

- Verification over all 782 cases (`scripts/crop_scan.py`, `scripts/crop_verify.py`): **no label voxel falls outside the box**
  (0 failures). A 300 mm length failed for 3 MDA cases (worst margin -17 mm), so the length is 350 mm. Smallest margin 0.0 mm for
  MDA-146 and MDA-210: the label touches the scan's own bottom slice, so the box face is the volume edge. 1st-percentile margin 13.1 mm.
  VERIFIED.
- Caveat: the two numbers (350, 300) were chosen after seeing all 782 labels, including the test centres. The rule itself never
  reads a label.
- Crop sizes: in-plane 110-512 voxels (median 307) = 250-300.7 mm; z 67-350 voxels (median 107) = 210-352 mm; 149 scans are shorter
  than 350 mm and keep everything; the crop keeps 20.8% of the voxels. VERIFIED.

## 7. Cache (`D:\hecktor_cache`, outside the repo)

- One memory-mappable `.acf` file per case (`src/data/crop_cache.py`): CT int16, PET float16, label uint8 with raw values 0/1/2, header
  with box, affine, spacing, rule signature. **PET is float32 in the 12 cases whose in-crop PET maximum exceeds 60,000** (MDA-406,
  410, 415, 418, 420, 421, 425, 434, 439, 441, 442, 444) because float16 overflows to infinity at 65,504; nothing is clipped or saturated.
  The first build stored these as float16 (infinite voxels); they were rebuilt and the other 770 files were left untouched.
- Verified (`scripts/verify_cache.py`): 782 files (none missing or extra, no temp files, all with the expected rule signature);
  **no infinite or NaN PET value in any case**; no CT value clipped at int16; label values {0, 1, 2}; a 5-case spot check against the
  zip (CHUM-001, CHUS-046, HGJ-043, MDA-439, USZ-003) matched exactly on shape, affine, CT, PET and label (47,252,292 voxels per
  modality). Total size 43.59 GiB (20-case projection was 41.5 GB); D: had 72.5 GB free afterwards.
- Build: 4.5 s per case measured on 20 cases; the full build log is `D:\hecktor_cache\build.log`.

## 8. Split, conditioning, classes, normalisation (decisions)

- **Test = CHUS + CHUV** (123 cases, whole centres). **Validation = a seeded random 10% of the other cases, stratified by centre**
  (`stratified_val_split`, seed 42): train 592 (CHUM 50, CHUP 67, HGJ 49, HMR 16, MDA 400, USZ 10), validation 67 (6, 8, 6, 2, 44, 1),
  test 123 (CHUS 72, CHUV 51). Committed as `configs/splits/hecktor2026_seed42.json`. Validation shares centres with training by
  design; only the test centres measure generalisation to unseen centres.
- Conditioning vector [sx, sy, sz], z-scored with statistics fitted on the 592 training cases only
  (`configs/splits/hecktor2026_seed42_conditioning_stats.json`: in-plane mean 0.987 / std 0.133 mm, slice thickness mean 2.910 /
  std 0.727 mm). VERIFIED.
- Spacing of test versus the training centres (training centres CHUM, CHUP, HGJ, HMR, MDA, USZ = 659 cases): in-plane test min / median /
  max 0.684 / 1.172 / 1.367 mm vs train 0.488 / 0.977 / 2.734; z test 1.00 / 3.00 / 5.00 vs train 1.25 / 3.27 / 5.00. **The test set is
  inside the training range except in z for 5 CHUV cases (z = 1.00 mm, below the training minimum of 1.25 mm).** In-plane no test case
  is outside the range, but the single 0.684 mm case has no training case within 0.15 mm, 71 test cases are at 1.172 mm (4 training
  cases there) and 77 of 123 test cases have an (in-plane, z) pair never seen in training. VERIFIED.
- **K = 3**: background, label 1, label 2 (`data.label_map: [0, 1, 2]`, `model.out_channels: 3`). What the labels denote is not
  verified (section 4). Headline Dice is on the merged foreground; per-class Dice for 1 and 2 is also reported.
- **Normalisation (identical for every case, nobody excluded)**: CT clipped to [-1000, 1000] HU and scaled to [0, 1]; PET divided by
  the 99.5th percentile of its positive voxels inside the CT body mask (CT > -500 HU), computed per case at load time
  (`src/data/acf_dataset.py`; no upper clip unless configured). Example, measured on the cache: MDA-410 (block case; scale 51,652,
  float32) has tumour mean 0.254 and maximum 1.77 after normalisation; CHUM-001 (scale 10.5) has 0.406 and 1.78.

## 9. Modes and what is wired

- `pet_mode: shared` (CT + PET as two channels on the CT grid, one conditioning vector): wired to the `.acf` cache, native and
  resampled (B1/B2 at 0.9766 x 0.9766 x 3.27 mm) modes, configs `configs/shared_*.yaml`. Models: `acf_shared.py`,
  `dual_encoder_shared.py`; B1 is the CT-only `unet3d.py`.
- `pet_mode: rederived` (coarse PET re-derived from the CT-grid PET): the original dual-resolution code and tests are kept
  (`acf.py`, `dual_encoder.py`, `physical_crop.py`, `dual_cache.py`, `configs/b*.yaml`, `src/data/pet_rederive.py`). **It is not wired to the `.acf`
  cache** and still has K = 2 configs; deriving the 4 mm PET from the cropped arrays on demand is not implemented.
- Spacing-conditioned hypernetworks for segmentation are HyperSpace (Joutard et al., MICCAI 2024, arXiv:2407.03681); this project builds
  on that and claims nothing new for the idea. With one shared grid the CT and PET hypernetworks receive the same vector and differ only
  in their weights.

## 10. PET-scale sidecar, resampled B1/B2 cache, loading speed (local measurements)

- **PET-scale sidecar** (`scripts/precompute_pet_scales.py`, `src/data/acf_dataset.py`): the per-case 99.5th-percentile scale is computed once
  for all 782 cases and stored in `D:\hecktor_cache\pet_scale.json` (409 s). Resumable; an entry is treated as stale if the case file size
  changed; a different rule invalidates the whole file. Scale min / median / max 0.77 / 10.06 / 58,476; 731 cases below 50, 51 at or
  above 50 (the high-scale cases found in section 3). VERIFIED.
- **Resampled B1/B2 cache, measured on 20 stratified seeded cases only (not built)** (`scripts/build_resampled_cache.py --measure 20`):
  target spacing 0.9766 x 0.9766 x 3.27 mm; 46.2 MiB and 6.4 s per case; projected full cache 35.3 GB, 1.39 h; D: free 72.5 GB
  now, 36.3 GB projected after a full build. Gates (80 GB; 20 GB free) not tripped. The probe was deleted. The datasets do not read this
  cache yet: B1/B2 still resample on the fly.
- **Loading speed**, seconds per fetch (one patient = 2 patches = one micro-batch of 2), 24 timed fetches per setting, real `.acf` cache,
  peak RAM 2.2 GB (guard 6.5 GB): native path (B3-B5) 0.80 / 0.304 / 0.213 s with 0 / 2 / 3 workers; on-the-fly resampled path (B1/B2)
  1.99 / 0.742 / 0.568 s. One epoch is 592 fetches, so the loader alone needs 7.9 / 3.0 / 2.1 min per epoch (native) and
  19.6 / 7.3 / 5.6 min (resampled). VERIFIED on this laptop; other machines differ.
- **GPU**: none on this machine (no `nvidia-smi`, CPU-only PyTorch, Intel HD Graphics 620 only). Peak VRAM and GPU seconds per step are
  NOT MEASURED. `scripts/colab_timing.py` times B1 and B4 (fp32 and mixed precision) on synthetic tensors on a CUDA runtime.

## 11. NOT VERIFIED / open

- PET units (SUV or not) and the cause of the MDA-402..447 scale; what labels 1 and 2 denote.
- The normalisation's effect on any model (no model has been trained); the 99.5th-percentile rule was computed on the cache, not tested for
  stability across scanners.
- Any GPU training time, VRAM or accuracy figure (no GPU has been used here).
- `CLAUDE.md` still describes the earlier design (native 4 mm PET encoder, physical-box crops) and was not edited.
