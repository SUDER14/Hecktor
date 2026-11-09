# PROJECT_CONTEXT.md

Living summary of where the project stands. `CLAUDE.md` remains the standing instruction file;
this file records design changes that came from looking at the real HECKTOR 2026 data. Every
data claim here is tagged **VERIFIED** (measured; see `docs/DATA_REPORT.md` and the cited script) or
**NOT VERIFIED**.

## Data now in hand (HECKTOR 2026 training zip, 782 cases)

- The zip (`E:\HECKTOR\HECKTOR 2026 Training Data.zip`) is read **in place** through
  `src/data/zip_reader.py` (in-memory, no extraction, never writes next to the zip). Junk
  (`__MACOSX/`, `.DS_Store`, `._*`) is skipped. VERIFIED (`tests/test_adapters.py`).
- For all 782 cases CT, PET and label share one shape, spacing and affine: **the released PET is
  already resampled onto the CT grid.** VERIFIED (`scripts/spacing_manifest.py`).
- Consequence: the true acquisition-grid PET is **not available**.

## Design change: two PET modes (both kept, both configured, both tested)

`data.pet_mode` in the config selects the preprocessing and model path.

| `pet_mode` | What the network sees | Model code | Conditioning |
|---|---|---|---|
| `shared` | CT and PET as two channels on the CT grid at each scan's native spacing; patches fixed in voxels | `src/models/acf_shared.py`, `dual_encoder_shared.py`; data `src/data/shared_cache.py`, `crop.py` | one vector `[sx, sy, sz]` (z-scored, TRAIN-only stats); `conditioning_zeroed` ablation flag; hypernetwork -> FiLM (gamma=1, beta=0 at init) on encoder blocks |
| `rederived` | CT on the CT grid; PET **re-derived** by anti-aliased down-sampling to a coarse grid (default 4x4x4 mm, configurable via `data.pet_spacing`) | original dual-resolution code: `acf.py`, `dual_encoder.py`, `physical_crop.py`, `dual_cache.py` (+ `src/data/pet_rederive.py`) | per-modality vectors using each grid's own spacing (PET: the re-derived spacing) |

Re-derived PET is an **approximation of the acquisition grid, never "native PET"**. It cannot
contain detail the released (already interpolated) PET lacks, and the blur/sampling is not the
scanner's PSF or reconstruction grid. How much detail the down/up round trip loses is measured
in `docs/DATA_REPORT.md` (section "PET round trip").

Baselines B1 (CT only) and B2 (dual encoder, no conditioning) are resampled to **one fixed
spacing chosen from the manifest median**: 0.9766 x 0.9766 x 3.27 mm (VERIFIED median over all
782 scans). Config key `baseline.resample_spacing`.

Evidential head: K configurable (`model.out_channels`); label-to-class mapping via
`data.label_map`. K = 3 (decided; see below).

## Prior art (unchanged)

Spacing-conditioned hypernetworks are HyperSpace (Joutard et al., MICCAI 2024, arXiv:2407.03681).
Nothing here is presented as new for that idea.

## Decisions taken (2026-11-09)

- **Test centres: CHUS + CHUV** (123 cases). **Validation: seeded 10% of the other cases, stratified by centre**; train 592 / val 67 /
  test 123 (`configs/splits/hecktor2026_seed42.json`, seed 42). Conditioning statistics fitted on the 592 training cases only.
- **K = 3** (background, label 1, label 2). What the two labels denote is **not verified**; code and docs call them label 1 and label 2.
- **Cache**: label-free head-and-neck box (350 mm long x 300 mm wide, CT only), stored as one `.acf` file per case in `D:\hecktor_cache`
  (CT int16, PET float16 or float32 where float16 would overflow, label uint8 raw). 43.59 GiB, 782 files, verified (`docs/DATA_REPORT.md`).
- **Normalisation** at load time, same for every case: CT clipped to [-1000, 1000] then scaled; PET divided by the 99.5th percentile of its
  positive voxels inside the CT body mask (CT > -500 HU). PET units are not documented and not assumed (SUV: NOT VERIFIED).

## Open items

- `pet_mode: rederived` is kept (old dual-resolution code and tests) but is not wired to the `.acf` cache and still has K = 2 configs.
- Nothing has been trained; no GPU has been used. A CPU timing probe exists (`scripts/timing_probe.py`); no long training until approved.
- The PET scale outliers (MDA-402..447 and five other cases) are handled only by the per-case normalisation; their cause is unknown.
- `CLAUDE.md` still describes the earlier design (native 4 mm PET encoder, physical-box crops); it was not edited in this change.
