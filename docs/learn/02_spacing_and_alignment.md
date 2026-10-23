# 02 Spacing and physical alignment

## The problem in one picture
CT: 1 mm voxels. PET: 4 mm voxels. Same patient, same anatomy, different grids. Voxel (40, 40, 40) of the CT
and voxel (40, 40, 40) of the PET are **not** the same place: CT's is at 40 mm, PET's at 160 mm. Cropping both
with the same indices (what `RandCropByPosNegLabeld` does) silently misaligns them.

## The rule we use: crop a box in millimetres
`src/data/physical_crop.py:physical_crop`:

1. Convert each affine to a `Grid` (origin, spacing). Convention: voxel i is centred at
   `origin + i*spacing`; its leading edge at `origin + (i - 0.5)*spacing`.
2. Choose patch extent `extent_mm = patch_voxels_CT * ct_spacing` (96 vox x 1 mm = 96 mm).
3. For each modality: voxels `n = round(extent / own_spacing)` (CT 96, PET 24 at 4 mm).
4. Choose a start in world mm (centre minus half extent), snap to each grid: `idx = round((start - edge0)/spacing)`.
   Each start is off by at most half a voxel of *that* grid, so PET start error is up to 2 mm.
5. Clamp the start **jointly** so every window stays inside every volume (clamping separately would shift one).
6. Near borders / small volumes (`pad=True`): centre on the overlap of the fields of view and read outside
   voxels as 0 (`extract_padded`), which keeps world positions exact.

## The safety check
`geometry.py:check_physical_extent` compares `voxels x spacing` of both crops and raises if they differ by
more than one coarser voxel per axis. The model repeats it on the batch it receives and, inside the network,
per fused level (`ACF._check_level`). Failing loudly beats training on misaligned data.

## Tested with synthetic scans
`tests/test_phase1_data.py`, `tests/test_physical_crop.py`: a bright blob at a known world position lands
within one PET voxel in both crops, even with arbitrary origins; padding keeps world positions.
The synthetic generator (`src/data/synthetic.py`) writes fake centres with 1x1x1 and 1x1x3 mm CT and 3-4 mm PET.

## Conditioning vector and leakage
`v = [sx, sy, sz, flag]`, z-scored with **train-only** statistics fitted per modality and saved to JSON; val/test
reuse them unchanged (fitting on all data would leak test information). HECKTOR spacing values themselves are
unknown until data arrives; the manifest script measures them.

## What is *not* solved
Oblique (non-axis-aligned) affines are rejected, not handled. If real HECKTOR contains them, either resample
those cases or extend the code. PET units (SUV vs raw activity) are assumed SUV and must be verified.
