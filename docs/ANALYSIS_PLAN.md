# Analysis plan (written before any test result exists)

Branch `fable/first-run`, 2026-11-16. No model has been trained on a GPU and no model output has been
computed on a test case when this file was committed. `scripts/evaluate_test.py` (Phase 7) refuses to run
unless this file is committed and its sha256 matches the value recorded in the test manifest. Any change
after the first test run must be listed in section 10 with the date and the reason.

Status words used below: PLANNED = decided here, code not written yet; CODED = code and a unit test exist;
nothing in this file is a result.

## 1. Data and models

- Split: `configs/splits/hecktor2026_seed42.json`, 592 train / 67 validation / 123 test. The test set is the
  two held-out centres, CHUS (72) and CHUV (51). Training and validation never see these centres.
- Labels: K = 3 (0 background, 1, 2). What labels 1 and 2 denote clinically has not been verified from the
  dataset documentation; the plan refers to them by number.
- Input: shared-PET path (CT and PET as two channels on one grid, `.acf` cache).
- Models: B1 U-Net, CT only, resampled grid. B2 dual encoder, resampled grid. B2b dual encoder matched in
  parameters to ACF, resampled grid. B3 ACF with the conditioning input zeroed, native grid. B4 ACF
  conditioned on voxel spacing, native grid. B5 = B4 with an evidential (Dirichlet) head.
  Spacing-conditioned hypernetworks are prior art (HyperSpace, Joutard et al., MICCAI 2024); B4 against B3
  tests whether the conditioning helps in this PET/CT setting, not whether the idea is new.
- nnU-Net: preparation is deferred (Phase 8). If a nnU-Net fold-0 model exists later it is reported
  descriptively through the same metric code; it is a single model, which is a stated limitation.
- B5 is included in the comparisons only if it passes the Phase 4 per-label overfit gate. On 2026-11-16 it
  fails that gate (label 1 not learned in the time budget; `results/fable_run/B5_note.txt`).

## 2. Scoring grid

Every model is scored on the native grid of the cropped case against the native label. B1, B2 and B2b
predict on the resampled grid (0.9766 x 0.9766 x 3.27 mm, the median spacing of the 592 training cases);
their probability maps are resampled back with linear interpolation and the class is the argmax after that.
Distances use the native voxel spacing in mm.

Known ceiling for the resampled path (measured on three training cases, `tests/test_resample_on_the_fly_real.py`):
a perfect prediction on the resampled grid scores foreground Dice 0.9025 (z = 1.25 mm), 1.0000 (z = 3.27 mm)
and 0.9883 (z = 5.0 mm) after mapping back. This loss is part of what the resampled baselines pay; it is
reported next to the results, not corrected for.

## 3. Per-case metrics (test set, native grid)

For each test case and each of label 1 and label 2:

- Dice, when the label is present in the truth. When the label is absent in the truth, Dice is not defined;
  the predicted volume of that label in mL is recorded as the false-positive volume instead.
- HD95 in mm, when the label is present in both truth and prediction. A case with truth present and an empty
  prediction is counted and reported as "prediction empty"; no distance is imputed.
- Surface Dice at 2 mm and at 3 mm tolerance, when the label is present in the truth.

Per-case mean Dice = mean of the Dice values of the labels present in the truth of that case (one or two
values). A case with neither label in the truth has no per-case mean Dice; it is excluded from the paired
tests and counted in the table.

Aggregated Dice per label, as in the HECKTOR protocol: 2 * sum over cases of |P and L| divided by the sum
over cases of (|P| + |L|), computed once over the 123 test cases, with a 95% bootstrap interval over cases
(10,000 resamples). Aggregated Dice is descriptive; the tests use per-case values.

## 4. Comparisons and test

Metric for all tests: per-case mean Dice.

| # | Comparison | Role |
|---|---|---|
| 1 | B4 against B3 | primary: does spacing conditioning help, all else equal |
| 2 | B4 against B2b | secondary: native conditioned against resampled, matched parameters |
| 3 | B4 against B2 | secondary |
| 4 | B5 against B4 | secondary: cost or gain of the evidential head in Dice |
| 5 | B2 against B1 | secondary: value of PET on the resampled path |
| - | nnU-Net against B4 | descriptive only, no test |

- Test: paired Wilcoxon signed-rank by case, two-sided, alpha 0.05.
- Multiplicity: Holm step-down over the five comparisons above (m = 5; a comparison that cannot be run stays
  in the count). The proposal said Bonferroni. Holm controls the same family-wise error rate under the same
  assumptions and its adjusted p-values are never larger than Bonferroni's, so it is valid and never less
  powerful. The primary comparison is inside the Holm family; this is the conservative choice.
- Effect: median paired difference with a 95% percentile bootstrap interval over cases, 10,000 resamples,
  fixed seed. Mean difference, matched-pairs rank-biserial correlation and Cohen's d_z are reported as
  secondary columns.
- Seeds: if a model has several seeds, its metric is averaged per case over seeds first; the test then has
  one value per case per model. The number of seeds is not decided here (GPU budget).
- Per-label Dice, HD95 and surface Dice are reported for the same pairs with median differences and
  intervals, without p-values.

CODED: `src/eval/stats.py` (`holm_adjust`, `paired_comparison`, `compare_models`), tests in
`tests/test_phase4_eval.py`.

## 5. Subgroups (descriptive only, no tests)

- Unseen spacing: the 77 test cases whose (in-plane, z) spacing pair does not occur in any training case,
  against the other 46. Counted on 2026-11-16 from cache headers only (spacing metadata; no image, label or
  prediction was read): 77 = CHUS 72 + CHUV 5. The count is the same when validation cases are added to the
  reference set and for rounding at 2, 3 or 4 decimals.
- z-spacing bins: z <= 2.0 mm (23 cases), 2.0 < z <= 3.0 (56), 3.0 < z <= 3.5 (38), z > 3.5 (6).
- Centre: CHUS (72) against CHUV (51).

These subgroups overlap heavily (all CHUS cases are in the unseen-spacing group), so a difference in one
cannot be attributed to spacing rather than to centre. This is stated wherever a subgroup table is shown.

## 6. Robustness checks

- Thickness sweep: for targets 3, 5 and 7 mm, each test case is re-sliced through-plane by block-averaging an
  integer number of slices f = round(target / native z). Cases with f = 1 at a target are left out of that
  level, and the achieved thickness (f * native z) is recorded, because it is not exactly the target. In the
  shared-PET path both channels are on one grid, so CT and PET are re-sliced together. Conditioned models
  (B4, B5) receive the conditioning vector of the new spacing; B3 receives none; resampled models get the
  thick volume interpolated back to their grid. Predictions are mapped back and scored against the unchanged
  native label. Reported: per-case mean Dice and HD95 per model per level, and the paired difference B4 minus
  B3 per level with its bootstrap interval. Descriptive.
- Wrong-spacing check: B4 is run on every test case with the spacing vector of another test case (one fixed
  seeded permutation with no case keeping its own vector; both the conditioning vector and the mm spacing
  input are swapped). Reported: paired difference in per-case mean Dice, correct minus shuffled, with its
  bootstrap interval, and the same split by whether the swapped z differs from the true z by more than 1 mm.

## 7. Calibration and uncertainty

- Sets: in distribution = the 67 validation cases (training centres); out of distribution = the 123 test cases.
- For B4 and B5: ECE with 15 equal-width bins, multi-class Brier score, NLL, and reliability diagrams, each
  over all voxels and inside a region of interest (the union of truth and prediction foreground dilated by 5
  voxels), pooled over cases.
- Temperature scaling for B4: one scalar fitted by minimising NLL on the validation set only, then applied
  unchanged to test. B4 is reported with and without it. B5 is not temperature-scaled.
- B5 uncertainty u = K / S: AUC for detecting wrongly classified voxels inside the region of interest, per
  case and pooled; risk-coverage curve (error rate against the fraction of voxels kept, most certain first);
  mean u on test against validation (reported with a bootstrap interval of the difference in case means);
  Spearman correlation across test cases between mean u and (1 - per-case mean Dice).

CODED: ECE, Brier, reliability bins, AUC. PLANNED: NLL, temperature scaling, risk-coverage curve.

## 8. Efficiency

Preprocessing wall-clock per volume and peak GPU memory at inference, resampled path against native path,
on the same GPU, same patch size and same sliding-window settings. Measured, never estimated; a value that
was not measured is written as "not measured". Needs a GPU, so nothing is measured yet.

## 9. What counts as support, and what counts against

Hypothesis: conditioning a native-resolution PET/CT network on voxel spacing (B4) segments better on unseen
centres than the same network without the conditioning (B3).

Support: B4 has a higher per-case mean Dice than B3 with a Holm-adjusted p below 0.05 and a 95% interval of
the median paired difference that lies above zero. It is stronger support if the wrong-spacing check lowers
B4's Dice and the thickness sweep shows the B4 minus B3 difference not shrinking as slices get thicker.

Against: the interval of the median paired difference between B4 and B3 includes zero or lies below it, or
the Holm-adjusted p is 0.05 or more. It also counts against the mechanism, even if B4 scores higher, when
shuffling the spacing vectors leaves B4's Dice unchanged, because then the gain does not come from the
spacing information.

A result on one split, with one test of two centres and possibly one seed, is evidence about this dataset
only. No claim is made about clinical use.

## 10. Mapping to E1-E6 (`src/eval/experiments.py`) and differences

| Plan item | Script | Difference today |
|---|---|---|
| Sections 3, 5: per-case metrics, centre tables | E1 | E1 reads the old two-grid loader and scores merged foreground; the plan needs the `.acf` loader and per-label metrics. Replaced by `scripts/evaluate_test.py` (Phase 7). |
| Section 6: thickness sweep | E2 | E2 uses levels 2, 3, 5, 7 mm on CT only (two-grid path); the plan uses 3, 5, 7 mm on the shared grid and drops cases with f = 1. The wrong-spacing check has no E script. |
| Section 8: efficiency | E3 | Same intent; E3 uses the old loader. Needs a GPU. |
| Section 7: calibration, uncertainty | E4 | E4 has ECE, reliability and AUC; Brier is coded but not wired in; NLL, temperature scaling and risk-coverage are not written. |
| Section 4: comparisons | E5 | Matches the plan after this commit (Wilcoxon, Holm, median difference, 10,000 resamples), given a per-case table with the plan's metric. |
| Section 4: statistics | E6 | Same statistics code as E5; E6's "every model against one reference" family is not the plan's family of five. Use E5 with the five pairs. |

## 11. Changes after the first test run

None.
