# 07 Experiments and statistics

Runner: `scripts/run_experiments.py`; library: `src/eval/experiments.py`, `src/eval/stats.py`. Every output row
carries the git hash. Configs B1..B5: `configs/README.md`.

| Exp | Question | Method |
|---|---|---|
| **E1** held-out centre | Does the model work on a hospital it never saw? | Evaluate on test centres only; per-case + per-centre tables |
| **E2** anisotropy | Does conditioning help when slices get thicker? | Block-average CT slices along z to 2, 3, 5, 7 mm; ACF is told the new spacing via conditioning, B1/B2 get the thick slices interpolated back to their grid; score against the unchanged native label |
| **E3** resources | What do the designs cost? | Peak VRAM (CUDA only; otherwise written as "not measured"), preprocessing wall-clock per patient, inference time per case |
| **E4** calibration | Is the evidential head better calibrated? | ECE (15 bins), reliability diagrams, uncertainty-vs-error AUROC/overlap |
| **E5** ablations | What does each component buy? | Paired comparisons B2 vs B1, B3 vs B2, B4 vs B3, B5 vs B4 |
| **E6** statistics | Are differences real? | Paired Wilcoxon, Bonferroni, effect sizes, bootstrap CI, per-centre tables |

## Fair scoring
All models are scored on the native CT grid against the native label. B1/B2 predict on a resampled grid, so
their probability and uncertainty maps are resampled back before argmax (`evaluate_cases.py`).

## E2 caveat
Block-averaging is a simple model of thick-slice acquisition (no slice profile, no noise change). It tests
sensitivity to a coarser z-spacing, not real thick-slice scanners.

## Statistics
* Pair by patient: `patient_id` is the pairing key. Cases with NaN for either model are dropped pairwise and
  counted (`n_dropped_nan`).
* Wilcoxon signed-rank (two-sided). The smallest attainable p with n pairs is 2/2^n, so tiny test sets cannot
  give small p-values; `n_pairs` is in every row.
* Bonferroni: p_adj = min(1, p * m), m = number of comparisons actually in the table.
* Effect size: rank-biserial r = (W+ - W-)/(W+ + W-); Cohen's d_z = mean(diff)/sd(diff).
* Bootstrap 95% CI of the mean paired difference (seeded).
* Per-centre tables: mean/std/count per (model, centre). With few cases per centre treat them descriptively.

## Pitfalls
1. Test centres chosen after looking at results. Fix them in the config first.
2. Using the validation split for model selection *and* reporting.
3. Mean Dice that silently excludes empty-label cases.
4. Comparing models of different capacity without saying so (chapter 04).
5. Single seed. Differences smaller than seed-to-seed variation mean nothing; run several seeds for the headline
   comparisons if compute allows.
