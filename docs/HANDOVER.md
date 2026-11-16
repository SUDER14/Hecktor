# Handover (branch `fable/first-run`)

State on 2026-11-16. Read this first; it replaces the conversation that produced the branch.
Nothing in this repository is a result: no model has been trained on a GPU and no model output has
been computed on a test case. `main` is untouched (c7d16b6). Do not merge into `main`.

## Rules that must not be broken

- Licence: no HECKTOR file (image, label, cache, clinical CSV, prediction) leaves the owner's machine.
  Keep the repo and any HECKTOR file out of synced folders (OneDrive, Google Drive G:, Dropbox).
  Real data on Colab or Kaggle only after written permission from the data provider; the code refuses it
  unless `ALLOW_REAL_DATA=1`.
- The test set (CHUS 72 + CHUV 51 = 123 cases) is touched only by `scripts/evaluate_test.py`, once per
  model, after `docs/ANALYSIS_PLAN.md` is committed. It informs no decision.
- Before every push: `git ls-files` must show nothing from `results/`, no CSV, no cache file, no image.
- This machine: 7.8 GB RAM, 4 cores, no GPU. One heavy job at a time, under a 6.5 GB RAM guard.
- Never invent a number. Never weaken a check to make it pass. Spacing-conditioned hypernetworks are
  prior art (HyperSpace, MICCAI 2024); do not call them novel.

## Decisions and why

- Split: `configs/splits/hecktor2026_seed42.json`, 592 train / 67 validation / 123 test. Test = two whole
  centres, so the test measures transfer to unseen centres. Conditioning statistics are fitted on the 592
  training cases only (`..._conditioning_stats.json`).
- K = 3 (labels 0, 1, 2). What labels 1 and 2 denote clinically is NOT verified from the dataset
  documentation; all code and documents refer to them by number.
- Crop rule: each case is stored as a cropped `.acf` file (782 files in `D:\hecktor_cache`, outside the
  repo). Rule and checks: `src/data/crop_rule.py`, `scripts/crop_verify.py`, `docs/DATA_REPORT.md`.
- PET rule: divide by the per-case 99.5th percentile of positive PET inside CT > -500 HU (sidecar
  `pet_scale.json` in the cache folder), then clip to [0, 20]. CT: clip [-1000, 1000] HU, scale to [0, 1].
  One rule for every case.
- pet_mode: shared only (CT and PET as two channels on one grid). The rederived-PET mode was dropped by
  the owner; `configs/b1..b5`, `b2b` are parked. Run `configs/shared_b*.yaml`.
- B1, B2, B2b run on a resampled grid (0.9766 x 0.9766 x 3.27 mm, the training median), resampled at load
  time; the stored resampled cache was not built (projected 35.3 GB, over the 30 GB limit).
  `scripts/build_resampled_cache.py` can build it later. Predictions are mapped back to the native grid
  before scoring. A perfect resampled prediction scores foreground Dice 0.9025 / 1.0000 / 0.9883 after
  mapping back on three training cases (z 1.25 / 3.27 / 5.0 mm): a ceiling the resampled baselines pay.
- Statistics: paired Wilcoxon by case, Holm over five comparisons, primary B4 against B3
  (`docs/ANALYSIS_PLAN.md`). Holm replaced the proposal's Bonferroni.

## Done, with commit

| Phase | Commit | What |
|---|---|---|
| 1 cleanup | 4eea17e | dead Decathlon files removed, audit scripts |
| 2 audit | (no commit, `results/fable_run/AUDIT.txt`) | 782 files, no inf/NaN, split and stats verified |
| 3 data pipeline (partial) | 75fea66, 5e05ec1 | PET clip, K = 3 configs, parked configs |
| 4 correctness gates | 9d563f1, 76c722a | `scripts/phase4_gates.py`, results in `results/fable_run/gates.json` |
| 5 analysis plan | f363575 | `docs/ANALYSIS_PLAN.md`, Holm in `src/eval/stats.py` |
| 6 portable training | c9a6f88 | `scripts/train.py`, `src/training/portable.py`, `notebooks/colab_train.ipynb` |
| 7 locked test script | 042c36e, merge e1b1d54 | `scripts/evaluate_test.py`, `src/eval/locked_test.py` (written on `fable/analysis`) |
| 7 follow-up | e9c57c5 | train-then-evaluate integration test, first Phase 10 handover |
| 9 review | 5f1071a, e71a687, 6c0fb6d, c96e90e, merge fe332a2 | four findings R1 to R4, each fixed with a test (written on `fable/analysis`) |
| trainer side of R3 and R4 | 45bfad2 | `scripts/train.py` stores no machine path and hashes text with CRLF normalised |

Phase 4 gates, latest real run of each: PASS for split leak, fresh-weights identity of B3 and B4,
B5 head properties, same-seed repeat, resume, spacing sensitivity after 50 steps (B4 yes, B3 no),
sliding window on one validation case for all six configs, foreground overfit for all six, per-label
overfit for B1, B2, B2b, B3, B4. FAIL: per-label overfit for B5 (below).

## B5 label-1 note (open)

B5 (ACF + evidential head) does not learn label 1 on the fixed overfit batch: label 1 Dice 0.000 after
212 steps (14 minutes), and again 0.000 after 170 steps with the KL term switched off. The 272 label-1
voxels end up predicted as background; class-1 evidence there moved only 0.50 to 0.54. Ruled out by those
runs: the KL schedule, label order, patch sampling. NOT established: the cause, and whether it is slow
or stuck. Details: `results/fable_run/B5_note.txt`. Options (rescaled evidence, loss weights, first a
lr 3e-3 run and a 60-minute run): `NEEDS_DECISION.txt` item 8. `scripts/phase4_b5_label1_diag.py` runs
the diagnosis (`--only lr_3e-3` is the variant not yet run). Do not train B5 on a GPU until it passes
`python scripts/phase4_gates.py --out results/fable_run/gates.json --only overfit_labels:shared_b5`.

## Review findings (Phase 9, all fixed)

Full text: `results/review/REVIEW.txt` (not in git). Tests: `tests/test_review_fixes.py`,
`tests/test_train_then_evaluate.py`.
- R1, high: sliding-window inference applied softmax twice for softmax heads. The predicted class was
  unchanged; probabilities were flattened (0.96 became 0.56), which corrupts ECE, Brier, NLL, temperature
  and uncertainty for B1 to B4. Any earlier ECE or uncertainty value for a softmax model is invalid.
- R2, high: the one-shot test lock could be bypassed by renaming the run or deleting the lock file.
- R3, medium: machine paths from the checkpoint config were carried into the test run. The trainer now
  stores split and statistics paths relative to the repository and blanks the cache, zip and run paths.
- R4, medium: text-file hashes depended on line endings. Trainer and test script now hash the split,
  statistics, config and plan with CRLF normalised to LF.

## Not done

Deferred by the owner (16 Nov 2026):
- Phase 3 item 7: loader seconds per fetch for each path, 1 and 2 workers.
- Phase 3 item 8: `docs/DATA_REPORT.md` update from script outputs.
- Phase 8: timing notebook (`notebooks/colab_timing.ipynb`) and nnU-Net preparation.
Dropped by the owner: rederived PET mode.
Not built by decision: stored resampled cache.

Half-done or not started:
- Not covered by the review: resume and gradient accumulation under mixed precision on a real GPU (no GPU
  here), and anything that only shows with real training.
- Nothing has run on a GPU: no timing, no memory figure, no validation Dice. The Colab kill-and-resume
  check with `--synthetic` is for the owner to run (steps in `notebooks/colab_train.ipynb`).
- `steps.max_steps: 14800` in the six configs only restates the old 100 epochs; not tuned. Number of
  seeds not decided (`NEEDS_DECISION.txt` item 9).
- `scripts/analyse.py` (the five planned comparisons plus nnU-Net, from `results/test/<run>/per_case.csv`
  to `results/stats/comparisons.csv`) is written and tested on synthetic tables only; it has read no real result.
- `src/eval/experiments.py` E1-E4 use the old two-grid loader; do not use them for the shared-PET runs.
- `CLAUDE.md` still describes the Decathlon stage and Colab-plus-Drive checkpointing; it is out of date
  (`NEEDS_DECISION.txt` item 7).
- Open decisions: `results/fable_run/NEEDS_DECISION.txt` items 2, 3, 4, 6, 7, 8, 9. That folder is not in
  git; it lives only on the owner's machine.

## Known risks and open questions

- B5 as above. The B5-against-B4 comparison in the plan is conditional on B5 passing its gate.
- Run manifests written before the trainer fix hold raw-byte hashes; the test script accepts both forms of
  the same text. `.gitattributes` also pins LF for the split, statistics, config and plan files.
- Resume is bit-identical on CPU only (tested). On a GPU, cuDNN is not bit-reproducible.
- The lock cannot stop someone deleting the whole `results/test` folder or editing the script; it is a
  guard against mistakes, not against intent.
- Label 1 is rare (2 to 3 percent of foreground in the gate batch); softmax models needed 90 to 160 steps
  before label 1 Dice left zero on the overfit batch. Watch per-label validation Dice, not the mean alone.
- The unseen-spacing subgroup (77 cases) contains all 72 CHUS cases, so spacing and centre cannot be
  separated in that subgroup.

## Commands

Tests (about 10 minutes on this machine):
    python -m pytest -q
Cache audit (read-only):
    python scripts/audit_cache.py --cache_dir D:/hecktor_cache --out results/fable_run/audit_cache.json --scan
Cache build from the zip (already done; only if the cache is lost):
    python scripts/build_hecktor_cache.py --help
Correctness gates (CPU, about 2.5 hours for all):
    python scripts/phase4_gates.py --out results/fable_run/gates.json
One training run (clean git tree required):
    python scripts/train.py --config configs/shared_b4_acf_cond.yaml --data-root D:/hecktor_cache --out-root <folder> --resume auto
Whole loop without data:
    python scripts/train.py --config configs/shared_b4_acf_cond.yaml --out-root <folder> --synthetic --set steps.max_steps=40 --set steps.val_every=20
Locked test, once per finished run (clean tree, committed plan):
    python scripts/evaluate_test.py --run-dir <folder>/runs/shared_b4_acf_cond_seed42 --data-root D:/hecktor_cache --out-root <folder outside the repo>

## Exact next prompt

"Resume on branch fable/first-run from docs/HANDOVER.md and results/fable_run/PROGRESS.txt. All rules in
the handover apply. Run the B5 label-1 diagnosis (NEEDS_DECISION item 8):
`python scripts/phase4_b5_label1_diag.py --out results/fable_run/b5_label1_diag.json --only lr_3e-3`
(17 minutes, RAM guard on). Then decide the B5 fix, implement it, and rerun the B5 gate:
`python scripts/phase4_gates.py --out results/fable_run/gates.json --only overfit_labels:shared_b5`.
Keep the pass rule (Dice above 0.9 for both labels); record FAIL until it passes. The Phase 9 review is
already done; see results/review/REVIEW.txt. Do not touch the test set. Do not merge into main. Report in
plain text under 60 lines."
