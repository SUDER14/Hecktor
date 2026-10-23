# 09 Code reading order

Read top to bottom; run the named test after each step to see the code work.

| # | File | Look for | Test |
|---|---|---|---|
| 1 | `src/data/geometry.py` | the extent rule: voxels x spacing; why tolerance is one *coarser* voxel | `tests/test_geometry.py` |
| 2 | `src/data/physical_crop.py` | Grid conventions (edge vs centre), joint clamp, `pad=True` | `tests/test_physical_crop.py`, `test_phase1_data.py` |
| 3 | `src/data/synthetic.py` | how one world offset is shared by CT, PET and label | `test_phase1_data.py::test_synthetic_spacings...` |
| 4 | `src/data/hecktor2026.py`, `splits.py`, `conditioning.py` | centre from id prefix; group splits; train-only stats | `test_phase1_data.py` |
| 5 | `src/data/dual_cache.py` | cache layout, deterministic crops per (seed, epoch, patient) | `test_phase1_data.py::test_crop_draws...` |
| 6 | `src/models/unet3d.py` | the plain U-Net; shape comments on every line | `test_phase2_models.py::test_unet3d_shapes` |
| 7 | `src/models/dual_encoder.py` | B2: fusion by level index | `test_structure_matches_b2_when_aligned` |
| 8 | `src/models/hypernet.py` | zero-init, `film`, Tier 2 init subtlety | `test_film_identity_at_init...`, `test_tier2...` |
| 9 | `src/models/acf.py` | level table, `pet_offset`, `_check_level`, forward | `test_acf_production_geometry_96mm_box` |
| 10 | `src/models/evidential.py`, `src/training/losses.py` | alpha, S, p, u; CE and KL | `test_phase3_losses_metrics.py` |
| 11 | `src/metrics/*.py`, `src/eval/metrics.py` | empty-mask convention, ECE binning | same file |
| 12 | `src/training/run.py` | micro-batches + accumulation, logging, resume | `test_phase4_train.py` |
| 13 | `src/eval/infer.py`, `evaluate_cases.py` | PET window from the CT window; resample-back | `test_phase4_eval.py` |
| 14 | `src/eval/stats.py`, `experiments.py` | pairing, Bonferroni, git hash in every row | `test_phase4_eval.py` |
| 15 | `scripts/smoke_test.py`, `overfit_one_batch.py` | what a pipeline check proves (and does not) | run them |

Habit: before reading a function, write down the tensor shape you expect going in and out; then check the
comment.

## Exercises
Five per chapter are at the end of each chapter-specific section in `docs/learn/exercises.md`; answers are
separate in `docs/learn/exercises_answers.md`. Try each before opening the answers.
