# Handover notes (branch `fable/first-run`)

State as of 2026-11-15. Nothing here is a result; no model has been trained on a GPU.

## Standing rule

Keep the repo and any HECKTOR file out of synced folders (OneDrive, Google Drive G:, Dropbox).
The repo lives at `D:\misc\Hecktor`; the cache lives at `D:\hecktor_cache`.

## What to run

Use `configs/shared_b1..b5` and `shared_b2b` (pet_mode shared, `.acf` cache, K = 3, test centres
CHUS + CHUV). The `configs/b1..b5` and `b2b` family (pet_mode rederived) is parked: rederived
mode is not wired to the `.acf` cache and has been dropped for now.

## Done and verified

- Phase 1 cleanup (commit 4eea17e).
- Phase 2 audit: `results/fable_run/AUDIT.txt` (782 cache files, no inf/NaN, split 592/67/123,
  conditioning stats fitted on the 592 training cases only).
- PET is divided by its per-case scale and clipped to [0, 20] by one rule for every case (commit 75fea66).

## Not done

Deferred by the owner (16 Nov 2026), not wanted now:

- Phase 3 item 7: loader seconds per fetch, each path, 1 and 2 workers.
- Phase 3 item 8: `docs/DATA_REPORT.md` update from script outputs.
- Phase 8: timing notebook and nnU-Net preparation.

Dropped by the owner: rederived PET mode (shared PET only).

Not built, by decision: the stored resampled cache for B1/B2 (20-case projection 35.3 GB, over the
30 GB limit). B1/B2 resample each cropped volume at load time instead (`data.mode: resampled`).
`scripts/build_resampled_cache.py` can build the stored cache later on a machine with the disk for it.

Still open:

- Phases 4 to 7, 9 and 10 (see `results/fable_run/PROGRESS.txt` for the current phase).
- `src/eval/experiments.py` E1-E6 still use the old two-grid loader and Bonferroni (Phase 5 replaces
  Bonferroni with Holm).
- Open decisions: `results/fable_run/NEEDS_DECISION.txt` items 2, 3, 4, 6, 7.
