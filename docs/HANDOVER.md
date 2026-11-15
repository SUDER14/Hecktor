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

- Phase 3 item 5: stored resampled cache for B1/B2 (20-case projection 35.3 GB, over the 30 GB
  pre-approval). Resampled mode currently resamples at read time.
- Phase 3 item 6: pet_mode rederived wired to the `.acf` cache (dropped for now).
- Phase 3 item 7: loader seconds per fetch, each path, 1 and 2 workers.
- Phase 3 item 8: `docs/DATA_REPORT.md` update from script outputs.
- Phase 3 item 1: sidecar test on one MDA-402..447 case.
- Phases 4 to 10 (correctness gates, analysis plan, and everything after).
- `src/eval/experiments.py` E1-E6 still use the old two-grid loader and Bonferroni.
- Open decisions: `results/fable_run/NEEDS_DECISION.txt` items 2, 3, 4, 6, 7.
