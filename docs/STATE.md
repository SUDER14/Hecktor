# Repository state (Phase 0 audit)

Audit date: 2026-10-23. Local `main` equals `origin/main` at audit start (`git status -sb`: `main...origin/main`, no ahead/behind).

## Exists and works (VERIFIED: `python -m pytest -q` -> 70 passed, 249 s, CPU, torch 2.14.0+cpu, MONAI 1.6.0)
- `src/data/`: Decathlon + HECKTOR (old layout) adapters, centre/random splits, conditioning stats (train-only), world-coordinate CT/PET crop (`physical_crop.py`), geometry extent check, preprocessing cache (single-modality), seeds.
- `src/models/`: `baselines.py` (MONAI UNet, B1), `dual_encoder.py` (B2-style dual encoder, no conditioning, physical-extent check).
- `src/training/`: train loop (single-modality only), checkpoint/resume, overfit-one-batch.
- `src/eval/metrics.py`: HD95 (scipy EDT, mm, NaN-safe).
- `frontend/`: Streamlit exploration dashboard (Decathlon only).

## Missing at audit (the work of this run)
- HECKTOR 2026 layout adapter (`hecktor2026_training/<CENTRE-NNN>/...` + Task CSV).
- Conditioning vector `[sx, sy, sz, modality_flag]` (old one was 5-d with two flags).
- Synthetic PET/CT generator, spacing manifest script, centre-grouped split guard for tiny centres.
- ACF model, hypernetwork/FiLM, evidential head, losses, ECE/Surface Dice/uncertainty metrics.
- Config-driven dual-modality training, sliding-window inference writing NIfTI, E1-E6 scripts, nnU-Net scripts, runners, app, docs/learn, notebooks.

## Not pushed / untracked
- `CLAUDE_CODE_MASTER_PROMPT.md`, `a.py` (scratch viewer), `1NF & 2NF, Step by Step.pptx` (unrelated; left untouched, never committed).

## Known gaps / risks
- Nothing has run on a GPU. HECKTOR data is unavailable. PET SUV assumption unverified.
- `PROJECT_CONTEXT.md` / `PROJECT_PLAN.md` named in the master prompt do not exist; `CLAUDE.md` was used instead.
