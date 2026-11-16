# ACF Studio, real-data edition

A local, read-only viewer on the cropped HECKTOR cache. It is a port of the two reference pages in
`app/studio/reference/` (kept for comparison, never served).

There are two viewers in this repository; they do not share code:

| Viewer | Start | What it is |
|---|---|---|
| ACF Studio (this folder) | `python -m app.studio` | FastAPI backend plus one page; real HECKTOR cases from the `.acf` cache |
| Streamlit viewer (`app/streamlit_app.py`) | `streamlit run app/streamlit_app.py` | the earlier viewer, unchanged |

## Start

    pip install -r app/studio/requirements.txt
    python -m app.studio          # prints http://127.0.0.1:8765

Paths (defaults follow `docs/HANDOVER.md`; override with environment variables):

| Variable | Default | Used for |
|---|---|---|
| `ACF_STUDIO_CACHE_DIR` | `D:/hecktor_cache` | `<case>.acf` files and `pet_scale.json` |
| `ACF_STUDIO_SPLIT` | `configs/splits/hecktor2026_seed42.json` | train / validation / test column |
| `ACF_STUDIO_META_CSV` | `results/crop/crop_scan.csv` | whole-scan PET maximum, for the PET-scale outlier flag |
| `ACF_STUDIO_PORT` | `8765` | port; the host is always 127.0.0.1 and cannot be changed |

A file that is absent is named in the page and the fields it feeds stay empty. Nothing is guessed.

## What each tab reads

Built in three stages. Stage 1 is done; the other tabs show a note saying which stage brings them.

| Tab | Stage | Reads |
|---|---|---|
| Case browser | 1 | every `.acf` header (centre, spacing), the split file, `pet_scale.json`, the scan CSV |
| Volume | 1 | one case's CT, PET and label through `src/data/crop_cache.py` (memory-mapped) |
| Fusion | 1 | the same display volume; three planes or CT / PET / fused, label outlines, synced crosshair |
| Spacing | 2 | conditioning statistics file; `results/runs/<run>/film_params.json` if present |
| Segment | 2 | `<out-root>/test_predictions/<run>/<case>_pred.nii.gz` and `<case>_uncertainty.nii.gz` (written by `scripts/evaluate_test.py`) |
| Layers | 3 | the display volume |
| Status | 3 | `docs/DATA_REPORT.md`, `docs/RUN_QUEUE.md`, `results/runs/*/manifest.json`, `results/test/*`, git HEAD |

With no case selected the page shows a synthetic phantom built in the browser, labelled "Synthetic phantom". The
phantom's evaluation numbers come from a fixed demo rule and are labelled so; that rule never runs on a real case.

Display volume: at most 256 voxels in-plane (every f-th voxel, label by block maximum), slices native, spacing
reported after the reduction. PET is the released value divided by the case's own scale and clipped to [0, 20],
which is the array the training loader gives the network (SUV-like, units not stated). Labels are called
"label 1" and "label 2"; their clinical meaning is not verified.

## Licence rules the app enforces

The HECKTOR agreement limits use to the VIT site and bars third-party access.

- The server binds to 127.0.0.1 only, refuses any request not addressed to 127.0.0.1 or localhost, and answers
  GET only. The page makes no request to any other host: fonts are vendored in `static/vendor/` (versions and
  licences in `VERSIONS.txt`), there is no CDN and no analytics, and a Content-Security-Policy of `'self'`
  blocks anything else.
- No button or endpoint exports, downloads or saves a patient image.
- Every view of a real case carries the watermark "Research use only. HECKTOR data. Not for publication." and is
  hidden when the page is printed.
- No patient image, cached volume, prediction or screenshot is committed (`.gitignore`, and a test that fails if
  one is tracked). Screenshots taken while testing go to `D:\misc\studio_shots`, outside the repository.
- Test-set cases are marked "test set: view only, not for tuning".

Tests: `python -m pytest tests/test_studio_stage1.py -q` (the real-cache tests use training cases only).
