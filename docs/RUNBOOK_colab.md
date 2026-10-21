# Colab runbook -- Baseline 1 on Task06_Lung

Notebook: `notebooks/colab_train_baseline.ipynb`. Config: `configs/train_baseline_decathlon_lung.yaml`.

**Status:** the notebook and the Drive path have not yet been run on a real Colab/Drive session. Resume logic is tested locally (`tests/test_train_resume.py`), including a truncated-latest-checkpoint case, but not against a real Drive mount. If something breaks on first launch, that is where to look first.

## One-off setup (local machine)

1. The repo is pushed to `https://github.com/SUDER14/Hecktor.git` (including `configs/splits/*.json` -- the run needs the split and conditioning stats). Commit and push before each run so the logged commit matches the code. A private repo needs a token in the URL (`https://<token>@github.com/...`) or a Colab secret.
2. Upload `data/Task06_Lung` to Drive at `MyDrive/hecktor_major_project/data/Task06_Lung` (`dataset.json`, `imagesTr/`, `labelsTr/`).
3. Optional: `wandb login` account; set `USE_WANDB = True` in the notebook.

## Start a run

1. Open the notebook in Colab. Runtime > Change runtime type > GPU.
2. Section 1: set `REPO_URL`, `RUN_NAME` (unique per experiment -- it names the Drive checkpoint folder), `USE_WANDB`.
3. Run all cells. Check the first lines of the launch output:
   - `device: cuda`
   - `git commit: <40-hex>  dirty: False` -- if it says `dirty: True` or `None`, the run is not traceable; fix before running anything you intend to report.
4. **Do a 2-epoch timing check first**, with a *different* `RUN_NAME` (e.g. `timing_check`) and `EXTRA_OVERRIDES = ['train.n_epochs=2']`. A different name matters: otherwise the real run would later resume from the timing run's checkpoints. Read the `(data Xs, gpu Ys, total Zs)` per-epoch breakdown from the log. The cached-pipeline epoch time has **not been measured yet**; this is how you get the number.

## Build the preprocessing cache (once per Colab VM, before training)

The production config reads `cache.dir` (`cache/decathlon_lung_1mm`, local disk) and **refuses to start without it**. A 2-epoch probe of the old uncached pipeline on a T4 with data read from Drive took 695 s/epoch; the cache removes load + resample from the training loop. The cache lives on the VM's local disk, so rebuild it after any runtime reset (the script skips valid entries, so rerunning after a crash is safe). Copy the data off Drive first -- reading volumes from Drive during the build is slow:

```
!cp -r /content/drive/MyDrive/hecktor_major_project/data/Task06_Lung /content/Task06_Lung
!python scripts/precompute_cache.py --config configs/train_baseline_decathlon_lung.yaml --data_dir /content/Task06_Lung --cache_dir cache/decathlon_lung_1mm
```

Expected size from the NIfTI headers: ~9.6 GB at the default float16 (16.0 GB with `--image_dtype float32`); float16 quantises the CT window by up to ~0.68 HU. Build time on Colab: NOT RUN (locally one 512x512x112 volume spent ~65 s of ~75 s in the resample step). The notebook (`notebooks/colab_train_baseline.ipynb`) has **not** been updated with this cell -- run it manually before the training cell.

Each epoch now logs `(data Xs, gpu Ys, total Zs)`, and flags `bottleneck: data loading|GPU compute` when total exceeds 90 s (`train.target_epoch_s`). The 90 s target is **not yet measured on Colab**.

## Resume after a session dies

Re-open the notebook, keep the **same `RUN_NAME` and settings**, run all cells. Section 7 uses `--resume`: it loads the newest readable checkpoint from `checkpoints/<RUN_NAME>/` on Drive (falling back to an older one if the newest is truncated), restores model/optimizer/scaler/scheduler/RNG, and continues from the next epoch. The log prints `resumed from <path> at epoch N`. If it prints `starting fresh`, the run name or Drive folder does not match -- stop and check.

- Work lost per crash: up to one epoch (checkpoints are per-epoch, not mid-epoch).
- Do not change model/data/optimizer settings between sessions. Overrides are not compared across sessions; a changed model shape fails to load, but a changed lr would silently apply.
- The wandb run continues under the same run id (stored in the checkpoint).

## Pull results down

Drive holds `checkpoints/<RUN_NAME>/`: `epoch_*.pt` (last 3), `best.pt` (best val Dice), all ~58 MB each (measured; ~230 MB per run).

- Easiest: download `best.pt` from drive.google.com, or use Drive for Desktop and copy the folder to `results/checkpoints/<RUN_NAME>/` (gitignored).
- Metrics (val Dice, HD95, loss, lr per epoch) are in wandb if enabled; otherwise only in the notebook's cell output -- copy it before closing the tab.
- Traceability: each checkpoint stores the commit it came from: `torch.load('best.pt', weights_only=False)['git']`. The same commit is in the wandb run config under `git`.

## Known risks (unverified)

- **RAM.** The production config resamples to 1 mm isotropic; on an 8 GB local machine that ran out of memory, which is why the smoke-test config uses 2 mm. Colab RAM may or may not suffice; watch the RAM gauge on the first epoch. Bump to a high-RAM runtime if it dies.
- **The overfit-one-batch smoke test has passed only at CPU smoke scale** (`configs/cpu_smoke.yaml`: 32³ patch, 2 mm grid, one crop; fg Dice 0.9187 at iter 55) -- a pipeline check, not evidence about the production 96³ / 1 mm config. On the GPU, run it once first (`!python -m src.training.train --config $CONFIG --overfit-one-batch`) and read the result before a long run.
- **Cached-pipeline epoch time is unmeasured.** The old uncached path took 695 s/epoch (45 train volumes, 96^3, T4, Drive). Whether the cache reaches the 90 s target is unknown until the 2-epoch timing check; the per-epoch breakdown names the bottleneck if it does not. Validation still reads whole volumes (`val_every: 5`) and runs the CPU HD95 -- not included in the per-epoch training time.
- **Resume reproducibility:** cached crops are a pure function of (seed, epoch, volume), so resumed runs redraw the same crops. Cached and uncached runs draw different crops from the same seed, so their numbers are not bit-comparable.
