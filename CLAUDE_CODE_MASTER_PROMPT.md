You are working in the repo github.com/SUDER14/Hecktor (local clone). Set git identity to name "Sudarshan" for every commit. Never push anything under another name.

## CONTEXT (read first)
M.Tech dissertation, VIT Vellore: Acquisition-Conditioned Fusion (ACF). Spacing-conditioned dual-encoder 3D U-Net for head-and-neck PET/CT tumour segmentation on HECKTOR 2026, with calibrated evidential (Dirichlet) uncertainty. Read PROJECT_CONTEXT.md, PROJECT_PLAN.md and every file under src/, configs/, scripts/, tests/ before changing anything. Prior art: HyperSpace (MICCAI 2024) already conditions a U-Net on spacing with a hypernetwork. Never write "first" or "novel" in code, docs or comments. The thesis matters more than the patent.

## NON-NEGOTIABLE RULES
1. Never state a result, speed, VRAM figure or Dice you did not measure in this session. If something cannot run here (GPU, HECKTOR data, Kaggle), write the code and tests anyway and list it under "NOT RUN" in the final report.
2. Every claim in the final report is tagged VERIFIED (command + output shown) or NOT VERIFIED.
3. Do not weaken a test to make it pass. Fix the code.
4. HECKTOR data is not available yet. Build and test everything with (a) a synthetic PET/CT generator with realistic different spacings and (b) Decathlon Task06 Lung for CT-only pipeline checks.
5. Commit in small steps with clear messages. Push to origin main at the end of each phase. At the very end run `git ls-remote origin main` and show that the pushed commit equals local HEAD.

## PHASE 0: AUDIT
List what exists, what is not pushed, what is broken. Write docs/STATE.md. Run the test suite and report.

## PHASE 1: DATA LAYER (native spacing, no resampling of inputs)
- HECKTOR adapter for the layout hecktor2026_training/<CENTRE-NNN>/<CENTRE-NNN>__CT.nii.gz, __PT.nii.gz, <CENTRE-NNN>.nii.gz (labels), plus HECKTOR_2026_Training_Task.csv. Centre = the prefix before the dash.
- Read spacing from the NIfTI header (sx, sy, sz) for CT and PET separately. Conditioning vector per modality: [sx, sy, sz, modality_flag] (CT flag 0, PET flag 1), z-scored with TRAIN-ONLY statistics saved to a json and reused at val/test.
- PHYSICAL ALIGNMENT (critical): pick a patch as a world-coordinate box (in mm) centred on a sampled point; crop CT and PET from the SAME box at each modality's native spacing, so a 96 mm box is 96^3 voxels at 1 mm for CT and 24^3 at 4 mm for PET. Use the affine for world to voxel. Raise a clear error if the physical extents of the two crops differ by more than one voxel. Pad in a world-consistent way near borders. Unit-test this with a synthetic scan pair.
- Label crop on the CT grid. Sampling oversamples tumour voxels (configurable ratio).
- Centre-grouped splits: train / val / held-out test centres chosen in a config; never split a patient across sets; tiny centres (HMR, USZ) are not used alone as the test set. Save the split to json with a seed.
- Spacing manifest script: one CSV for all scans (spacing, shape, centre, anisotropy), plus histogram figures per centre.
- Caching to local fast disk (Kaggle /tmp, not /kaggle/working) so epochs are fast.
- Synthetic generator: random blobs as tumours, PET at 4 mm and CT at 1 mm (also 1x1x3 mm variants), several fake centres.

## PHASE 2: MODELS (src/models/)
All with exhaustive shape comments on every forward line.
- B1: UNet3D single-modality baseline (resampled input).
- B2: DualEncoderUNet (resampled PET/CT, no conditioning): CT and PET encoders, fusion by concat + 1x1x1 conv, shared decoder.
- ACF: same as B2 but native spacing: separate CT (5 levels, 16-32-64-128-256) and PET encoders (depth configurable per modality; for a 24^3 PET patch use 3 levels: 16@24^3, 32@12^3, 64@6^3), fusion only at levels where CT and PET voxel size match (L3, L4, L5), L1/L2 CT-only skips. Fuse by resizing PET feature maps to the CT feature grid, but first CHECK that physical extents match (patch_mm / voxel_mm); fail loudly if not.
- Hypernetwork (Tier 1): MLP vector -> gamma, beta for every block of that modality's encoder. FiLM F' = gamma*F + beta, initialised gamma=1, beta=0 (output = 1 + delta). PET and CT conditioned independently. `conditioning_zeroed` flag for the ablation.
- Tier 2 (flag, off by default): hypernetwork outputs conv kernels directly. Implement only after Tier 1 tests pass.
- Evidential head: 1x1x1 conv -> K logits -> softplus -> evidence e; alpha = e+1, S = sum alpha, p = alpha/S, u = K/S. K = 2 (background, tumour); make K configurable.
- Tests: output shapes at every level, parameter counts printed, FiLM identity at init, gradient reaches the hypernetwork, ACF with zeroed conditioning equals B2 in structure.

## PHASE 3: LOSSES AND METRICS (src/training, src/metrics)
- Soft Dice, Dice+CE, evidential CE sum_k y_k (digamma(S) - digamma(alpha_k)), annealed KL on alpha_tilde = y + (1-y)*alpha with weight min(1, epoch/anneal_epochs). Unit tests against hand-computed values: for evidence [[0.2,9.0],[0.4,0.6],[12,0.1]] with labels [tumour,tumour,background] the Dice loss is about 0.186, evidential CE mean about 0.333, KL mean about 0.024.
- HD95 in mm using scipy distance transforms and the voxel spacing (no torch.quantile on big tensors; handle empty masks and return NaN-safe values with a documented convention). Surface Dice at 2 and 3 mm. ECE (15 bins) and reliability diagram. Uncertainty-vs-error overlap metric.

## PHASE 4: TRAIN / EVAL (src/training, scripts/)
- Config-driven (YAML). Configs: b1_unet_resample, b2_dual_resample, b3_acf_zeroed, b4_acf_cond, b5_acf_evidential. Add a README table mapping them to the thesis labels.
- AMP, gradient accumulation, AdamW, cosine schedule, deterministic seeds, checkpoint every epoch with resume, early best-checkpoint by val Dice, log git hash + config + environment to every run directory, CSV/JSON logs.
- Sliding-window inference at native spacing, writing masks and uncertainty maps as NIfTI.
- Experiment scripts E1-E6: held-out centre, anisotropy degradation (coarsen through-plane spacing to 2,3,5,7 mm), resource measurement (peak VRAM, preprocessing and inference time), calibration, ablations, statistics (paired Wilcoxon, Bonferroni, effect size, per-centre tables). Output tables as CSV and figures as PNG. Every table row carries the git hash.
- A `scripts/overfit_one_batch.py` and `scripts/smoke_test.py` that must pass on CPU in under 5 minutes.

## PHASE 5: nnU-Net v2 REFERENCE
Scripts to convert the same splits to nnU-Net raw format (CT and PET as channels), plan, train a fold, predict, and evaluate with OUR metric code so the numbers are comparable. Document that nnU-Net resamples to its own median spacing. Provide the exact commands; do not claim results.

## PHASE 6: RUNNERS
Kaggle notebook (.py cells) and Colab notebook: GPU check, CUDA torch check, clone, install, data download via Kaggle secrets, copy to /tmp, 2-epoch timing probe, then full runs. Include the CUDA-build fix (cu124) as an optional cell.

## PHASE 7: FRONTEND
A Streamlit app (app/streamlit_app.py): upload or pick a case, show CT slice with PET heat overlay, predicted tumour outline, uncertainty heat map, scan spacing, and Dice/HD95/ECE when a ground-truth mask is loaded; slice slider and three planes. It must run on synthetic data with no GPU. Add a screenshot script.

## PHASE 8: LEARNING MATERIAL (I am a student who must understand every line)
- Heavy but accurate comments: every tensor shape, every formula, and the reason for each design choice.
- docs/learn/ with: 00_glossary.md; 01_unet_and_nnunet.md; 02_spacing_and_alignment.md; 03_hypernetwork_film.md; 04_dual_encoder_fusion.md; 05_dirichlet_uncertainty.md; 06_losses_and_metrics.md (use the worked 3-voxel example); 07_experiments_and_statistics.md; 08_reading_list.md (the 15 papers with arXiv links: U-Net 1505.04597, 3D U-Net 1606.06650, nnU-Net 1809.10486, nnU-Net Revisited 2404.09556, Iantsen 2102.10446, SwinCross 2302.03861, Saeed 2509.00367, HyperSpace 2407.03681, Consispace 2606.31839, Diaz 2503.21829, HyperNetworks 1609.09106, FiLM 1709.07871, Sensoy 1806.01768, TBraTS 2206.09309, Guo 1706.04599); 09_code_reading_order.md (which file to read first and what to look for) with 5 exercises per chapter and answers in a separate file.
- Two notebooks (notebooks/01_tensor_walkthrough.ipynb, 02_uncertainty_walkthrough.ipynb) that run on synthetic data and print each shape.

## PHASE 9: FINISH
Run the full test suite, flake/ruff, and the smoke test. Produce docs/FINAL_REPORT.md with: what was built, a VERIFIED / NOT VERIFIED table, exact commands to run next on Kaggle, the open risks (HECKTOR not yet available, nnU-Net may win, prior-art search not done), and the pushed commit hash. Show git ls-remote proof.
