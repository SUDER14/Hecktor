# 00 Glossary

Terms in the order you meet them. Code pointers are `file:function`.

| Term | Meaning here |
|---|---|
| **Voxel** | One cell of a 3D image. Its size in millimetres is the *spacing* (sx, sy, sz). |
| **Spacing / voxel size** | mm per voxel along each axis. Read from the NIfTI header (`header.get_zooms()`) or the diagonal of the affine. HECKTOR CT is around 1 mm, PET around 4 mm (design assumption, to be verified on the real data with `scripts/spacing_manifest.py`). |
| **Affine** | 4x4 matrix mapping voxel index (i,j,k) to world millimetres. For our axis-aligned, RAS volumes it is diagonal (spacing) plus an origin. |
| **RAS** | Axis order/direction convention (x to Right, y to Anterior, z to Superior). `Orientationd("RAS")` reorders arrays so every volume agrees. It never interpolates. |
| **World coordinates** | Millimetres in the scanner frame. The only frame in which CT and PET agree about *where* something is. |
| **Native resolution** | Using each scan on its own grid, with no resampling of the inputs. |
| **Resampling** | Interpolating a volume to a new grid. Used only for baselines B1/B2 and for nnU-Net. |
| **Anisotropy** | Spacing differing between axes, e.g. 1x1x3 mm (thick slices). |
| **Patch** | The sub-volume fed to the network in training. Ours is a *physical box* (e.g. 96 mm) so CT is 96^3 voxels and PET 24^3. |
| **Physical extent** | patch voxels x spacing, in mm. `geometry.py:check_physical_extent` compares CT and PET extents. |
| **Encoder / decoder / skip** | U-Net parts: downsampling path, upsampling path, and feature maps passed across at equal scale. |
| **Level (L1..L5, P1..P3)** | Depth in an encoder. L1 is full resolution; each level halves the grid. |
| **Fusion** | Combining CT and PET features: concatenate channels, then a 1x1x1 convolution. |
| **Instance norm** | Normalises each channel of each sample over space. Needs more than one voxel per axis (why a 24^3 PET patch gets only 3 levels). |
| **Hypernetwork** | A small network whose output *is the parameters* (here modulation parameters) of another network (Ha et al. 2016). |
| **FiLM** | Feature-wise Linear Modulation, F' = gamma * F + beta per channel (Perez et al. 2018). |
| **Conditioning vector** | v = [sx, sy, sz, flag]; spacing z-scored with train-only statistics, flag 0 = CT, 1 = PET. `conditioning.py:to_cond4`. |
| **Tier 1 / Tier 2** | Tier 1 = hypernetwork emits FiLM parameters (default). Tier 2 = emits (low-rank updates of) conv kernels (flag, off). |
| **Logits** | Raw network outputs before softmax. |
| **Softmax** | Turns K logits into probabilities summing to 1. |
| **Evidence e_k** | Non-negative per-class output of the evidential head (softplus of a logit). |
| **Dirichlet** | A distribution over probability vectors, parameterised by alpha (K positive numbers). |
| **alpha, S, p, u** | alpha = e + 1; S = sum alpha; p = alpha / S; u = K / S (vacuity). |
| **Vacuity** | u in (0,1]: 1 means "no evidence at all". Uncertainty from ignorance, not from class overlap. |
| **Dice** | 2 x overlap / (size of prediction + size of reference). 1 = perfect. |
| **HD95** | 95th percentile of surface-to-surface distance, in mm. Less sensitive to a single stray voxel than the maximum. |
| **Surface Dice @ t mm** | Fraction of both surfaces within t mm of the other surface. |
| **ECE** | Expected Calibration Error: gap between confidence and accuracy, averaged over 15 confidence bins. |
| **Reliability diagram** | Accuracy plotted against confidence per bin; the diagonal is perfect calibration. |
| **Held-out centre** | A hospital whose scans are never seen in training; tests generalisation to a new acquisition geometry. |
| **Wilcoxon signed-rank** | Paired non-parametric test on per-case differences. |
| **Bonferroni** | Multiply each p-value by the number of comparisons made. |
| **Rank-biserial r** | Effect size for the Wilcoxon test, in [-1, 1]. |
| **AMP** | Automatic mixed precision (`torch.amp`): float16 where safe, float32 elsewhere. |
| **Gradient accumulation** | Summing gradients of several micro-batches before one optimiser step (effective batch 8 = micro 2 x accum 4). |
| **Checkpoint / resume** | Saved training state so a killed Colab/Kaggle session continues rather than restarts. |
| **nnU-Net** | Self-configuring U-Net framework; the strong reference baseline. It resamples to its own median spacing. |
| **HyperSpace** | Prior art that already conditions a U-Net on spacing with a hypernetwork (MICCAI 2024). |
