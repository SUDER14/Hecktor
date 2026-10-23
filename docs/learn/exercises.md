# Exercises (answers in `exercises_answers.md`)

Try each on paper or in a Python shell before opening the answers.

## Chapter 00 Glossary
1. A CT volume has shape (512, 512, 120) and spacing (0.98, 0.98, 3.0) mm. What is its physical size in mm?
2. Why is "PET voxel (40,40,40)" not the same location as "CT voxel (40,40,40)"?
3. What is the difference between resampling a scan and resizing a feature map?
4. Why does a z-score statistic fitted on train+test leak information?
5. State what u = 1 means for a Dirichlet output, in words.

## Chapter 01 U-Net and nnU-Net
1. Patch (B, 1, 96, 96, 96), four stride-2 steps, channels 16..256: give the shape at L4.
2. Why must 96 (and not e.g. 90) be the patch side for four stride-2 steps?
3. What does a skip connection restore that the bottleneck has lost?
4. A 3x3x3 kernel at L3 of a 1 mm input covers how many millimetres per axis (voxel size at L3 is 4 mm)?
5. Name one way nnU-Net's treatment of spacing differs from ACF's.

## Chapter 02 Spacing and alignment
1. CT 1 mm, PET 4 mm, box 96 mm: voxels per side in each modality?
2. CT 1x1x3 mm, 96 voxels per side: the z-extent in mm, and PET (4 mm) z voxels for that extent?
3. Why is the start index clamped jointly for CT and PET rather than separately?
4. What is the maximum start-position error on the PET grid caused by snapping, in mm?
5. A scan has an oblique affine. What does `grid_from_affine` do, and why?

## Chapter 03 Hypernetwork and FiLM
1. Compute F' for F = 3, gamma = 1.5, beta = -1.
2. Why is the hypernetwork's last layer zero-initialised?
3. With a zero-initialised last layer, does the first-layer weight get gradient at step 0? Explain.
4. What exactly differs between B3 and B4?
5. Why is zeroing both low-rank factors in Tier 2 a bug?

## Chapter 04 Dual-encoder fusion
1. Which CT levels are fused with PET when `pet_offset=2`, three PET levels, five CT levels?
2. Why can a 24^3 PET patch have only about three encoder levels?
3. When CT is 1x1x3 mm and PET 4 mm, why are PET features resized before fusion?
4. What does the per-level extent check protect against?
5. B2 has 9.48 M parameters, B4 6.23 M. Give one consequence for interpreting B2 vs B4.

## Chapter 05 Dirichlet uncertainty
1. evidence (3, 0) for K=2: give alpha, S, p, u.
2. evidence (0, 0): give p and u.
3. Two voxels both have p=(0.5,0.5). How can their u differ and what does it mean?
4. Why average evidence, not u, across overlapping sliding windows?
5. Why is the KL term annealed?

## Chapter 06 Losses and metrics
1. In the worked example, which voxel contributes most to the evidential CE and why?
2. Why is Dice averaged over foreground classes only?
3. What does HD95 return when the reference is empty? When only the prediction is empty?
4. A model is always 90% confident and right 70% of the time: what is its ECE (single bin)?
5. Why not `torch.quantile` for HD95?

## Chapter 07 Experiments and statistics
1. With 6 test patients, what is the smallest possible two-sided Wilcoxon p-value?
2. You run 3 pairwise comparisons; raw p = 0.02. Bonferroni-adjusted p?
3. Why are B3 vs B4 the key ablation for the thesis claim?
4. Why score B1/B2 on the native label rather than their resampled label?
5. Name two reasons a mean Dice can be misleading.
