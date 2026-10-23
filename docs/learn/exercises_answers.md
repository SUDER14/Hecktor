# Answers

## Chapter 00
1. 512 x 0.98 = 501.8 mm, same in y, 120 x 3.0 = 360 mm: about 501.8 x 501.8 x 360 mm.
2. Different spacings: CT (40) is at about 40 mm, PET (40) at about 160 mm (plus each volume's own origin).
3. Resampling interpolates the *input scan* to a new grid before the network. Resizing a feature map interpolates
   learned activations inside the network, after the encoder saw the native data.
4. The mean/std would then encode test-set properties; the model would be normalised using information it should
   not have, which makes held-out results optimistic.
5. u = K/S = 1 means S = K, i.e. all alpha = 1: no evidence at all, a uniform "I don't know".

## Chapter 01
1. L4 is three halvings: 96/8 = 12, channels 128: (B, 128, 12, 12, 12).
2. 96 is divisible by 16 (96/16 = 6); 90/16 is not an integer, so upsampled and skip maps would not match in size.
3. Fine spatial detail (edges, small structures) that downsampling discarded.
4. Voxel 4 mm x 3 = 12 mm.
5. nnU-Net resamples all inputs to its median target spacing (and needs PET on the CT grid); ACF keeps native grids
   and passes the spacing to the network.

## Chapter 02
1. CT 96, PET 24.
2. z extent 96 x 3 = 288 mm; PET: 288 / 4 = 72 voxels.
3. Separate clamping could shift one window relative to the other, destroying the common physical box.
4. Half a PET voxel = 2 mm per modality's snapping (a PET-vs-CT start difference of up to about 2.5 mm).
5. It raises a ValueError: the crop logic assumes diagonal-positive affines; handling oblique volumes would need
   real resampling, which this pipeline does not do.

## Chapter 03
1. 1.5 x 3 - 1 = 3.5.
2. So gamma = 1, beta = 0 at init: the model starts unconditioned and conditioning must be learned.
3. No. d loss/d(first layer) passes through the last layer's weights, which are zero; only the last layer gets
   gradient at step 0. After its first update the earlier layers receive gradient.
4. Only the input to the hypernetworks: B3 gets zeros, B4 the real [sx, sy, sz, flag]. Parameter count is identical.
5. delta = A C with both zero: d delta/dA = C = 0 and d delta/dC = A = 0, so neither factor ever gets gradient.

## Chapter 04
1. L3, L4, L5 (0-based levels 2, 3, 4).
2. Each halving 24 -> 12 -> 6 -> 3; instance norm needs more than one voxel per axis, and deeper levels would reach 1.
3. The CT feature grid is anisotropic (voxels 4,4,12 mm at L3) while PET's is (4,4,4); shapes differ, so PET features
   are interpolated onto the CT grid, after the physical-extent check.
4. Fusing two feature maps that cover different physical regions (misaligned crops or wrong spacings).
5. B2 has more capacity (full-depth PET encoder at 1 mm), so a difference is not purely due to the conditioning or
   native-spacing design; compare with that confound stated.

## Chapter 05
1. alpha = (4, 1), S = 5, p = (0.8, 0.2), u = 2/5 = 0.4.
2. alpha = (1, 1), S = 2: p = (0.5, 0.5), u = 1.
3. u = K/S: one voxel may have S = 2 (no evidence, u = 1), the other S = 20 (u = 0.1, strong evidence on both
   classes): ignorance versus genuine ambiguity.
4. A Dirichlet is defined by alpha (evidence); averaging evidence gives a valid Dirichlet. Averaging u values does
   not correspond to any single distribution.
5. Early in training the network has no useful evidence; a strong KL would push everything to u = 1 and prevent
   learning, so it is switched on gradually.

## Chapter 06
1. Voxel 2 (evidence 0.4, 0.6; true class tumour). Per-voxel terms are about 0.119 (v1), 0.797 (v2), 0.084 (v3),
   mean 0.333. Voxel 2 has little evidence for its true class (alpha 1.6 out of S = 3), the others have a lot.
2. A tumour-free crop is dominated by background; including background would reward predicting "all background".
3. Empty reference: NaN (undefined, excluded). Empty prediction with non-empty reference: the volume's physical
   diagonal penalty with `pred_empty = True`.
4. |0.7 - 0.9| = 0.2.
5. It fails ("input tensor is too large") on full-resolution volumes; the scipy/numpy implementation avoids it.

## Chapter 07
1. 2 / 2^6 = 0.03125 (all six differences the same sign).
2. min(1, 0.02 x 3) = 0.06.
3. Same architecture and parameters; the only change is whether the acquisition vector carries information, so a
   difference isolates the value of conditioning.
4. Otherwise each model is scored on a different grid and label (resampled versus native), so numbers are not
   comparable.
5. Excluded empty-label cases hidden; a few cases with Dice 0 versus many near 1 giving the same mean; no
   confidence intervals; centre imbalance; single seed.
