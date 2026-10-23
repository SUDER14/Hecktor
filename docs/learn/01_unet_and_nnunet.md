# 01 U-Net and nnU-Net

## The idea
A U-Net (Ronneberger et al. 2015, arXiv:1505.04597; 3D version Cicek et al. 2016, arXiv:1606.06650) is an
encoder-decoder. The encoder halves the grid and doubles channels, so deep features see a large region
(context) but are coarse. The decoder upsamples back. **Skip connections** copy the encoder feature map at
each scale to the decoder, restoring fine detail. Output: one score per class per voxel.

## Shapes in our B1 (`src/models/unet3d.py`)
Input patch `(B, 1, 96, 96, 96)`, channels 16-32-64-128-256, four stride-2 steps:

| level | feature shape (B, C, X, Y, Z) | voxel size if input is 1 mm |
|---|---|---|
| L1 | (B, 16, 96, 96, 96) | 1 mm |
| L2 | (B, 32, 48, 48, 48) | 2 mm |
| L3 | (B, 64, 24, 24, 24) | 4 mm |
| L4 | (B, 128, 12, 12, 12) | 8 mm |
| L5 | (B, 256, 6, 6, 6) | 16 mm |

Decoder: up-block j takes the deeper map, upsamples by the stride, concatenates the skip, applies a residual
block. Final 1x1x1 conv -> `(B, K, 96, 96, 96)` logits. Measured parameter count of the configured B1:
5,730,274 (printed by instantiating `configs/b1_unet_resample.yaml`).

## Why the size must divide
Each stride-2 step halves the grid, so X must be divisible by 2^4 = 16 or skip and upsampled maps disagree.
`UNet3D.forward` raises if not.

## Why spacing matters to a plain U-Net
A 3x3x3 kernel covers 3 voxels. At 1 mm that is 3 mm; at 4 mm it is 12 mm. A network trained on one spacing
sees a different physical scale at another. Standard remedy: resample everything to one spacing (B1, B2).
ACF keeps native grids and tells the network the spacing instead (chapters 02, 03).

## nnU-Net
nnU-Net (arXiv:1809.10486) inspects the dataset and configures preprocessing, patch size, batch size and
architecture automatically; "Revisited" (arXiv:2404.09556) argues that a well-configured nnU-Net is still very
hard to beat. It resamples to a **median target spacing** of the training set, then trains a U-Net.
Consequence for us: nnU-Net is a *resampling* method. It may well win; the thesis question is whether native
spacing + conditioning narrows the gap or helps under spacing shift (experiments E1, E2). See
`docs/nnunet_reference.md`.

## Why the loss is Dice + cross-entropy
Tumour voxels are a tiny fraction of the volume. Cross-entropy alone is dominated by background; Dice
measures overlap directly. Sum them (`losses.py:dice_ce_loss`).
