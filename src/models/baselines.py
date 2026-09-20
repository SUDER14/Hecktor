"""Baseline 1: plain single-modality 3D U-Net, resampled to isotropic spacing.

This is the reference number every later ACF variant (dual-encoder, FiLM
hypernetwork, evidential head) gets compared against -- see CLAUDE.md's
project checklist. Deliberately plain: MONAI's own UNet, a standard
channel/stride schedule, no spacing conditioning, no FiLM, no PET fusion.
Those are exactly what ACF adds on top of this.

Unlike ACF's own pipeline (src/data/transforms.py's build_train_transforms /
build_val_transforms, which never resample -- see that module's docstring),
this baseline is trained on volumes resampled to a fixed isotropic spacing
(src/data/transforms.py's build_baseline_train_transforms /
build_baseline_val_transforms). A plain UNet has no way to account for
per-scan voxel spacing, so resampling to a common grid is the standard way to
make a fixed-size patch mean the same physical extent across scans -- that's
the whole comparison point.

Outputs raw logits, shape (B, out_channels, *spatial). No final activation:
softmax/argmax happens in the loss (DiceCELoss(softmax=True)) and at
inference time, not in the model.
"""

from __future__ import annotations

from monai.networks.nets import UNet


def build_baseline_unet(config: dict) -> UNet:
    """Builds Baseline 1 from a config's "model" block.

    Expected config["model"] keys:
        in_channels: 1 for single-modality (CT-only Decathlon Task06_Lung).
        out_channels: number of segmentation classes including background
            (2 for Task06_Lung: background/cancer).
        channels: sequence of channel counts per encoder level, top first.
        strides: sequence of strides between levels, len(channels) - 1.
        num_res_units: residual units per block (optional, default 2).
    """
    m = config["model"]
    return UNet(
        spatial_dims=3,
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        channels=tuple(m["channels"]),
        strides=tuple(m["strides"]),
        num_res_units=m.get("num_res_units", 2),
    )
