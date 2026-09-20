"""Pulls N samples through the real train-time pipeline and writes PNG
overlays (image + label) so a human can eyeball them before training anything.

Usage:
    python -m src.data.sanity_check --config configs/data_decathlon_lung.yaml --n 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.dataset import build_monai_datasets, load_config
from src.data.seed import seed_everything


def _to_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _slice_with_most_label(label: np.ndarray) -> int:
    """Through-plane index with the largest foreground area in this patch,
    falling back to the middle slice if the patch has no foreground at all
    (expected sometimes -- that's what the neg side of pos/neg sampling is
    for)."""
    per_slice = label[0].sum(axis=(0, 1))
    if per_slice.max() == 0:
        return label.shape[-1] // 2
    return int(per_slice.argmax())


def _save_overlay(image: np.ndarray, label: np.ndarray, out_path: Path, title: str) -> None:
    z = _slice_with_most_label(label)
    img_slice = image[0, :, :, z]
    lbl_slice = label[0, :, :, z]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_slice.T, cmap="gray", origin="lower")
    masked = np.ma.masked_where(lbl_slice.T == 0, lbl_slice.T)
    ax.imshow(masked, cmap="autumn", alpha=0.5, origin="lower")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("results/sanity"))
    args = ap.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"])

    datasets = build_monai_datasets(config)
    ds = datasets[args.split]
    if len(ds) == 0:
        raise SystemExit(f"split {args.split!r} is empty -- check the split file / config")

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"\nPulling up to {args.n} sample(s) from split={args.split!r} "
          f"({len(ds)} volumes available)\n" + "=" * 64)

    collected = 0
    idx = 0
    while collected < args.n and idx < len(ds):
        item = ds[idx]
        # RandCropByPosNegLabeld yields num_samples crops per volume as a list.
        crops = item if isinstance(item, list) else [item]
        for crop in crops:
            if collected >= args.n:
                break

            image = _to_numpy(crop["image"])
            label = _to_numpy(crop["label"])
            cond = np.round(_to_numpy(crop["cond_vector"]), 4).tolist()
            geom = crop["geometry"]
            n_fg = int((label > 0).sum())

            print(f"[{collected}] patient={crop['patient_id']}  "
                  f"modality={crop['modality']}  centre={crop['centre']}")
            print(f"    spacing (mm) : x={geom['spacing_x']:.4f} "
                  f"y={geom['spacing_y']:.4f} z={geom['slice_thickness']:.4f}")
            print(f"    patch shape  : {tuple(image.shape)}")
            print(f"    intensity    : min={image.min():.4f} max={image.max():.4f}")
            print(f"    conditioning : {cond}")
            print(f"    foreground   : {n_fg} / {label.size} voxels")

            out_path = args.out / f"{args.split}_{collected:02d}_{crop['patient_id']}.png"
            _save_overlay(image, label, out_path,
                          f"{crop['patient_id']} ({crop['modality']})  fg={n_fg}vox")
            print(f"    overlay      -> {out_path}")

            collected += 1
        idx += 1

    print("=" * 64)
    if collected < args.n:
        print(f"! only found {collected}/{args.n} crops (ran out of volumes in "
              f"split {args.split!r})")
    print(f"\nWrote {collected} overlays to {args.out}/")


if __name__ == "__main__":
    main()
