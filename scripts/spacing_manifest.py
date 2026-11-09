"""One CSV of header geometry for every scan, plus per-centre spacing histograms.

Why: the whole premise of acquisition conditioning is that spacing varies between
scans and centres; this is the measured evidence for (or against) that, from
headers only (no voxel data is read).

    python scripts/spacing_manifest.py --data_dir <hecktor2026_training or the .zip> --out results/manifest

Zip input is read in memory, header bytes only (see src/data/zip_reader.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.hecktor2026 import Hecktor2026Adapter  # noqa: E402
from src.data.zip_reader import load_header  # noqa: E402

OBLIQUE_TOL = 1e-3


def is_oblique(affine: np.ndarray, tol: float = OBLIQUE_TOL) -> bool:
    """True if any voxel axis has an off-principal-axis component above `tol` of its length.

    Oblique grids cannot be handled by a pure axis flip/permute reorientation.
    """
    lin = np.asarray(affine, float)[:3, :3]
    norms = np.linalg.norm(lin, axis=0)
    dom = np.abs(lin).max(axis=0)
    return bool(((norms - dom) / np.maximum(norms, 1e-12) > tol).any())


def build_manifest(data_dir: str | Path) -> pd.DataFrame:
    rows = []
    for s in Hecktor2026Adapter(data_dir).training_samples():
        for mod, path in zip(s["modalities"] + ["LABEL"], s["image"] + [s["label"]]):
            h = load_header(path)
            aff = h.get_best_affine()
            sx, sy, sz = (float(z) for z in h.get_zooms()[:3])
            shape = h.get_data_shape()[:3]
            rows.append({"patient_id": s["patient_id"], "centre": s["centre"], "modality": mod,
                         "sx": sx, "sy": sy, "sz": sz, "nx": shape[0], "ny": shape[1], "nz": shape[2],
                         "anisotropy": max(sx, sy, sz) / min(sx, sy, sz),
                         "fov_x_mm": sx * shape[0], "fov_y_mm": sy * shape[1], "fov_z_mm": sz * shape[2],
                         "orientation": "".join(nib.aff2axcodes(aff)), "oblique": is_oblique(aff),
                         "dtype": str(h.get_data_dtype()),
                         "affine": " ".join(f"{v:.6g}" for v in aff[:3].ravel())})
    return pd.DataFrame(rows)


def histograms(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    img = df[df.modality != "LABEL"]
    top = max(float(img[["sx", "sy", "sz"]].max().max()), 1.0) * 1.05
    for centre, sub in img.groupby("centre"):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
        for ax, col in zip(axes, ("sx", "sy", "sz")):
            for mod, m in sub.groupby("modality"):
                ax.hist(m[col], bins=np.linspace(0, top, 24), alpha=0.6, label=mod)
            ax.set_xlabel(f"{col} (mm)")
        axes[0].set_ylabel("scans")
        axes[0].legend()
        fig.suptitle(f"voxel spacing, centre {centre} (n={sub.patient_id.nunique()})")
        fig.tight_layout()
        p = out_dir / f"spacing_hist_{centre}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="results/manifest")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = build_manifest(a.data_dir)
    df.to_csv(out / "spacing_manifest.csv", index=False)
    figs = histograms(df, out)
    print(f"{len(df)} rows -> {out / 'spacing_manifest.csv'}; {len(figs)} histograms")
    print(df[df.modality != "LABEL"].groupby(["centre", "modality"])[["sx", "sy", "sz"]].agg(["min", "max"]).round(2))


if __name__ == "__main__":
    main()
