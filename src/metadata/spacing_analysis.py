"""
spacing_analysis.py
===================
Week 2 deliverable for the ACF / HECKTOR project.

Walks a directory of NIfTI volumes, reads the physical voxel spacing out of each
header, writes a manifest CSV, builds the normalised conditioning vectors that
the hypernetwork will consume, and produces the spacing-heterogeneity figures.

Works on any NIfTI dataset. Develop against Medical Segmentation Decathlon while
waiting for HECKTOR approval, then point --data-dir at HECKTOR and rerun.

Usage
-----
    python spacing_analysis.py --data-dir ./data/Task01_BrainTumour
    python spacing_analysis.py --data-dir ./data/hecktor2025 --out ./results/hecktor

Outputs (in --out)
------------------
    spacing_manifest.csv      one row per volume
    conditioning_vectors.csv  normalised vectors, ready for the model
    norm_stats.json           normalisation statistics  <-- REUSE AT TEST TIME
    fig1_spacing_distribution.png
    fig2_anisotropy_scatter.png
    fig3_centre_breakdown.png (only if centres are detected)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Filename conventions
# ----------------------------------------------------------------------------
# HECKTOR: CHUM-001__CT.nii.gz / CHUM-001__PT.nii.gz / CHUM-001.nii.gz (label)
# Centre prefixes seen across editions: CHUM CHUS CHUP CHUV HGJ HMR MDA USZ ...
CENTRE_RE = re.compile(r"^([A-Za-z]{2,6})[-_]\d+")
MODALITY_RE = re.compile(r"__(CT|PT|PET|MR)\b", re.IGNORECASE)

LABEL_HINTS = ("label", "seg", "gtv", "mask")


def infer_modality(path: Path) -> str:
    """CT / PT / LABEL / UNKNOWN, from the filename."""
    m = MODALITY_RE.search(path.name)
    if m:
        tag = m.group(1).upper()
        return "PT" if tag in ("PT", "PET") else tag
    low = path.name.lower()
    if any(h in low for h in LABEL_HINTS):
        return "LABEL"
    # Decathlon puts images and labels in sibling folders
    parent = path.parent.name.lower()
    if parent.startswith("labels"):
        return "LABEL"
    if parent.startswith("images"):
        return "IMAGE"
    return "UNKNOWN"


def infer_centre(path: Path) -> str:
    m = CENTRE_RE.match(path.name)
    return m.group(1).upper() if m else "UNKNOWN"


def infer_patient(path: Path) -> str:
    stem = path.name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.split("__")[0]


# ----------------------------------------------------------------------------
# Header reading
# ----------------------------------------------------------------------------
def read_volume_geometry(path: Path) -> dict | None:
    """
    Physical geometry from a NIfTI header.

    Spacing is taken from header.get_zooms() rather than the raw affine, because
    zooms already account for qform/sform scaling. Index 2 is the through-plane
    axis for standard axial acquisitions -- verify orientation before trusting
    that on a new dataset.
    """
    import nibabel as nib  # imported here so --help works without it

    try:
        img = nib.load(str(path))
    except Exception as exc:  # corrupt file, wrong extension, truncated download
        print(f"  ! could not read {path.name}: {exc}", file=sys.stderr)
        return None

    zooms = img.header.get_zooms()[:3]
    sx, sy, sz = (float(z) for z in zooms)
    shape = img.shape[:3]

    if min(sx, sy, sz) <= 0:
        print(f"  ! non-positive spacing in {path.name}: {zooms}", file=sys.stderr)
        return None

    in_plane = float(np.sqrt(sx * sy))  # geometric mean, robust to sx != sy

    return {
        "patient_id": infer_patient(path),
        "centre": infer_centre(path),
        "modality": infer_modality(path),
        "filename": path.name,
        "relpath": str(path),
        "spacing_x": round(sx, 4),
        "spacing_y": round(sy, 4),
        "slice_thickness": round(sz, 4),
        "in_plane_spacing": round(in_plane, 4),
        "anisotropy_ratio": round(sz / in_plane, 4),
        "shape_x": shape[0],
        "shape_y": shape[1],
        "shape_z": shape[2],
        "fov_x_mm": round(shape[0] * sx, 1),
        "fov_y_mm": round(shape[1] * sy, 1),
        "fov_z_mm": round(shape[2] * sz, 1),
        "n_voxels": int(np.prod(shape)),
    }


def build_manifest(data_dir: Path, include_labels: bool) -> pd.DataFrame:
    files = sorted(
        list(data_dir.rglob("*.nii.gz")) + list(data_dir.rglob("*.nii"))
    )
    files = [f for f in files if not f.name.startswith(".")]  # skip ._ resource forks

    if not files:
        raise SystemExit(
            f"No NIfTI files found under {data_dir}\n"
            "Check the path, and that the archive was actually extracted."
        )

    print(f"Found {len(files)} NIfTI files under {data_dir}")
    rows = []
    for i, f in enumerate(files, 1):
        if i % 50 == 0 or i == len(files):
            print(f"  read {i}/{len(files)}")
        rec = read_volume_geometry(f)
        if rec is not None:
            rows.append(rec)

    df = pd.DataFrame(rows)
    if not include_labels:
        df = df[df["modality"] != "LABEL"].reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Conditioning vectors
# ----------------------------------------------------------------------------
COND_FEATURES = ["spacing_x", "spacing_y", "slice_thickness"]


def fit_norm_stats(df: pd.DataFrame) -> dict:
    """
    Per-modality z-score statistics.

    Per-modality matters: PET spacing (~4mm) and CT spacing (~1mm) occupy
    different ranges, and pooling them would compress the within-modality
    variation the network is supposed to be sensitive to.
    """
    stats = {}
    for mod, sub in df.groupby("modality"):
        stats[mod] = {
            feat: {
                "mean": float(sub[feat].mean()),
                "std": float(sub[feat].std(ddof=0)) or 1.0,  # guard constant columns
            }
            for feat in COND_FEATURES
        }
    return stats


def apply_norm_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Build v = [sx_norm, sy_norm, tz_norm, is_CT, is_PT] per volume."""
    out = []
    for _, r in df.iterrows():
        mod = r["modality"]
        s = stats.get(mod)
        if s is None:  # modality unseen at fit time
            continue
        vec = {
            "patient_id": r["patient_id"],
            "centre": r["centre"],
            "modality": mod,
        }
        for feat in COND_FEATURES:
            vec[f"{feat}_norm"] = round(
                (r[feat] - s[feat]["mean"]) / s[feat]["std"], 5
            )
        vec["is_CT"] = int(mod == "CT")
        vec["is_PT"] = int(mod == "PT")
        out.append(vec)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
NAVY, ORANGE, GREEN = "#1F3864", "#C55A11", "#2E7D32"
MOD_COLOURS = {"CT": NAVY, "PT": ORANGE, "IMAGE": GREEN, "UNKNOWN": "#777777"}


def fig_spacing_distribution(df: pd.DataFrame, out: Path):
    mods = [m for m in df["modality"].unique() if m != "LABEL"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, col, title in zip(
        axes,
        ["in_plane_spacing", "slice_thickness"],
        ["In-plane spacing", "Slice thickness (through-plane)"],
    ):
        for mod in mods:
            vals = df.loc[df["modality"] == mod, col]
            ax.hist(vals, bins=25, alpha=0.65, label=f"{mod} (n={len(vals)})",
                    color=MOD_COLOURS.get(mod, "#777777"), edgecolor="white", lw=0.5)
        ax.set_xlabel(f"{title}  (mm)")
        ax.set_ylabel("Number of volumes")
        ax.set_title(title, fontsize=11, color=NAVY)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, ls=":")

    fig.suptitle("Acquisition geometry heterogeneity across the cohort",
                 fontsize=12.5, color=NAVY, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig1_spacing_distribution.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_anisotropy(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for mod in [m for m in df["modality"].unique() if m != "LABEL"]:
        sub = df[df["modality"] == mod]
        ax.scatter(sub["in_plane_spacing"], sub["slice_thickness"], s=26,
                   alpha=0.6, label=f"{mod} (n={len(sub)})",
                   color=MOD_COLOURS.get(mod, "#777777"), edgecolors="none")

    lim = max(df["in_plane_spacing"].max(), df["slice_thickness"].max()) * 1.1
    ax.plot([0, lim], [0, lim], ls="--", lw=1, color="#999999")
    ax.text(lim * 0.62, lim * 0.66, "isotropic", fontsize=9, color="#777777",
            rotation=38)

    ax.set_xlabel("In-plane spacing (mm)")
    ax.set_ylabel("Slice thickness (mm)")
    ax.set_title("Anisotropy: points above the line are anisotropic volumes",
                 fontsize=11, color=NAVY)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(out / "fig2_anisotropy_scatter.png", dpi=200, facecolor="white")
    plt.close(fig)


def fig_centre_breakdown(df: pd.DataFrame, out: Path):
    centres = sorted(c for c in df["centre"].unique() if c != "UNKNOWN")
    if len(centres) < 2:
        print("  (skipping centre figure - no centre codes found in filenames)")
        return

    fig, ax = plt.subplots(figsize=(max(7, len(centres) * 1.5), 5))
    data, labels, colours = [], [], []
    for centre in centres:
        for mod in ["CT", "PT"]:
            vals = df[(df["centre"] == centre) & (df["modality"] == mod)]["slice_thickness"]
            if len(vals):
                data.append(vals.values)
                labels.append(f"{centre}\n{mod}")
                colours.append(MOD_COLOURS[mod])

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    for med in bp["medians"]:
        med.set_color("#222222")

    ax.set_ylabel("Slice thickness (mm)")
    ax.set_title("Slice thickness by contributing centre", fontsize=11, color=NAVY)
    ax.grid(alpha=0.25, ls=":", axis="y")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_centre_breakdown.png", dpi=200, facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------
def summarise(df: pd.DataFrame):
    print("\n" + "=" * 64)
    print("SPACING SUMMARY")
    print("=" * 64)
    for mod, sub in df.groupby("modality"):
        if mod == "LABEL":
            continue
        print(f"\n{mod}  (n = {len(sub)})")
        for col, name in [("in_plane_spacing", "in-plane"),
                          ("slice_thickness", "thickness"),
                          ("anisotropy_ratio", "aniso")]:
            print(f"  {name:<10} min {sub[col].min():6.2f}   "
                  f"median {sub[col].median():6.2f}   max {sub[col].max():6.2f}")
        iso = (sub['anisotropy_ratio'].between(0.95, 1.05)).mean() * 100
        print(f"  isotropic volumes: {iso:.0f}%")

    centres = sorted(c for c in df["centre"].unique() if c != "UNKNOWN")
    if len(centres) >= 2:
        print(f"\nCentres detected: {', '.join(centres)}")
    print("=" * 64 + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("./results/spacing"))
    ap.add_argument("--include-labels", action="store_true",
                    help="also profile label volumes (normally excluded)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = build_manifest(args.data_dir, args.include_labels)
    df.to_csv(args.out / "spacing_manifest.csv", index=False)
    print(f"\nWrote {len(df)} rows -> {args.out / 'spacing_manifest.csv'}")

    summarise(df)

    stats = fit_norm_stats(df)
    with open(args.out / "norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    cond = apply_norm_stats(df, stats)
    cond.to_csv(args.out / "conditioning_vectors.csv", index=False)
    print(f"Wrote {len(cond)} conditioning vectors -> conditioning_vectors.csv")
    print("NOTE: norm_stats.json must be reused unchanged at validation and test "
          "time. Refitting on test data leaks information.")

    fig_spacing_distribution(df, args.out)
    fig_anisotropy(df, args.out)
    fig_centre_breakdown(df, args.out)
    print(f"\nFigures written to {args.out}/")


if __name__ == "__main__":
    main()
