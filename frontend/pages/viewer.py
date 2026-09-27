"""Page 2: tri-planar slice viewer with the ground-truth label overlaid."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from common import VIEWS, data_configs, load_samples, load_split_membership, load_volume, oriented_slice

st.title("Volume viewer")

configs = data_configs()
if not configs:
    st.warning("No `configs/data_*.yaml` found.")
    st.stop()
cfg_name = st.sidebar.selectbox("Dataset config", list(configs))
cfg = configs[cfg_name]

try:
    samples = load_samples(cfg["dataset"], cfg["data_dir"])
except (FileNotFoundError, OSError, ValueError) as e:
    st.info(f"`{cfg['data_dir']}` is not readable yet ({type(e).__name__}: {e}).")
    st.stop()
if not samples:
    st.info(f"No samples found under `{cfg['data_dir']}`.")
    st.stop()

by_id = {s["patient_id"]: s for s in samples}
split_of = load_split_membership(cfg)
pid = st.sidebar.selectbox(
    "Patient",
    sorted(by_id),
    format_func=lambda p: f"{p}  ({split_of[p]})" if p in split_of else p,
)
sample = by_id[pid]
modality = st.sidebar.selectbox("Modality", sample["modalities"])
image_path = sample["image"][sample["modalities"].index(modality)]

vol, spacing = load_volume(image_path)
label, label_spacing = load_volume(sample["label"])

# HECKTOR PET and CT live on different grids, so a label drawn on one may not
# line up voxel-for-voxel with the other. Refuse to overlay rather than show a
# misregistered contour.
overlay_ok = label.shape == vol.shape and np.allclose(label_spacing, spacing, atol=1e-3)

st.sidebar.markdown("---")
if modality == "CT":
    presets = {"Training window (-1000, 400)": (-1000, 400), "Lung (-1350, 150)": (-1350, 150),
               "Mediastinum (-160, 240)": (-160, 240), "Custom": None}
    choice = st.sidebar.selectbox("CT window (HU)", list(presets))
    lo, hi = presets[choice] or st.sidebar.slider("Custom window", -1500, 3000, (-1000, 400))
    cmap = "gray"
else:
    p99 = float(np.percentile(vol, 99.5))
    lo, hi = st.sidebar.slider(f"{modality} display range", 0.0, max(p99 * 2, 1.0), (0.0, max(p99, 1e-3)))
    cmap = "hot"
alpha = st.sidebar.slider("Label opacity", 0.0, 1.0, 0.45, 0.05)
show_label = st.sidebar.checkbox("Show label", value=overlay_ok, disabled=not overlay_ok)

label_values = [int(v) for v in np.unique(label) if v != 0]
fg = np.argwhere(label > 0)
centre = fg.mean(axis=0).round().astype(int) if len(fg) else np.array(vol.shape) // 2

st.caption(
    f"`{Path(image_path).name}` · shape {vol.shape} (RAS) · spacing "
    f"{spacing[0]:.3f} × {spacing[1]:.3f} × {spacing[2]:.3f} mm · "
    f"label voxels {len(fg):,} · classes {label_values or 'none'}"
    + (f" · split: **{split_of[pid]}**" if pid in split_of else "")
)
if not overlay_ok:
    st.warning(
        f"Label grid {label.shape} @ {tuple(round(s, 3) for s in label_spacing)} mm does not match "
        f"this {modality} grid {vol.shape} @ {tuple(round(s, 3) for s in spacing)} mm; overlay disabled."
    )
if not len(fg):
    st.info("This label has no foreground voxels.")

palette = ["#ff3b30", "#34c759", "#0a84ff", "#ffcc00", "#bf5af2"]
label_cmap = ListedColormap(palette)

cols = st.columns(3)
for col, view in zip(cols, VIEWS):
    fixed = VIEWS[view][0]
    n = vol.shape[fixed]
    with col:
        idx = st.slider(f"{view} slice", 0, n - 1, int(centre[fixed]), key=f"{pid}-{modality}-{view}")
        img, aspect, (left, right, top, bottom) = oriented_slice(vol, spacing, view, idx)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, aspect=aspect, interpolation="nearest")
        if show_label:
            lab, _, _ = oriented_slice(label, spacing, view, idx)
            masked = np.ma.masked_where(lab == 0, (lab.astype(int) - 1) % len(palette))
            ax.imshow(masked, cmap=label_cmap, vmin=0, vmax=len(palette) - 1,
                      alpha=alpha, aspect=aspect, interpolation="nearest")
        for x, y, t, ha, va in [(0.01, 0.5, left, "left", "center"), (0.99, 0.5, right, "right", "center"),
                                (0.5, 0.99, top, "center", "top"), (0.5, 0.01, bottom, "center", "bottom")]:
            ax.text(x, y, t, transform=ax.transAxes, ha=ha, va=va, color="yellow", fontsize=11, weight="bold")
        ax.set_title(f"{view} {idx + 1}/{n}", fontsize=10)
        ax.axis("off")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

if show_label and label_values:
    fig, ax = plt.subplots(figsize=(4, 0.4))
    ax.legend(handles=[Patch(color=palette[(v - 1) % len(palette)], label=f"label {v}") for v in label_values],
              loc="center", ncol=len(label_values), frameon=False)
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
