"""Page 1: spacing/anisotropy summary from spacing_analysis.py's outputs."""

from __future__ import annotations

import streamlit as st

from common import REPO_ROOT, data_configs, load_split_membership, read_csv, spacing_result_dirs

st.title("Dataset overview")

dirs = spacing_result_dirs()
if not dirs:
    st.warning(
        "No `results/*/spacing_manifest.csv` found. Run `src/metadata/spacing_analysis.py` first; "
        "this dashboard only reads its outputs."
    )
    st.stop()

result_dir = st.sidebar.selectbox(
    "Spacing analysis output", dirs, format_func=lambda p: str(p.relative_to(REPO_ROOT))
)
manifest = read_csv(str(result_dir / "spacing_manifest.csv"))
cond_path = result_dir / "conditioning_vectors.csv"
conditioning = read_csv(str(cond_path)) if cond_path.exists() else None

# Split membership comes from the committed split file of whichever data
# config points at the same patients; purely informational.
split_of: dict[str, str] = {}
for cfg in data_configs().values():
    membership = load_split_membership(cfg)
    if membership and set(membership) & set(manifest["patient_id"]):
        split_of = membership
        break
# The manifest lists every NIfTI under the dataset folder: label files (which
# share their image's spacing) and unlabelled test images (imagesTs) included.
# Headline numbers use image rows only, as spacing_analysis.py's figures do.
manifest = manifest.assign(
    folder=manifest["relpath"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-2]
)
if split_of:
    manifest = manifest.assign(split=manifest["patient_id"].map(split_of).fillna("-"))
images = manifest[manifest["modality"] != "LABEL"]

c = st.columns(5)
c[0].metric("Image volumes", len(images), help=", ".join(f"{k}: {v}" for k, v in images["folder"].value_counts().items()))
c[1].metric("Centres", images["centre"].nunique())
c[2].metric("Modalities", ", ".join(sorted(images["modality"].astype(str).unique())))
c[3].metric(
    "In-plane spacing (mm)",
    f"{images['in_plane_spacing'].min():.2f} to {images['in_plane_spacing'].max():.2f}",
)
c[4].metric(
    "Slice thickness (mm)",
    f"{images['slice_thickness'].min():.2f} to {images['slice_thickness'].max():.2f}",
)
st.caption(
    "Rows per folder: " + ", ".join(f"`{k}` {v}" for k, v in manifest["folder"].value_counts().sort_index().items())
    + ". Label rows are hidden by default below; they repeat their image's spacing."
)

st.subheader("Figures")
figs = sorted(result_dir.glob("*.png"))
if figs:
    cols = st.columns(min(len(figs), 2))
    for i, fig in enumerate(figs):
        cols[i % len(cols)].image(str(fig), caption=fig.name, width="stretch")
else:
    st.info("No figures in this output folder.")

st.subheader("Spacing manifest")
st.caption("Click a column header to sort. Filters below narrow the rows shown.")
f1, f2, f3, f4 = st.columns(4)
centres = f1.multiselect("Centre", sorted(manifest["centre"].astype(str).unique()))
all_mods = sorted(manifest["modality"].astype(str).unique())
modalities = f2.multiselect("Modality", all_mods, default=[m for m in all_mods if m != "LABEL"])
st_lo, st_hi = float(manifest["slice_thickness"].min()), float(manifest["slice_thickness"].max())
thick = f3.slider("Slice thickness (mm)", st_lo, max(st_hi, st_lo + 0.01), (st_lo, max(st_hi, st_lo + 0.01)))
an_lo, an_hi = float(manifest["anisotropy_ratio"].min()), float(manifest["anisotropy_ratio"].max())
aniso = f4.slider("Anisotropy ratio", an_lo, max(an_hi, an_lo + 0.01), (an_lo, max(an_hi, an_lo + 0.01)))
pid_query = st.text_input("Patient ID contains", "")

view = manifest
if centres:
    view = view[view["centre"].astype(str).isin(centres)]
if modalities:
    view = view[view["modality"].astype(str).isin(modalities)]
view = view[view["slice_thickness"].between(*thick) & view["anisotropy_ratio"].between(*aniso)]
if pid_query:
    view = view[view["patient_id"].astype(str).str.contains(pid_query, case=False, regex=False)]
st.write(f"{len(view)} of {len(manifest)} rows")
st.dataframe(view, width="stretch", hide_index=True)

if conditioning is not None:
    st.subheader("Conditioning vectors")
    st.caption("Spacing normalised with train-split statistics (fit on train only, reused at val/test).")
    st.dataframe(conditioning, width="stretch", hide_index=True)
