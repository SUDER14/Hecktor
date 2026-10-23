"""ACF case viewer: CT with PET heat overlay, predicted outline, uncertainty map, spacing, metrics.

    streamlit run app/streamlit_app.py --server.address localhost

Runs with no GPU and no HECKTOR data: "Synthetic demo" generates a small PET/CT cohort (different
spacings per fake centre) and shows a clearly-labelled DEMO heuristic prediction. Point it at a
trained run directory to see real model output, or upload NIfTI files.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app import viewer_core as vc  # noqa: E402
from src.data.synthetic import SynthConfig, write_synthetic_hecktor2026  # noqa: E402

st.set_page_config(page_title="ACF case viewer", layout="wide")
st.title("Acquisition-Conditioned Fusion: case viewer")


@st.cache_resource(show_spinner="Generating synthetic PET/CT cohort...")
def synthetic_root() -> str:
    root = Path(tempfile.mkdtemp(prefix="acf_viewer_")) / "hecktor2026_training"
    write_synthetic_hecktor2026(root, SynthConfig(fov_mm=(64, 64, 64), n_per_centre=2, seed=11))
    return str(root)


def _save_upload(up) -> str | None:
    if up is None:
        return None
    suffix = ".nii.gz" if up.name.endswith(".gz") else ".nii"
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(up.getvalue())
    f.close()
    return f.name


with st.sidebar:
    st.header("Case")
    source = st.radio("Source", ["Synthetic demo", "Folder (HECKTOR 2026 layout)", "Upload NIfTI"])
    run_dir = st.text_input("Trained run directory (optional)", "", help="folder with config.yaml + checkpoints/")
    device = st.selectbox("Device", ["cpu", "cuda"])

case = None
paths = {}
try:
    if source == "Synthetic demo":
        root = synthetic_root()
        pid = st.sidebar.selectbox("Patient", sorted(p.name for p in Path(root).iterdir() if p.is_dir()))
        paths = {"ct": Path(root) / pid / f"{pid}__CT.nii.gz", "pet": Path(root) / pid / f"{pid}__PT.nii.gz"}
        case = vc.load_case_from_dir(root, pid)
    elif source == "Folder (HECKTOR 2026 layout)":
        root = st.sidebar.text_input("Folder", "data/hecktor2026_training")
        ids = sorted(p.name for p in Path(root).iterdir() if p.is_dir()) if Path(root).is_dir() else []
        if ids:
            pid = st.sidebar.selectbox("Patient", ids)
            paths = {"ct": Path(root) / pid / f"{pid}__CT.nii.gz", "pet": Path(root) / pid / f"{pid}__PT.nii.gz"}
            case = vc.load_case_from_dir(root, pid)
        else:
            st.info(f"No patient folders found in {root!r}.")
    else:
        ct_up = st.sidebar.file_uploader("CT (.nii/.nii.gz)", type=["nii", "gz"])
        pet_up = st.sidebar.file_uploader("PET (.nii/.nii.gz)", type=["nii", "gz"])
        lab_up = st.sidebar.file_uploader("Ground-truth mask (optional)", type=["nii", "gz"])
        pr_up = st.sidebar.file_uploader("Prediction mask (optional)", type=["nii", "gz"])
        un_up = st.sidebar.file_uploader("Uncertainty map (optional)", type=["nii", "gz"])
        if ct_up and pet_up:
            paths = {"ct": _save_upload(ct_up), "pet": _save_upload(pet_up)}
            case = vc.load_case(paths["ct"], paths["pet"], _save_upload(lab_up), ct_up.name)
            vc.attach_files(case, _save_upload(pr_up), _save_upload(un_up))
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the case: {e}")

if case is None:
    st.info("Choose a case in the sidebar.")
    st.stop()

if case.pred is None:
    if run_dir.strip():
        with st.spinner("Running the trained model (sliding window)..."):
            try:
                vc.model_prediction_from_paths(case, paths["ct"], paths["pet"], run_dir.strip(), device)
            except Exception as e:  # noqa: BLE001
                st.error(f"Model inference failed: {e}")
    if case.pred is None:
        vc.demo_prediction(case)
for n in case.notes:
    st.warning(n)
st.caption(f"Prediction source: **{case.source}**")

c1, c2, c3 = st.columns(3)
c1.metric("CT spacing (mm)", " x ".join(f"{s:g}" for s in case.ct_spacing))
c2.metric("PET spacing (mm)", " x ".join(f"{s:g}" for s in case.pet_spacing))
c3.metric("CT shape (voxels)", " x ".join(map(str, case.ct.shape)))

with st.sidebar:
    st.header("View")
    plane = st.radio("Plane", list(vc.PLANES), horizontal=True)
    n = vc.axis_length(case, plane)
    idx = st.slider("Slice", 0, n - 1, n // 2)
    show_pet = st.checkbox("PET heat overlay", True)
    pet_alpha = st.slider("PET opacity", 0.0, 1.0, 0.45)
    show_pred = st.checkbox("Predicted outline", True)
    show_unc = st.checkbox("Uncertainty map", True)

st.pyplot(vc.render(case, plane, idx, show_pet, show_pred, show_unc, pet_alpha))

m = vc.case_metrics(case)
if m is None:
    st.caption("Load a ground-truth mask to see Dice / HD95 / ECE.")
else:
    st.subheader("Metrics vs ground truth")
    cols = st.columns(len([k for k in m if k != "note"]))
    for col, (k, v) in zip(cols, [(k, v) for k, v in m.items() if k != "note"]):
        col.metric(k, f"{v:.4f}")
    if "note" in m:
        st.caption(m["note"])
    if case.source == "demo":
        st.caption("These numbers describe the DEMO heuristic on synthetic data, not a model result.")
