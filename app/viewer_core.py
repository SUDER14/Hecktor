"""Rendering and loading logic for the Streamlit viewer, kept free of streamlit so it is testable.

Everything is displayed on the CT grid (RAS). PET, which lives on its own coarser grid, is
resampled onto it FOR DISPLAY ONLY (the model never sees that resampled copy).

Prediction sources, in order of honesty:
  * "model":  a trained run directory (config.yaml + checkpoints/) run on CPU/GPU via sliding window.
  * "files":  prediction / uncertainty NIfTI files the user supplies (e.g. from scripts or nnU-Net).
  * "demo":   a PET-threshold HEURISTIC with a pseudo-uncertainty. It is NOT a trained model and is
              labelled as such in the UI; it exists so the viewer can be exercised with no checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from src.eval.infer import resample_to_grid  # noqa: E402
from src.metrics.calibration import ece  # noqa: E402
from src.metrics.segmentation import dice, hd95, surface_dice  # noqa: E402

PLANES = {"axial": 2, "coronal": 1, "sagittal": 0}  # plane name -> axis that the slider walks


@dataclass
class Case:
    patient_id: str
    ct: np.ndarray                    # (X,Y,Z) HU
    pet: np.ndarray                   # (X,Y,Z) PET resampled onto the CT grid (display only)
    ct_spacing: tuple
    pet_spacing: tuple
    label: np.ndarray | None = None   # (X,Y,Z) uint8
    pred: np.ndarray | None = None    # (X,Y,Z) uint8
    unc: np.ndarray | None = None     # (X,Y,Z) float in [0,1]
    prob: np.ndarray | None = None    # (K,X,Y,Z) for ECE
    source: str = "none"
    notes: list = field(default_factory=list)


def _canon(img: nib.Nifti1Image):
    c = nib.as_closest_canonical(img)
    return np.asarray(c.dataobj, dtype=np.float32), c.affine


def load_case(ct_path, pet_path, label_path=None, patient_id: str = "case") -> Case:
    ct, ct_aff = _canon(nib.load(str(ct_path)))
    pet_raw, pet_aff = _canon(nib.load(str(pet_path)))
    pet = resample_to_grid(pet_raw, pet_aff, ct_aff, ct.shape, order=1)
    label = None
    if label_path is not None:
        label = (_canon(nib.load(str(label_path)))[0] > 0).astype(np.uint8)
    sp = lambda a: tuple(float(abs(x)) for x in np.diag(a)[:3])  # noqa: E731
    return Case(patient_id, ct, pet, sp(ct_aff), sp(pet_aff), label)


def load_case_from_dir(root, patient_id: str) -> Case:
    d = Path(root) / patient_id
    lab = d / f"{patient_id}.nii.gz"
    return load_case(d / f"{patient_id}__CT.nii.gz", d / f"{patient_id}__PT.nii.gz", lab if lab.exists() else None, patient_id)


def attach_files(case: Case, pred_path=None, unc_path=None) -> Case:
    if pred_path is not None:
        case.pred = (_canon(nib.load(str(pred_path)))[0] > 0).astype(np.uint8)
        case.source = "files"
    if unc_path is not None:
        case.unc = _canon(nib.load(str(unc_path)))[0]
    return case


def demo_prediction(case: Case) -> Case:
    """PET-threshold heuristic + distance-to-threshold pseudo-uncertainty. NOT a trained model."""
    pet = case.pet
    thr = 0.5 * float(np.percentile(pet, 99.9)) if pet.max() > 0 else 1.0
    pred = (pet > thr)
    pred = ndimage.binary_opening(pred, iterations=1)
    margin = np.abs(pet - thr) / max(thr, 1e-6)
    case.pred = pred.astype(np.uint8)
    case.unc = np.clip(1.0 - margin, 0, 1).astype(np.float32)
    case.source = "demo"
    case.notes.append("DEMO heuristic (PET > 50% of near-max), not a trained model; uncertainty is a stand-in.")
    return case


def model_prediction_from_paths(case: Case, ct_path, pet_path, run_dir, device: str = "cpu") -> Case:
    """Run a trained run directory on this case, re-reading CT/PET on their own native grids."""
    import torch

    from src.data.conditioning import load_stats, make_cond4
    from src.eval.evaluate_cases import call_model
    from src.eval.infer import sliding_window_dual
    from src.eval.experiments import load_run

    cfg, model, _ = load_run(run_dir, device=device)
    stats = load_stats(cfg["conditioning"]["stats_path"])
    ct, ct_aff = _canon(nib.load(str(ct_path)))
    pet, pet_aff = _canon(nib.load(str(pet_path)))
    w, suv = cfg["intensity"]["ct_window"], cfg["intensity"]["pet"]["suv_clip_max"]
    ctn = np.clip((ct - w["min"]) / (w["max"] - w["min"]), 0, 1)
    petn = np.clip(pet / suv, 0, 1)
    csp = tuple(float(abs(x)) for x in np.diag(ct_aff)[:3])
    psp = tuple(float(abs(x)) for x in np.diag(pet_aff)[:3])
    cc = torch.from_numpy(make_cond4(csp, "CT", stats))[None].to(device)
    cp = torch.from_numpy(make_cond4(psp, "PT", stats))[None].to(device)
    cs, ps = torch.tensor(csp)[None].to(device), torch.tensor(psp)[None].to(device)
    head = getattr(model, "head_type", "softmax")

    def fwd(c, p):
        b = c.shape[0]
        return call_model(model, c, p, cc.expand(b, -1), cp.expand(b, -1), cs.expand(b, -1), ps.expand(b, -1))

    res = sliding_window_dual(fwd, ctn, petn, ct_aff, pet_aff, tuple(cfg["patch"]["spatial_size"]),
                              cfg["model"].get("out_channels", 2), head, cfg.get("infer", {}).get("overlap", 0.5), 1, device)
    case.pred, case.unc, case.prob, case.source = res["pred"], res["u"][0], res["p"], "model"
    return case


def case_metrics(case: Case) -> dict | None:
    """Dice / HD95 / Surface Dice (2 mm) / ECE when a ground truth AND a prediction exist."""
    if case.label is None or case.pred is None:
        return None
    sp = case.ct_spacing
    h = hd95(case.pred > 0, case.label > 0, sp)
    out = {"Dice": dice(case.pred > 0, case.label > 0), "HD95 (mm)": h.value,
           "Surface Dice @2mm": surface_dice(case.pred > 0, case.label > 0, sp, 2.0)}
    if h.pred_empty:
        out["note"] = "prediction empty: HD95 is the volume-diagonal penalty"
    if case.prob is not None:
        out["ECE (15 bins)"] = ece(case.prob, case.label.astype(int))
    return out


def _slice(vol: np.ndarray, plane: str, idx: int) -> np.ndarray:
    ax = PLANES[plane]
    s = np.take(vol, idx, axis=ax)
    return s.T[::-1]  # display with the first remaining axis horizontal, last axis up


def _aspect(spacing, plane: str) -> float:
    ax = PLANES[plane]
    h_ax, v_ax = [a for a in (0, 1, 2) if a != ax]  # horizontal, vertical axes (after the transpose in _slice)
    return spacing[v_ax] / spacing[h_ax]


def render(case: Case, plane: str, idx: int, show_pet=True, show_pred=True, show_unc=True,
           pet_alpha: float = 0.45, window=(-160, 240)) -> plt.Figure:
    """Two panels: CT + PET heat + predicted outline (+ GT outline), and the uncertainty map."""
    panels = 2 if (show_unc and case.unc is not None) else 1
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 5.2), squeeze=False)
    ax = axes[0, 0]
    asp = _aspect(case.ct_spacing, plane)
    ax.imshow(_slice(case.ct, plane, idx), cmap="gray", vmin=window[0], vmax=window[1], aspect=asp)
    if show_pet:
        p = _slice(case.pet, plane, idx)
        ax.imshow(np.ma.masked_less(p, 0.3 * max(case.pet.max(), 1e-6)), cmap="hot", alpha=pet_alpha,
                  vmin=0, vmax=max(case.pet.max(), 1e-6), aspect=asp)
    if show_pred and case.pred is not None:
        m = _slice(case.pred.astype(float), plane, idx)
        if m.max() > 0:
            ax.contour(m, levels=[0.5], colors="cyan", linewidths=1.6)
    if case.label is not None:
        m = _slice(case.label.astype(float), plane, idx)
        if m.max() > 0:
            ax.contour(m, levels=[0.5], colors="lime", linewidths=1.2, linestyles="--")
    ax.set_title(f"{plane} {idx}   cyan = prediction, green dashed = ground truth", fontsize=8)
    ax.axis("off")
    if panels == 2:
        a2 = axes[0, 1]
        im = a2.imshow(_slice(case.unc, plane, idx), cmap="magma", vmin=0, vmax=1, aspect=asp)
        a2.set_title("uncertainty (0 = certain, 1 = no evidence)", fontsize=8)
        a2.axis("off")
        fig.colorbar(im, ax=a2, fraction=0.046)
    fig.tight_layout()
    return fig


def axis_length(case: Case, plane: str) -> int:
    return case.ct.shape[PLANES[plane]]
