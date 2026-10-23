"""Calibration and uncertainty-quality metrics (numpy, any array shape).

ECE (Guo et al. 2017, arXiv:1706.04599): bin voxels by top-label confidence
c = max_k p_k into M equal-width bins on [0, 1]; in each bin compare mean confidence
with accuracy (argmax == label); ECE = sum_m (n_m / N) |acc_m - conf_m|.

For volumes dominated by easy background, ECE over *all* voxels is flattered by the
background; pass `roi` (e.g. a dilated union of label and prediction) to evaluate where
the decision is actually hard, and report which one was used.
"""

from __future__ import annotations

import numpy as np


def _flat(prob: np.ndarray, labels: np.ndarray, roi: np.ndarray | None):
    """prob (K, ...) + labels (...) -> conf (N,), correct (N,) restricted to roi."""
    prob = np.asarray(prob)
    labels = np.asarray(labels).astype(np.int64)
    conf = prob.max(axis=0)
    correct = (prob.argmax(axis=0) == labels)
    if roi is not None:
        m = np.asarray(roi).astype(bool)
        conf, correct = conf[m], correct[m]
    return conf.ravel().astype(np.float64), correct.ravel().astype(np.float64)


def reliability_bins(prob: np.ndarray, labels: np.ndarray, n_bins: int = 15, roi: np.ndarray | None = None) -> dict:
    """Per-bin count, mean confidence, accuracy. Empty bins are NaN (kept so plots show gaps)."""
    conf, correct = _flat(prob, labels, roi)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)  # bin m holds (edge_m, edge_{m+1}]
    count = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    sum_acc = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf, acc = sum_conf / count, sum_acc / count
    return {"edges": edges, "count": count, "confidence": mean_conf, "accuracy": acc}


def ece(prob: np.ndarray, labels: np.ndarray, n_bins: int = 15, roi: np.ndarray | None = None) -> float:
    """Expected calibration error in [0, 1]; NaN if there are no voxels."""
    b = reliability_bins(prob, labels, n_bins, roi)
    n = b["count"].sum()
    if n == 0:
        return float("nan")
    ok = b["count"] > 0
    return float(np.sum(b["count"][ok] / n * np.abs(b["accuracy"][ok] - b["confidence"][ok])))


def reliability_diagram(prob: np.ndarray, labels: np.ndarray, path: str, n_bins: int = 15,
                        roi: np.ndarray | None = None, title: str = "") -> float:
    """Save a reliability diagram PNG; returns the ECE it displays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = reliability_bins(prob, labels, n_bins, roi)
    e = ece(prob, labels, n_bins, roi)
    centres = (b["edges"][:-1] + b["edges"][1:]) / 2
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    ax[0].plot([0, 1], [0, 1], "k--", label="perfect")
    ok = b["count"] > 0
    ax[0].bar(centres[ok], b["accuracy"][ok], width=1 / n_bins, alpha=0.6, edgecolor="k", label="accuracy")
    ax[0].plot(b["confidence"][ok], b["accuracy"][ok], "o-", ms=3, label="(confidence, accuracy)")
    ax[0].set(xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1), title=f"{title} ECE={e:.4f}")
    ax[0].legend(loc="upper left")
    ax[1].bar(centres, b["count"], width=1 / n_bins, edgecolor="k")
    ax[1].set(yscale="log", xlabel="confidence", ylabel="voxels (log)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return e


def uncertainty_error_overlap(uncertainty: np.ndarray, pred: np.ndarray, label: np.ndarray,
                              roi: np.ndarray | None = None) -> dict:
    """How well does the uncertainty map point at the voxels that are actually wrong?

    * overlap_dice: Dice between the error set E = {pred != label} and the k most
      uncertain voxels, with k = |E| (same size, so 1.0 = uncertainty ranks exactly
      the errors first). Uses np.argpartition -- no full sort, no torch.quantile.
    * auroc: probability that a random error voxel has higher uncertainty than a random
      correct voxel (Mann-Whitney U, ties averaged); 0.5 = uninformative.
    * mean_u_error / mean_u_correct: mean uncertainty on each group.
    NaN fields when there are no errors (or no correct voxels).
    """
    u = np.asarray(uncertainty, dtype=np.float64)
    err = (np.asarray(pred).astype(np.int64) != np.asarray(label).astype(np.int64))
    if roi is not None:
        m = np.asarray(roi).astype(bool)
        u, err = u[m], err[m]
    u, err = u.ravel(), err.ravel()
    n_err, n = int(err.sum()), err.size
    nan = {"overlap_dice": float("nan"), "auroc": float("nan"), "mean_u_error": float("nan"),
           "mean_u_correct": float("nan"), "n_error": n_err, "n_voxels": n}
    if n_err == 0 or n_err == n:
        return nan
    top = np.argpartition(-u, n_err - 1)[:n_err]  # indices of the n_err largest uncertainties
    sel = np.zeros(n, bool)
    sel[top] = True
    overlap = 2.0 * (sel & err).sum() / (sel.sum() + err.sum())
    from scipy.stats import rankdata

    ranks = rankdata(u)  # average ranks for ties
    auroc = (ranks[err].sum() - n_err * (n_err + 1) / 2) / (n_err * (n - n_err))
    return {"overlap_dice": float(overlap), "auroc": float(auroc), "mean_u_error": float(u[err].mean()),
            "mean_u_correct": float(u[~err].mean()), "n_error": n_err, "n_voxels": n}
