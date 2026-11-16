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


def brier(prob: np.ndarray, labels: np.ndarray, roi: np.ndarray | None = None) -> float:
    """Multi-class Brier score: mean over voxels of sum_k (p_k - y_k)^2, in [0, 2]; 0 is perfect.
    prob (K, ...), labels (...) integer classes. NaN if there are no voxels. Unlike ECE it is a proper
    scoring rule, so it cannot be improved by being confidently wrong in a balanced way."""
    prob = np.asarray(prob, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    if prob.shape[1:] != labels.shape:
        raise ValueError(f"shape mismatch: prob {prob.shape} vs labels {labels.shape}")
    k = prob.shape[0]
    p, y = prob.reshape(k, -1), labels.ravel()
    if roi is not None:
        m = np.asarray(roi).astype(bool).ravel()
        p, y = p[:, m], y[m]
    if y.size == 0:
        return float("nan")
    p_true = p[y, np.arange(y.size)]
    return float(((p ** 2).sum(axis=0) - 2.0 * p_true + 1.0).mean())


# --- added for the locked test script (docs/ANALYSIS_PLAN.md section 7) -------------------------------------
def nll(prob: np.ndarray, labels: np.ndarray, roi: np.ndarray | None = None, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the true class, -log p_y, natural log. prob (K, ...), labels (...).
    Probabilities are clamped at `eps` so one voxel with p_y = 0 gives -log(eps), not infinity. NaN if no voxels."""
    prob = np.asarray(prob)
    labels = np.asarray(labels).astype(np.int64)
    if prob.shape[1:] != labels.shape:
        raise ValueError(f"shape mismatch: prob {prob.shape} vs labels {labels.shape}")
    p_true = np.take_along_axis(prob, labels[None], axis=0)[0]
    if roi is not None:
        p_true = p_true[np.asarray(roi).astype(bool)]
    if p_true.size == 0:
        return float("nan")
    return float(-np.log(np.clip(p_true.astype(np.float64), eps, None)).mean())


def apply_temperature(prob: np.ndarray, temperature: float, eps: float = 1e-12) -> np.ndarray:
    """p_T proportional to p^(1/T), renormalised over the class axis 0. For one softmax output this equals
    softmax(logits / T) exactly. For sliding-window output (probabilities averaged over overlapping windows) the
    logits no longer exist, so this is temperature scaling of the log of the averaged probability; say so
    wherever a temperature-scaled number is reported."""
    lp = np.log(np.clip(np.asarray(prob, dtype=np.float32), eps, None)) / np.float32(temperature)
    lp -= lp.max(axis=0, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(axis=0, keepdims=True)


def nll_sum_on_grid(prob: np.ndarray, labels: np.ndarray, grid) -> np.ndarray:
    """SUM over voxels of -log p_T[y] for every temperature in `grid`. Sums (not means) so that cases can be
    added before the temperature is chosen: the fit is over all validation voxels, not a mean of case means."""
    prob = np.asarray(prob, dtype=np.float32)
    labels = np.asarray(labels).astype(np.int64)
    out = np.zeros(len(grid), np.float64)
    for i, t in enumerate(grid):
        out[i] = nll(apply_temperature(prob, float(t)), labels) * labels.size
    return out


def fit_temperature(nll_sums: np.ndarray, grid) -> tuple[float, bool]:
    """(temperature with the smallest summed NLL on the grid, at_edge). at_edge=True means the minimum sits on
    the first or last grid point, so the true minimum may lie outside the grid; report it, do not hide it."""
    i = int(np.argmin(nll_sums))
    return float(np.asarray(grid)[i]), i in (0, len(grid) - 1)


def u_histograms(uncertainty: np.ndarray, pred: np.ndarray, label: np.ndarray, n_bins: int = 1000,
                 roi: np.ndarray | None = None) -> dict:
    """Counts of wrong and of correct voxels per uncertainty bin on [0, 1]. Histograms add across cases, which is
    what lets a pooled AUC and a pooled risk-coverage curve be computed without holding every voxel in memory.
    The price is that ties inside one bin (width 1 / n_bins) are not ordered."""
    u = np.clip(np.asarray(uncertainty, dtype=np.float64), 0.0, 1.0)
    wrong = np.asarray(pred).astype(np.int64) != np.asarray(label).astype(np.int64)
    if roi is not None:
        m = np.asarray(roi).astype(bool)
        u, wrong = u[m], wrong[m]
    idx = np.minimum((u.ravel() * n_bins).astype(np.int64), n_bins - 1)
    w = wrong.ravel()
    return {"wrong": np.bincount(idx[w], minlength=n_bins).astype(np.float64),
            "right": np.bincount(idx[~w], minlength=n_bins).astype(np.float64)}


def auroc_from_histograms(wrong: np.ndarray, right: np.ndarray) -> float:
    """P(u_wrong > u_right) + 0.5 P(same bin). NaN when either group is empty."""
    wrong, right = np.asarray(wrong, float), np.asarray(right, float)
    nw, nr = wrong.sum(), right.sum()
    if nw == 0 or nr == 0:
        return float("nan")
    right_below = np.cumsum(right) - right          # correct voxels in strictly lower bins
    return float((wrong * (right_below + 0.5 * right)).sum() / (nw * nr))


def risk_coverage_from_histograms(wrong: np.ndarray, right: np.ndarray) -> dict:
    """Keep voxels from the most certain bin upward. coverage[i] = fraction of voxels kept after bin i,
    risk[i] = error rate among them. aurc = area under risk against coverage (trapezoid; lower is better)."""
    wrong, right = np.asarray(wrong, float), np.asarray(right, float)
    kept, bad = np.cumsum(wrong + right), np.cumsum(wrong)
    total = kept[-1] if len(kept) else 0.0
    if total == 0:
        return {"coverage": np.array([]), "risk": np.array([]), "aurc": float("nan")}
    ok = kept > 0
    cov, risk = kept[ok] / total, bad[ok] / kept[ok]
    aurc = float(np.sum((cov[1:] - cov[:-1]) * (risk[1:] + risk[:-1]) / 2.0) + cov[0] * risk[0])
    return {"coverage": cov, "risk": risk, "aurc": aurc}
