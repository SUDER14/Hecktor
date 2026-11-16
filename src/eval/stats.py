"""Statistics for comparing models on the same cases.

* paired Wilcoxon signed-rank test (scipy) on per-case differences, two-sided;
* Holm step-down correction over the family of comparisons actually run (m is explicit). The proposal said
  Bonferroni; Holm controls the same family-wise error rate and is never less powerful (docs/ANALYSIS_PLAN.md);
* effect size: matched-pairs rank-biserial correlation r_rb = (W+ - W-) / (W+ + W-)
  (in [-1, 1]; sign = which model is better) and Cohen's d_z = mean(diff) / sd(diff);
* median paired difference with a percentile bootstrap 95% interval over cases (10,000 resamples, seeded);
  the same interval for the mean difference is kept as a secondary column;
* with several seeds per model, values are averaged per case over seeds BEFORE any test;
* per-centre tables.

Cases where either model has NaN on the metric (e.g. empty reference) are dropped
pairwise and COUNTED, never silently ignored. With few pairs the Wilcoxon p-value is
coarse (the smallest attainable two-sided p for n pairs is 2/2^n); the result row
carries n so that is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

N_BOOT = 10_000


def holm_adjust(p_values, m: int | None = None) -> np.ndarray:
    """Holm step-down adjusted p-values. adj_(i) = max_{j<=i} min(1, (m - j) * p_(j)), j = 0.. in ascending order.

    `m` is the family size; it defaults to len(p_values). NaN p-values (a comparison that could not be tested)
    stay NaN but still count in m, so an untestable comparison never makes the others easier to pass.
    """
    p = np.asarray(p_values, float)
    m = len(p) if m is None else int(m)
    adj = np.full(p.shape, np.nan)
    order = [i for i in np.argsort(p, kind="stable") if np.isfinite(p[i])]
    running = 0.0
    for j, i in enumerate(order):
        running = max(running, min(1.0, (m - j) * p[i]))
        adj[i] = running
    return adj


def paired_comparison(a: np.ndarray, b: np.ndarray, seed: int = 0, n_boot: int = N_BOOT) -> dict:
    """Compare model A to model B on matched cases (higher = better assumed by the sign of mean_diff)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    n = int(ok.sum())
    out = {"n_pairs": n, "n_dropped_nan": int((~ok).sum()), "mean_a": float(a[ok].mean()) if n else np.nan,
           "mean_b": float(b[ok].mean()) if n else np.nan, "mean_diff": float(d.mean()) if n else np.nan,
           "median_diff": float(np.median(d)) if n else np.nan}
    if n < 2 or np.allclose(d, 0):
        return {**out, "p_value": np.nan if n < 2 else 1.0, "rank_biserial": 0.0 if n >= 2 else np.nan,
                "cohen_dz": np.nan, "ci_low": np.nan, "ci_high": np.nan, "median_ci_low": np.nan, "median_ci_high": np.nan}
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    w_pos, w_neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    p = float(stats.wilcoxon(nz, alternative="two-sided").pvalue)
    rng = np.random.default_rng(seed)
    res = rng.choice(d, (n_boot, n), replace=True)          # resample CASES with replacement
    boots, med = res.mean(1), np.median(res, axis=1)
    sd = d.std(ddof=1)
    return {**out, "p_value": p, "rank_biserial": float((w_pos - w_neg) / (w_pos + w_neg)),
            "cohen_dz": float(d.mean() / sd) if sd > 0 else np.nan,
            "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
            "median_ci_low": float(np.percentile(med, 2.5)), "median_ci_high": float(np.percentile(med, 97.5))}


def compare_models(per_case: pd.DataFrame, metric: str, pairs: list[tuple[str, str]], model_col: str = "model",
                   case_col: str = "patient_id", alpha: float = 0.05, git: str | None = None) -> pd.DataFrame:
    """One row per (A, B) pair with the Holm-adjusted p over len(pairs) comparisons.

    Several rows per (case, model), i.e. several seeds, are averaged per case first (NaN-aware mean)."""
    wide = per_case.pivot_table(index=case_col, columns=model_col, values=metric, aggfunc="mean")
    rows = []
    for ma, mb in pairs:
        r = paired_comparison(wide[ma].to_numpy(), wide[mb].to_numpy())
        rows.append({"metric": metric, "model_a": ma, "model_b": mb, **r})
    df = pd.DataFrame(rows)
    m = len(pairs)
    df["p_holm"] = holm_adjust(df["p_value"].to_numpy(), m)
    df["significant"] = df["p_holm"] < alpha
    df["n_comparisons"] = m
    df["git"] = git
    return df


def per_centre_table(per_case: pd.DataFrame, metrics: list[str], model_col: str = "model",
                     git: str | None = None) -> pd.DataFrame:
    """mean / std / n per (model, centre); NaN-aware."""
    g = per_case.groupby([model_col, "centre"])[metrics]
    t = g.agg(["mean", "std", "count"])
    t.columns = [f"{m}_{s}" for m, s in t.columns]
    t = t.reset_index()
    t["git"] = git
    return t
