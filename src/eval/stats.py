"""Statistics for comparing models on the same cases.

* paired Wilcoxon signed-rank test (scipy) on per-case differences, two-sided;
* Bonferroni correction over the family of comparisons actually run (m is explicit);
* effect size: matched-pairs rank-biserial correlation r_rb = (W+ - W-) / (W+ + W-)
  (in [-1, 1]; sign = which model is better) and Cohen's d_z = mean(diff) / sd(diff);
* bootstrap 95% CI of the mean paired difference (percentile, seeded);
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


def paired_comparison(a: np.ndarray, b: np.ndarray, seed: int = 0, n_boot: int = 2000) -> dict:
    """Compare model A to model B on matched cases (higher = better assumed by the sign of mean_diff)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    n = int(ok.sum())
    out = {"n_pairs": n, "n_dropped_nan": int((~ok).sum()), "mean_a": float(a[ok].mean()) if n else np.nan,
           "mean_b": float(b[ok].mean()) if n else np.nan, "mean_diff": float(d.mean()) if n else np.nan}
    if n < 2 or np.allclose(d, 0):
        return {**out, "p_value": np.nan if n < 2 else 1.0, "rank_biserial": 0.0 if n >= 2 else np.nan,
                "cohen_dz": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    w_pos, w_neg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    p = float(stats.wilcoxon(nz, alternative="two-sided").pvalue)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, n), replace=True).mean(1)
    sd = d.std(ddof=1)
    return {**out, "p_value": p, "rank_biserial": float((w_pos - w_neg) / (w_pos + w_neg)),
            "cohen_dz": float(d.mean() / sd) if sd > 0 else np.nan,
            "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5))}


def compare_models(per_case: pd.DataFrame, metric: str, pairs: list[tuple[str, str]], model_col: str = "model",
                   case_col: str = "patient_id", alpha: float = 0.05, git: str | None = None) -> pd.DataFrame:
    """One row per (A, B) pair with Bonferroni-adjusted p over len(pairs) comparisons."""
    wide = per_case.pivot_table(index=case_col, columns=model_col, values=metric, aggfunc="first")
    rows = []
    for ma, mb in pairs:
        r = paired_comparison(wide[ma].to_numpy(), wide[mb].to_numpy())
        rows.append({"metric": metric, "model_a": ma, "model_b": mb, **r})
    df = pd.DataFrame(rows)
    m = len(pairs)
    df["p_bonferroni"] = np.minimum(df["p_value"] * m, 1.0)
    df["significant"] = df["p_bonferroni"] < alpha
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
