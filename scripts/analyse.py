"""Comparisons between trained models, exactly as docs/ANALYSIS_PLAN.md section 4 states them.

    python scripts/analyse.py                      # reads results/test/<run>/per_case.csv, writes results/stats/
    python scripts/analyse.py --results-root <folder> --out <folder> --run B4=my_run_name --run B4=my_other_seed

Reads ONLY the per_case.csv tables that scripts/evaluate_test.py wrote (and each run's manifest.json, to refuse
mixing synthetic and real runs). It opens no image, label, cache file or checkpoint.

Family (fixed by the plan, not by what happens to be available):
    1 B4 vs B3 (primary)   2 B4 vs B2b   3 B4 vs B2   4 B5 vs B4   5 B2 vs B1      nnU-Net vs B4 (descriptive)
Metric for the tests: per-case mean Dice (column mean_dice). Paired Wilcoxon signed-rank by case, two-sided;
Holm step-down over the five comparisons with m = 5 always (a comparison that cannot be run stays in the count);
median paired difference with a 95% percentile bootstrap interval over cases, 10,000 resamples, fixed seed.
Several runs of one model are seeds: the metric is averaged per case over seeds first.
Per-label Dice, HD95 and surface Dice are reported for the same pairs with median differences and intervals,
without p-values. Differences are always (first model) minus (second model).

Runs are matched to models by folder name: shared_b1_* -> B1, shared_b2b_* -> B2b, shared_b2_* -> B2,
shared_b3_* -> B3, shared_b4_* -> B4, shared_b5_* -> B5, nnunet* -> nnU-Net; --run MODEL=name overrides this.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.stats import N_BOOT, holm_adjust, paired_comparison

ALPHA = 0.05
# (first, second, role); the five tested comparisons form the Holm family, in the plan's order
FAMILY = (("B4", "B3", "primary"), ("B4", "B2b", "secondary"), ("B4", "B2", "secondary"),
          ("B5", "B4", "secondary"), ("B2", "B1", "secondary"))
DESCRIPTIVE = (("nnU-Net", "B4", "descriptive"),)
PRIMARY_METRIC = "mean_dice"
# metric -> True when a larger value is better (only used to word the summary; differences are never flipped)
SECONDARY_METRICS = {"dice_1": True, "dice_2": True, "hd95_mm_1": False, "hd95_mm_2": False,
                     "surface_dice_2mm_1": True, "surface_dice_2mm_2": True,
                     "surface_dice_3mm_1": True, "surface_dice_3mm_2": True}
PREFIXES = (("shared_b2b_", "B2b"), ("shared_b1_", "B1"), ("shared_b2_", "B2"), ("shared_b3_", "B3"),
            ("shared_b4_", "B4"), ("shared_b5_", "B5"), ("nnunet", "nnU-Net"))
MODELS = ("B1", "B2", "B2b", "B3", "B4", "B5", "nnU-Net")


def model_of(run_name: str) -> str | None:
    low = run_name.lower()
    return next((m for pre, m in PREFIXES if low.startswith(pre)), None)


def discover(results_root: Path, explicit: list[str]) -> dict[str, list[str]]:
    """model -> run folder names. With --run given, ONLY those runs are used (nothing is picked up silently)."""
    runs: dict[str, list[str]] = {}
    if explicit:
        for item in explicit:
            if "=" not in item:
                raise SystemExit(f"bad --run {item!r}; expected MODEL=run_name with MODEL one of {MODELS}")
            m, name = item.split("=", 1)
            if m not in MODELS:
                raise SystemExit(f"--run {item!r}: unknown model {m!r}; use one of {MODELS}")
            runs.setdefault(m, []).append(name)
    else:
        for d in sorted(p for p in results_root.iterdir() if (p / "per_case.csv").exists()):
            m = model_of(d.name)
            if m is not None:
                runs.setdefault(m, []).append(d.name)
    for m, names in runs.items():
        if len(set(names)) != len(names):
            raise SystemExit(f"{m}: the same run is listed twice: {names}")
    return runs


def load_tables(results_root: Path, runs: dict[str, list[str]]) -> tuple[dict[str, pd.DataFrame], dict]:
    """model -> per-case table averaged over that model's seeds (index patient_id), plus what was read."""
    tables, info, synthetic = {}, {}, set()
    for m, names in runs.items():
        parts = []
        for name in names:
            f = results_root / name / "per_case.csv"
            if not f.exists():
                raise SystemExit(f"{f} does not exist")
            df = pd.read_csv(f)
            if "patient_id" not in df or PRIMARY_METRIC not in df:
                raise SystemExit(f"{f} has no patient_id / {PRIMARY_METRIC} column; it was not written by scripts/evaluate_test.py")
            if df["patient_id"].duplicated().any():
                raise SystemExit(f"{f} lists a case twice")
            mf = results_root / name / "manifest.json"
            if mf.exists():
                synthetic.add(bool(json.loads(mf.read_text(encoding="utf-8")).get("synthetic")))
            parts.append(df)
        ids = [frozenset(p["patient_id"]) for p in parts]
        if len(set(ids)) != 1:
            raise SystemExit(f"{m}: its runs {names} do not cover the same cases; seeds must be averaged over identical cases")
        num = [c for c in parts[0].columns if c != "patient_id" and pd.api.types.is_numeric_dtype(parts[0][c])
               and not pd.api.types.is_bool_dtype(parts[0][c])]
        # average over seeds per case first (plan section 4); a NaN in one seed is skipped, not propagated
        tables[m] = pd.concat([p[["patient_id", *num]] for p in parts]).groupby("patient_id")[num].mean()
        info[m] = {"runs": names, "n_seeds": len(names), "n_cases": len(tables[m]),
                   "n_cases_without_mean_dice": int(tables[m][PRIMARY_METRIC].isna().sum())}
    if len(synthetic) > 1:
        raise SystemExit("refusing: synthetic and real runs are mixed in one analysis")
    return tables, {"models": info, "synthetic": bool(synthetic and next(iter(synthetic)))}


def _row(tables: dict, a: str, b: str, role: str, metric: str, tested: bool) -> dict:
    base = {"metric": metric, "comparison": f"{a} vs {b}", "model_a": a, "model_b": b, "role": role, "tested": tested}
    missing = [m for m in (a, b) if m not in tables]
    if missing:
        return {**base, "status": f"not run: no results for {', '.join(missing)}", "n_pairs": 0, "p_value": np.nan}
    if metric not in tables[a] or metric not in tables[b]:
        return {**base, "status": f"not run: column {metric} missing", "n_pairs": 0, "p_value": np.nan}
    ids = sorted(set(tables[a].index) | set(tables[b].index))
    va, vb = tables[a][metric].reindex(ids).to_numpy(float), tables[b][metric].reindex(ids).to_numpy(float)
    r = paired_comparison(va, vb, seed=0, n_boot=N_BOOT)
    ok = ~(np.isnan(va) | np.isnan(vb))
    r["median_a"], r["median_b"] = (float(np.median(va[ok])), float(np.median(vb[ok]))) if ok.any() else (np.nan, np.nan)
    if not tested:
        r["p_value"] = np.nan   # descriptive rows carry no p-value, by the plan
    status = "ok" if r["n_pairs"] >= 2 else "not run: fewer than 2 paired cases"
    return {**base, "status": status, **r, "n_cases_only_in_one_model": int(len(ids) - len(set(tables[a].index) & set(tables[b].index)))}


def analyse(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [_row(tables, a, b, role, PRIMARY_METRIC, True) for a, b, role in FAMILY]
    p = np.array([r["p_value"] for r in rows], float)
    for r, adj in zip(rows, holm_adjust(p, m=len(FAMILY))):   # m = 5 always; an untested comparison stays in the count
        r["p_holm"], r["holm_m"] = adj, len(FAMILY)
        r["significant"] = bool(np.isfinite(adj) and adj < ALPHA)
    rows += [_row(tables, a, b, role, PRIMARY_METRIC, False) for a, b, role in DESCRIPTIVE]
    for metric in SECONDARY_METRICS:   # same pairs, median difference and interval only
        rows += [_row(tables, a, b, role, metric, False) for a, b, role in (*FAMILY, *DESCRIPTIVE)]
    cols = ["metric", "comparison", "model_a", "model_b", "role", "tested", "status", "n_pairs", "n_dropped_nan",
            "n_cases_only_in_one_model", "median_a", "median_b", "median_diff", "median_ci_low", "median_ci_high",
            "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high", "rank_biserial", "cohen_dz", "p_value", "p_holm",
            "holm_m", "significant"]
    return pd.DataFrame(rows).reindex(columns=cols)


def _f(x, nd=4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def summary_text(res: pd.DataFrame, info: dict) -> str:
    out = ["COMPARISONS (docs/ANALYSIS_PLAN.md section 4). Differences are first model minus second model.",
           f"Written {time.strftime('%Y-%m-%d %H:%M:%S')}. Metric for the tests: per-case mean Dice.",
           (f"Paired Wilcoxon by case, two-sided; Holm over {len(FAMILY)} comparisons (m = {len(FAMILY)} always); "
           f"median paired difference with a 95% bootstrap interval over cases, {N_BOOT} resamples."), ""]
    if info["synthetic"]:
        out += ["*** SYNTHETIC RUNS: these numbers mean nothing; they only show that the script works. ***", ""]
    out.append("Runs read (seeds are averaged per case first):")
    for m in MODELS:
        i = info["models"].get(m)
        out.append(f"  {m}: no results" if i is None else
                   f"  {m}: {i['n_seeds']} seed(s), {i['n_cases']} cases, {i['n_cases_without_mean_dice']} without a per-case mean Dice; runs {', '.join(i['runs'])}")
    out.append("")
    main = res[res["metric"] == PRIMARY_METRIC]
    for _, r in main.iterrows():
        head = f"{r['comparison']} ({r['role']})"
        if r["status"] != "ok":
            out.append(f"{head}: {r['status']}" + (" (still counted in the Holm family)" if r["tested"] else ""))
            continue
        line = (f"{head}: n = {int(r['n_pairs'])} cases, median difference {_f(r['median_diff'])} "
                f"[{_f(r['median_ci_low'])}, {_f(r['median_ci_high'])}], mean difference {_f(r['mean_diff'])}")
        if r["tested"]:
            verdict = "significant at 0.05 after Holm" if r["significant"] else "not significant after Holm"
            line += f", Wilcoxon p = {_f(r['p_value'], 5)}, Holm p = {_f(r['p_holm'], 5)}: {verdict}"
        else:
            line += " (descriptive, no test)"
        out.append(line)
    if (main["median_ci_low"].isna() & (main["status"] == "ok")).any():
        out.append("An interval shown as n/a means every paired difference was exactly zero.")
    out += ["", "Per-label Dice, HD95 and surface Dice for the same pairs are in comparisons.csv (no p-values, by the plan).",
            "For HD95 a negative difference favours the first model; for the Dice measures a positive one does."]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=ROOT / "results" / "test")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "stats")
    ap.add_argument("--run", action="append", default=[], metavar="MODEL=RUN_NAME",
                    help="use exactly these runs (repeat for seeds); without it runs are found by folder name")
    a = ap.parse_args()
    if not a.results_root.is_dir():
        raise SystemExit(f"{a.results_root} does not exist; scripts/evaluate_test.py has written no results there")
    runs = discover(a.results_root, a.run)
    if not runs:
        raise SystemExit(f"no run with a per_case.csv found under {a.results_root}")
    tables, info = load_tables(a.results_root, runs)
    res = analyse(tables)
    a.out.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out / "comparisons.csv", index=False)
    text = summary_text(res, info)
    (a.out / "comparisons_summary.txt").write_text(text, encoding="utf-8")
    (a.out / "comparisons_inputs.json").write_text(json.dumps({"results_root": a.results_root.name, **info}, indent=1), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
