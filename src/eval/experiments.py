"""Experiments E1-E6. Each takes `runs` = {label: run_dir} (a run dir written by
src/training/run.py: config.yaml + checkpoints/best.pt) and writes CSV tables + PNG figures.
Every table row carries the git hash of the code that produced it.

 E1 held-out centre   per-case + per-centre metrics on the held-out test split
 E2 anisotropy        CT through-plane spacing coarsened to 2, 3, 5, 7 mm
 E3 resources         peak VRAM, preprocessing and inference wall-clock
 E4 calibration       ECE + reliability diagrams + uncertainty/error overlap
 E5 ablations         paired comparisons between named model pairs (B3 vs B4, B4 vs B5, ...)
 E6 statistics        paired Wilcoxon + Bonferroni + effect sizes on any per-case table

Nothing here fabricates a number: with no checkpoints or no GPU the functions either raise or
write an explicit "not measured" marker.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.conditioning import load_stats  # noqa: E402
from src.data.dual_cache import build_dual_datasets, build_entry  # noqa: E402
from src.eval.evaluate_cases import evaluate_dataset, pooled_ece, summarize  # noqa: E402
from src.eval.stats import compare_models, per_centre_table  # noqa: E402
from src.training.run import build_model  # noqa: E402
from src.training.train import get_git_info  # noqa: E402

DEGRADE_MM = (None, 2.0, 3.0, 5.0, 7.0)  # None = native acquisition


def load_run(run_dir: str | Path, ckpt: str = "best.pt", device="cpu"):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    path = run_dir / "checkpoints" / ckpt
    if not path.exists():
        cands = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
        if not cands:
            raise FileNotFoundError(f"no checkpoint in {run_dir / 'checkpoints'}; train first")
        path = cands[-1]
    model = build_model(cfg)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model_state"])
    return cfg, model.to(device), path


def _git() -> str | None:
    return get_git_info()["commit"]


def _tagged(d: pd.DataFrame, label: str) -> pd.DataFrame:
    """Copy with a model column and without attrs (pooled reliability arrays break pandas' concat)."""
    out = d.assign(model=label)
    out.attrs = {}
    return out


def _stack(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([_tagged(d, k) for k, d in parts.items()], ignore_index=True)


def e1_heldout(runs: dict, out: str | Path, split: str = "test", device="cpu", max_cases=None) -> pd.DataFrame:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    git, parts = _git(), {}
    for label, rd in runs.items():
        cfg, model, _ = load_run(rd, device=device)
        ds = build_dual_datasets(cfg)[split]
        parts[label] = evaluate_dataset(model, ds, cfg, load_stats(cfg["conditioning"]["stats_path"]), device,
                                        max_cases=max_cases, git_hash=git, tag="E1")
    per_case = _stack(parts)
    per_case.to_csv(out / "e1_per_case.csv", index=False)
    summ = pd.DataFrame([{"model": k, **summarize(v), "git": git} for k, v in parts.items()])
    summ.to_csv(out / "e1_summary.csv", index=False)
    per_centre_table(per_case, ["dice", "hd95_mm", "surface_dice_2mm", "ece_roi"], git=git).to_csv(
        out / "e1_per_centre.csv", index=False)
    return per_case


def e2_anisotropy(runs: dict, out: str | Path, split: str = "test", levels=DEGRADE_MM, device="cpu",
                  max_cases=None) -> pd.DataFrame:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    git, parts = _git(), []
    for label, rd in runs.items():
        cfg, model, _ = load_run(rd, device=device)
        ds = build_dual_datasets(cfg)[split]
        stats = load_stats(cfg["conditioning"]["stats_path"])
        for lv in levels:
            parts.append(_tagged(evaluate_dataset(model, ds, cfg, stats, device, degrade_z_mm=lv, max_cases=max_cases,
                                                  git_hash=git, tag="E2"), label))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out / "e2_per_case.csv", index=False)
    g = df.assign(level=df.degrade_z_mm.fillna(0)).groupby(["model", "level"])[["dice", "hd95_mm"]].mean().reset_index()
    g["git"] = git
    g.to_csv(out / "e2_summary.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for m, sub in g.groupby("model"):
        for a, c in zip(ax, ("dice", "hd95_mm")):
            a.plot(sub.level, sub[c], "o-", label=m)
    ax[0].set(xlabel="CT slice thickness (mm; 0 = native)", ylabel="Dice")
    ax[1].set(xlabel="CT slice thickness (mm; 0 = native)", ylabel="HD95 (mm)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out / "e2_anisotropy.png", dpi=130)
    plt.close(fig)
    return df


def e3_resources(runs: dict, out: str | Path, device="cpu") -> pd.DataFrame:
    """Peak VRAM needs CUDA; without it the field is written as 'not measured', never estimated."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    git, rows = _git(), []
    cuda = torch.device(device).type == "cuda"
    for label, rd in runs.items():
        cfg, model, _ = load_run(rd, device=device)
        sample = _first_train_sample(cfg)
        t0 = time.perf_counter()
        mode = cfg.get("data", {}).get("mode", "native")
        build_entry(sample, cfg, Path(out) / f"_prep_{label}", mode, "float16", int(cfg["seed"]))
        prep_s = time.perf_counter() - t0
        ds = build_dual_datasets(cfg)
        stats = load_stats(cfg["conditioning"]["stats_path"])
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        evaluate_dataset(model, ds["val"], cfg, stats, device, max_cases=1)
        inf_s = time.perf_counter() - t0
        rows.append({"model": label, "mode": mode, "params": sum(p.numel() for p in model.parameters()),
                     "preprocess_s_per_patient": prep_s, "inference_s_per_case": inf_s,
                     "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if cuda else "not measured (no CUDA)",
                     "device": str(device), "git": git})
    df = pd.DataFrame(rows)
    df.to_csv(out / "e3_resources.csv", index=False)
    return df


def _first_train_sample(cfg):
    from src.data.dataset import build_samples
    return build_samples(cfg)["train"][0]


def e4_calibration(runs: dict, out: str | Path, split: str = "test", device="cpu", max_cases=None) -> pd.DataFrame:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    git, rows = _git(), []
    for label, rd in runs.items():
        cfg, model, _ = load_run(rd, device=device)
        ds = build_dual_datasets(cfg)[split]
        df = evaluate_dataset(model, ds, cfg, load_stats(cfg["conditioning"]["stats_path"]), device,
                              max_cases=max_cases, git_hash=git, tag="E4")
        pooled = df.attrs["reliability"]
        _plot_pooled(pooled, out / f"e4_reliability_{label}.png", label)
        s = summarize(df)
        rows.append({"model": label, "ece_pooled_roi": pooled_ece(pooled), "ece_all_mean": s["ece_all"],
                     "unc_auroc": s["unc_auroc"], "dice": s["dice"], "git": git})
    res = pd.DataFrame(rows)
    res.to_csv(out / "e4_calibration.csv", index=False)
    return res


def _plot_pooled(p: dict, path: Path, title: str) -> None:
    ok = p["count"] > 0
    centres = ((p["edges"][:-1] + p["edges"][1:]) / 2)
    conf, acc = p["sum_conf"][ok] / p["count"][ok], p["sum_acc"][ok] / p["count"][ok]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k--")
    ax.bar(centres[ok], acc, width=1 / 15, alpha=0.6, edgecolor="k")
    ax.plot(conf, acc, "o-", ms=3)
    ax.set(xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1),
           title=f"{title}\nECE(roi, pooled)={pooled_ece(p):.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def e5_ablations(per_case: pd.DataFrame, pairs: list[tuple[str, str]], out: str | Path,
                 metrics=("dice", "hd95_mm", "surface_dice_2mm")) -> pd.DataFrame:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    git = _git()
    tabs = [compare_models(per_case, m, pairs, git=git) for m in metrics]
    res = pd.concat(tabs, ignore_index=True)
    res.to_csv(out / "e5_ablations.csv", index=False)
    return res


def e6_statistics(per_case: pd.DataFrame, out: str | Path, reference: str, metrics=("dice", "hd95_mm")) -> pd.DataFrame:
    """Every other model vs `reference`, Bonferroni over that family, plus per-centre means."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    others = [m for m in per_case["model"].unique() if m != reference]
    pairs = [(reference, o) for o in others]
    git = _git()
    res = pd.concat([compare_models(per_case, m, pairs, git=git) for m in metrics], ignore_index=True)
    res.to_csv(out / "e6_statistics.csv", index=False)
    per_centre_table(per_case, list(metrics), git=git).to_csv(out / "e6_per_centre.csv", index=False)
    return res
