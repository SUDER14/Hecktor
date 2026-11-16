"""Phase 4 correctness gates on CPU, shared-PET path, one fixed batch of real TRAINING crops.

    python scripts/phase4_gates.py --out results/fable_run/gates.json            # all gates, resumable per gate
    python scripts/phase4_gates.py --out ... --only overfit:shared_b4_acf_cond   # one gate

These are wiring checks, not results: a model that cannot memorise two crops, or whose conditioned and
zeroed variants differ at initialisation, has a broken pipeline. No test case is ever opened. Thresholds
are the ones in the run specification and are not tuned to make a gate pass; a failing gate is recorded
as FAIL with its measured value.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from monai.data import list_data_collate
from torch.amp import GradScaler

from src.data.acf_dataset import AcfCropDataset, AcfVolumeDataset
from src.data.conditioning import load_stats
from src.data.seed import seed_everything
from src.data.splits import load_split
from src.eval.evaluate_shared import call_model
from src.eval.infer import resample_to_grid, sliding_window
from src.models.evidential import dirichlet_summary
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.losses import kl_weight
from src.training.run import build_model, build_optimizer_and_scheduler, compute_loss

CONFIGS = ("shared_b1_unet_resample", "shared_b2_dual_resample", "shared_b2b_dual_matched",
           "shared_b3_acf_zeroed", "shared_b4_acf_cond", "shared_b5_acf_evidential")
BATCH_CASES = ("MDA-071", "MDA-284")   # training cases (asserted below); median and thickest slice spacing
PATCH = [48, 48, 48]
SW_CASE = "HMR-012"   # smallest validation volume (219 x 219 x 107 voxels); asserted to be in val, not test
OVERFIT_DICE, OVERFIT_MAX_STEPS, OVERFIT_LR = 0.9, 400, 1e-3
LABEL_BUDGET_S = 14 * 60   # per config for the per-label overfit gate: six configs stay inside 90 minutes in total
STEPS_PER_EPOCH = 30   # only to advance the evidential KL annealing schedule during the overfit gate


def load_config(name: str) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
    cfg["patch"] = {**cfg["patch"], "spatial_size": PATCH, "num_samples": 1, "pos": 1, "neg": 0}
    cfg["split"]["path"] = str(ROOT / cfg["split"]["path"])
    cfg["conditioning"]["stats_path"] = str(ROOT / cfg["conditioning"]["stats_path"])
    return cfg


def fixed_batch(cfg: dict) -> dict:
    split = load_split(cfg["split"]["path"])
    assert set(BATCH_CASES) <= set(split["train"]) and not set(BATCH_CASES) & set(split["test"])
    ds = AcfCropDataset([{"patient_id": p, "centre": p.split("-")[0]} for p in BATCH_CASES], cfg["cache"]["dir"], cfg,
                        load_stats(cfg["conditioning"]["stats_path"]))
    ds.set_epoch(1)
    b = list_data_collate([ds[0], ds[1]])
    return {k: v for k, v in b.items() if torch.is_tensor(v)}


def forward(model, b):
    return call_model(model, b["image"], b["cond"], b["spacing"])


def predict(out, cfg):
    """Class map from logits (softmax head) or evidence (evidential head); argmax is the same for both."""
    return out.argmax(1)


def fg_dice(pred, label) -> float:
    p, t = pred > 0, label[:, 0] > 0
    return float(2 * (p & t).sum() / max(1, int(p.sum() + t.sum())))


def label_dice(pred, label) -> dict:
    out = {}
    for k in (1, 2):
        p, t = pred == k, label[:, 0] == k
        if t.any():
            out[str(k)] = float(2 * (p & t).sum() / max(1, int(p.sum() + t.sum())))
    return out


def fresh(cfg):
    seed_everything(cfg["seed"])
    return build_model(cfg)


def gate_overfit(name: str) -> dict:
    """Memorise the fixed batch; pass = label-foreground Dice > 0.9. For ACF configs the step-50 model is also probed:
    does changing only the spacing information change the output?"""
    cfg = load_config(name)
    b = fixed_batch(cfg)
    model = fresh(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=OVERFIT_LR, weight_decay=0.0)
    res = {"n_params": sum(p.numel() for p in model.parameters()), "label_voxels": {str(k): int((b["label"] == k).sum()) for k in (0, 1, 2)},
           "patch": PATCH, "lr": OVERFIT_LR, "threshold": OVERFIT_DICE, "curve": []}
    t0, best, step = time.perf_counter(), 0.0, 0
    for step in range(1, OVERFIT_MAX_STEPS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, _ = compute_loss(forward(model, b), b["label"], cfg, epoch=1 + (step - 1) // STEPS_PER_EPOCH)
        loss.backward()
        opt.step()
        if step == 50 and cfg["model"]["name"] == "acf":
            res["spacing_probe_step50"] = spacing_probe(model, b)
        if step % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = predict(forward(model, b), cfg)
            d = fg_dice(pred, b["label"])
            best = max(best, d)
            res["curve"].append([step, round(float(loss), 5), round(d, 4)])
            print(f"{name} step {step} loss {float(loss):.4f} fg Dice {d:.4f}", flush=True)
            if d > OVERFIT_DICE:
                res["label_dice_at_pass"] = label_dice(pred, b["label"])
                break
    res.update(steps=step, best_fg_dice=best, seconds=round(time.perf_counter() - t0, 1), passed=best > OVERFIT_DICE)
    return res


def spacing_probe(model, b) -> dict:
    """Max |output difference| when (i) only the z-scored conditioning vector changes, (ii) the mm spacing argument
    changes too. The image is untouched."""
    model.eval()
    with torch.no_grad():
        base = call_model(model, b["image"], b["cond"], b["spacing"])
        cond2 = b["cond"].flip(0) + 1.0
        d_cond = float((call_model(model, b["image"], cond2, b["spacing"]) - base).abs().max())
        d_both = float((call_model(model, b["image"], cond2, b["spacing"].flip(0)) - base).abs().max())
    return {"max_abs_diff_cond_only": d_cond, "max_abs_diff_cond_and_mm": d_both, "output_abs_max": float(base.abs().max())}


def gate_fresh_identity() -> dict:
    """With fresh weights B4 (conditioned) and B3 (conditioning zeroed) must give identical output to 1e-5."""
    c3, c4 = load_config("shared_b3_acf_zeroed"), load_config("shared_b4_acf_cond")
    b = fixed_batch(c4)
    m3, m4 = fresh(c3).eval(), fresh(c4).eval()
    same_init = all(torch.equal(p, q) for p, q in zip(m3.state_dict().values(), m4.state_dict().values()))
    with torch.no_grad():
        d = float((forward(m3, b) - forward(m4, b)).abs().max())
    return {"same_initial_weights": same_init, "max_abs_diff": d, "tolerance": 1e-5, "passed": same_init and d <= 1e-5}


def gate_b5_head() -> dict:
    cfg = load_config("shared_b5_acf_evidential")
    b = fixed_batch(cfg)
    model = fresh(cfg).eval()
    with torch.no_grad():
        s = dirichlet_summary(forward(model, b))
    k = s["alpha"].shape[1]
    an = cfg["loss"]["kl_anneal_epochs"]
    sched = [kl_weight(e, an) for e in range(an + 3)]
    r = {"K": k, "alpha_min": float(s["alpha"].min()), "prob_sum_max_abs_err": float((s["p"].sum(1) - 1).abs().max()),
         "u_min": float(s["u"].min()), "u_max": float(s["u"].max()),
         "u_equals_K_over_S_max_abs_err": float((s["u"] - k / s["S"]).abs().max()), "kl_anneal_epochs": an, "kl_weights": sched}
    r["passed"] = bool(k == 3 and r["alpha_min"] >= 1.0 and r["prob_sum_max_abs_err"] < 1e-5 and 0.0 < r["u_min"] and r["u_max"] <= 1.0
                       and r["u_equals_K_over_S_max_abs_err"] < 1e-6 and all(0.0 <= w <= 1.0 for w in sched)
                       and all(a <= c for a, c in itertools.pairwise(sched)) and sched[an] == 1.0 and sched[-1] == 1.0)
    return r


def _steps(cfg, b, n, model=None, opt=None, sched=None, start=0):
    if model is None:
        model = fresh(cfg)
        opt, sched = build_optimizer_and_scheduler(model, cfg)
    losses = []
    for i in range(start, start + n):
        model.train()
        opt.zero_grad(set_to_none=True)
        noise = torch.randn_like(b["image"]) * 0.01          # consumes the torch RNG, so RNG restore is exercised
        loss, _ = compute_loss(call_model(model, b["image"] + noise, b["cond"], b["spacing"]), b["label"], cfg, epoch=1 + i // 10)
        loss.backward()
        opt.step()
        if sched is not None and (i + 1) % 10 == 0:
            sched.step()
        losses.append(float(loss))
    return model, opt, sched, losses


def gate_seed_repeat() -> dict:
    cfg = load_config("shared_b4_acf_cond")
    b = fixed_batch(cfg)
    a = _steps(cfg, b, 10)[3]
    c = _steps(cfg, b, 10)[3]
    return {"losses_run1": a, "losses_run2": c, "max_abs_diff": float(np.abs(np.array(a) - np.array(c)).max()), "passed": a == c}


def gate_resume() -> dict:
    """20 steps, checkpoint, rebuild everything from the file, 20 more == 40 straight (model, optimiser, scheduler,
    scaler, RNG). The trainer itself checkpoints once per epoch; this exercises the same save/load functions per step."""
    cfg = load_config("shared_b4_acf_cond")
    cfg["train"] = {**cfg["train"], "n_epochs": 4}   # cosine T_max, so the scheduler state matters within 40 steps
    b = fixed_batch(cfg)
    m_ref, o_ref, _, l_ref = _steps(cfg, b, 40)
    m, opt, sched, l_a = _steps(cfg, b, 20)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "epoch_0002.pt"
        save_checkpoint(p, 2, m, opt, GradScaler(device="cpu", enabled=False), -1.0, cfg, scheduler=sched, extra={"step": 20})
        torch.manual_seed(999)   # wreck the RNG; the checkpoint must restore it
        m2 = build_model(cfg)
        opt2, sched2 = build_optimizer_and_scheduler(m2, cfg)
        ck = load_checkpoint(p, m2, opt2, GradScaler(device="cpu", enabled=False), sched2, map_location="cpu")
    step = ck.get("step")   # save_checkpoint merges `extra` keys into the top level of the payload
    _, _, _, l_b = _steps(cfg, b, 20, m2, opt2, sched2, start=20)
    wdiff = max(float((x - y).abs().max()) for x, y in zip(m_ref.state_dict().values(), m2.state_dict().values()))
    ldiff = float(np.abs(np.array(l_ref) - np.array(l_a + l_b)).max())
    lr_ref, lr_res = o_ref.param_groups[0]["lr"], opt2.param_groups[0]["lr"]
    return {"restored_step": step, "restored_epoch": ck.get("epoch"), "max_abs_loss_diff": ldiff, "max_abs_weight_diff": wdiff,
            "lr_ref_vs_resumed": [lr_ref, lr_res], "lr_at_start": cfg["train"]["lr"],
            "passed": step == 20 and ck.get("epoch") == 2 and ldiff == 0.0 and wdiff == 0.0 and lr_ref == lr_res
            and lr_ref < cfg["train"]["lr"]}


def gate_overfit_labels(name: str) -> dict:
    """Stricter overfit gate: memorise the fixed batch until the Dice of EACH label (1 and 2) is above 0.9. The
    foreground gate above can pass with label 1 unlearned, because label 1 is ~2% of the foreground voxels in
    the batch. A config that runs out of its time budget is a FAIL with its best values recorded."""
    cfg = load_config(name)
    b = fixed_batch(cfg)
    vox = {str(k): int((b["label"] == k).sum()) for k in (0, 1, 2)}
    if min(vox["1"], vox["2"]) == 0:
        return {"passed": False, "label_voxels": vox, "error": "fixed batch lacks label 1 or label 2; gate is meaningless"}
    model = fresh(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=OVERFIT_LR, weight_decay=0.0)
    res = {"label_voxels": vox, "label_voxels_per_crop": [{str(k): int((b["label"][i] == k).sum()) for k in (1, 2)} for i in range(b["label"].shape[0])],
           "lr": OVERFIT_LR, "threshold_each_label": OVERFIT_DICE, "budget_s": LABEL_BUDGET_S, "curve": []}
    t0, best, step, passed = time.perf_counter(), {"1": 0.0, "2": 0.0}, 0, False
    while time.perf_counter() - t0 < LABEL_BUDGET_S:
        step += 1
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, _ = compute_loss(forward(model, b), b["label"], cfg, epoch=1 + (step - 1) // STEPS_PER_EPOCH)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = predict(forward(model, b), cfg)
            d = label_dice(pred, b["label"])
            best = {k: max(best[k], d[k]) for k in best}
            res["curve"].append([step, round(float(loss), 5), round(d["1"], 4), round(d["2"], 4)])
            print(f"{name} step {step} loss {float(loss):.4f} Dice label1 {d['1']:.4f} label2 {d['2']:.4f}", flush=True)
            if min(d.values()) > OVERFIT_DICE:
                passed, res["label_dice_at_pass"] = True, d
                break
    res.update(steps=step, best_label_dice_seen=best, seconds=round(time.perf_counter() - t0, 1), passed=passed)
    return res


def gate_spacing_after_50() -> dict:
    """After 50 training steps on the fixed batch, changing only the spacing vector must change B4's output and
    must not change B3's (its conditioning input is zeroed). Its own gate, so it always trains the full 50 steps."""
    r = {}
    for key, name in (("b3", "shared_b3_acf_zeroed"), ("b4", "shared_b4_acf_cond")):
        cfg = load_config(name)
        b = fixed_batch(cfg)
        model = fresh(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=OVERFIT_LR, weight_decay=0.0)
        for _ in range(50):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss, _ = compute_loss(forward(model, b), b["label"], cfg, epoch=1)
            loss.backward()
            opt.step()
        r[key] = {**spacing_probe(model, b), "loss_at_step_50": float(loss)}
    r["passed"] = r["b3"]["max_abs_diff_cond_only"] == 0.0 and r["b4"]["max_abs_diff_cond_only"] > 1e-4
    return r


def gate_sliding_window() -> dict:
    """Sliding-window inference on one VALIDATION case (the smallest, HMR-012) at the configured 96^3 patch and
    overlap, fresh weights: the output must land on the native label grid. B4 predicts on the native grid; B1 and
    B2 predict on the resampled grid and are mapped back before scoring, exactly as evaluate_shared does."""
    r = {}
    for name in CONFIGS:
        cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
        split = load_split(ROOT / cfg["split"]["path"])
        assert SW_CASE in split["val"] and SW_CASE not in split["test"]
        ds = AcfVolumeDataset([{"patient_id": SW_CASE, "centre": SW_CASE.split("-")[0]}], cfg["cache"]["dir"], cfg,
                              load_stats(ROOT / cfg["conditioning"]["stats_path"]))
        item = ds[0]
        image, aff = item["image"].numpy(), item["affine"].numpy()
        label, lab_aff = item["native_label"].numpy(), item["native_affine"].numpy()
        model = fresh(cfg).eval()
        cc, csp = item["cond"][None], item["spacing"][None]
        k = cfg["model"]["out_channels"]
        t0 = time.perf_counter()
        res = sliding_window(lambda x, m=model, cc=cc, csp=csp: call_model(m, x, cc.expand(x.shape[0], -1), csp.expand(x.shape[0], -1)),
                             image, tuple(cfg["patch"]["spatial_size"]), k, cfg["model"].get("head", "softmax"),
                             cfg["infer"]["overlap"], 1, "cpu")
        prob = res["p"]
        same_grid = prob.shape[1:] == label.shape and np.allclose(aff, lab_aff)
        if not same_grid:
            prob = resample_to_grid(prob, aff, lab_aff, label.shape, order=1)
        pred = prob.argmax(0)
        r[name] = {"mode": cfg["data"]["mode"], "model_grid": list(image.shape[1:]), "label_grid": list(label.shape),
                   "mapped_back": not same_grid, "pred_grid": list(pred.shape), "finite": bool(np.isfinite(prob).all()),
                   "prob_sum_max_abs_err": float(np.abs(prob.sum(0) - 1).max()), "seconds": round(time.perf_counter() - t0, 1),
                   "shape_ok": tuple(pred.shape) == tuple(label.shape)}
        print(name, r[name], flush=True)
    r["passed"] = bool(len(r) == len(CONFIGS) and all(
        v["shape_ok"] and v["finite"] and v["prob_sum_max_abs_err"] < 1e-3 and v["mapped_back"] == (v["mode"] == "resampled")
        for v in r.values()))
    return r


def gate_split_leak() -> dict:
    """No test ID in any training or validation list, and the conditioning stats file names no test statistic."""
    r = {}
    for name in CONFIGS:
        cfg = load_config(name)
        s = load_split(cfg["split"]["path"])
        test = set(s["test"])
        stats_text = Path(cfg["conditioning"]["stats_path"]).read_text(encoding="utf-8")
        st = json.loads(stats_text)
        r[name] = {"train_test_overlap": len(set(s["train"]) & test), "val_test_overlap": len(set(s["val"]) & test),
                   "test_centres_in_train_val": sorted({i.split("-")[0] for i in s["test"]} & {i.split("-")[0] for i in s["train"] + s["val"]}),
                   "config_test_centres": cfg["split"]["test_centres"], "stats_n_cases": st.get("n_cases", st.get("n")),
                   "stats_mentions_test_id": any(i in stats_text for i in test)}
    ok = all(v["train_test_overlap"] == 0 and v["val_test_overlap"] == 0 and not v["test_centres_in_train_val"]
             and not v["stats_mentions_test_id"] for v in r.values())
    return {"per_config": r, "passed": ok}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    torch.set_num_threads(max(1, (torch.get_num_threads() or 2)))
    only = a.only.split(",") if a.only else None   # gate names or prefixes ("overfit_labels" = all six); forces a re-run
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    gates = {"split_leak": gate_split_leak, "fresh_identity": gate_fresh_identity, "b5_head": gate_b5_head,
             "seed_repeat": gate_seed_repeat, "resume": gate_resume, "spacing_after_50": gate_spacing_after_50,
             "sliding_window": gate_sliding_window}
    gates.update({f"overfit:{n}": (lambda n=n: gate_overfit(n)) for n in CONFIGS})
    gates.update({f"overfit_labels:{n}": (lambda n=n: gate_overfit_labels(n)) for n in CONFIGS})
    for key, fn in gates.items():
        if (only and not any(key == o or key.startswith(o + ":") for o in only)) or (key in res and not only):
            continue
        print(f"=== {key}", flush=True)
        t0 = time.perf_counter()
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001 - recorded as FAIL with its message; a gate that cannot run is not a pass
            r = {"passed": False, "error": f"{type(e).__name__}: {e}"}
        r["seconds_total"] = round(time.perf_counter() - t0, 1)
        res[key] = r
        a.out.write_text(json.dumps(res, indent=1))
        print(f"=== {key}: {'PASS' if r.get('passed') else 'FAIL'} ({r['seconds_total']} s)", flush=True)
    print(json.dumps({k: copy.deepcopy(v).get("passed") for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
