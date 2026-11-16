"""Diagnosis for the Phase 4 per-label overfit gate, B5 (evidential head) only.

    python scripts/phase4_b5_label1_diag.py --out results/fable_run/b5_label1_diag.json --budget_s 1020

The gate (scripts/phase4_gates.py, overfit_labels) left label 1 at Dice 0.0 for B5 after 212 steps. This script
re-runs the same fixed batch with ONE thing changed per variant and records, every 10 steps, the loss parts and
what the label-1 voxels are predicted as. It is a diagnosis; it does not write to gates.json and cannot turn the
gate into a pass.

    kl_off : KL weight held at 0 (tests the loss weighting / the gate's fast KL ramp)
    lr_3e-3: gate schedule, learning rate 3e-3 instead of 1e-3 (tests the learning rate)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase4_gates import (
    OVERFIT_DICE,
    OVERFIT_LR,
    STEPS_PER_EPOCH,
    compute_loss,
    fixed_batch,
    forward,
    fresh,
    label_dice,
    load_config,
)

VARIANTS = {"kl_off": {"lr": OVERFIT_LR, "kl": False}, "lr_3e-3": {"lr": 3e-3, "kl": True}}


def run(variant: str, budget_s: float) -> dict:
    v = VARIANTS[variant]
    cfg = load_config("shared_b5_acf_evidential")
    b = fixed_batch(cfg)
    model = fresh(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=v["lr"], weight_decay=0.0)
    is1 = b["label"][:, 0] == 1
    res = {"variant": v, "label1_voxels": int(is1.sum()), "curve": []}
    t0, step, passed = time.perf_counter(), 0, False
    while time.perf_counter() - t0 < budget_s:
        step += 1
        model.train()
        opt.zero_grad(set_to_none=True)
        epoch = 1 + (step - 1) // STEPS_PER_EPOCH if v["kl"] else 0
        loss, parts = compute_loss(forward(model, b), b["label"], cfg, epoch=epoch)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            model.eval()
            with torch.no_grad():
                ev = forward(model, b)
            pred = ev.argmax(1)
            d = label_dice(pred, b["label"])
            row = {"step": step, "loss": round(float(loss), 5), "parts": parts, "dice": d,
                   "label1_voxels_predicted_as": [int((pred[is1] == k).sum()) for k in (0, 1, 2)],
                   "voxels_predicted_1_anywhere": int((pred == 1).sum()),
                   "mean_evidence_in_label1_voxels": [round(float(ev[:, k][is1].mean()), 4) for k in (0, 1, 2)]}
            res["curve"].append(row)
            print(variant, json.dumps(row), flush=True)
            if min(d.values()) > OVERFIT_DICE:
                passed = True
                break
    res.update(steps=step, seconds=round(time.perf_counter() - t0, 1), both_labels_above_threshold=passed)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--budget_s", type=float, default=1020.0)
    ap.add_argument("--only", default=",".join(VARIANTS))
    a = ap.parse_args()
    out = json.loads(a.out.read_text()) if a.out.exists() else {}
    for name in a.only.split(","):
        out[name] = run(name, a.budget_s)
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
