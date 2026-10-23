"""Overfit one batch: a model that cannot memorise 2 crops has a broken pipeline.

    python scripts/overfit_one_batch.py --kind acf_softmax

Draws crops from the synthetic cohort that contain tumour, trains on that one batch only
(no validation, no checkpoints) and passes when foreground Dice on it reaches the threshold.
This is a wiring check (loss reduction, shapes that broadcast silently, detached gradients),
not a result. Exit code 0 = PASS.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from monai.data import DataLoader  # noqa: E402

from src.data.dual_cache import build_dual_datasets  # noqa: E402
from src.data.seed import seed_everything  # noqa: E402
from src.eval.evaluate_cases import call_model  # noqa: E402
from src.training.run import build_model, compute_loss  # noqa: E402
from src.training.smoke import SMOKE_KINDS, smoke_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="acf_softmax", choices=SMOKE_KINDS)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--max-iters", type=int, default=150)
    ap.add_argument("--dice-threshold", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-2)
    a = ap.parse_args()
    work = Path(a.workdir or tempfile.mkdtemp(prefix="acf_overfit_"))
    cfg = smoke_config(work, a.kind)
    seed_everything(cfg["seed"])
    ds = build_dual_datasets(cfg)["train"]
    ds.set_epoch(1)
    loader = DataLoader(ds, batch_size=cfg["train"]["loader_batch_size"], shuffle=True, num_workers=0)
    batch = None
    for _ in range(30):
        b = next(iter(loader))
        fg = (b["label"] > 0).flatten(1).sum(1) > 20
        if int(fg.sum()) >= 2:
            batch = {k: (v[fg][:2] if torch.is_tensor(v) else v) for k, v in b.items()}
            break
    if batch is None:
        print("FAIL: could not draw 2 crops with tumour")
        return 1
    print(f"batch: ct {tuple(batch['ct'].shape)} pet {tuple(batch['pet'].shape)} fg voxels {int((batch['label'] > 0).sum())}")
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    t0, best = time.perf_counter(), 0.0
    lab = batch["label"]
    for it in range(1, a.max_iters + 1):
        opt.zero_grad()
        out = call_model(model, batch["ct"], batch["pet"], batch["cond_ct"], batch["cond_pet"],
                         batch["ct_spacing"], batch["pet_spacing"])
        loss, _ = compute_loss(out, lab, cfg, epoch=10)
        loss.backward()
        opt.step()
        pred = out.argmax(1, keepdim=True)
        inter = ((pred == 1) & (lab == 1)).sum().item()
        dice = 2 * inter / max(((pred == 1).sum() + (lab == 1).sum()).item(), 1)
        best = max(best, dice)
        if it % 10 == 0 or dice >= a.dice_threshold:
            print(f"  iter {it:4d} loss {loss.item():.4f} fg_dice {dice:.4f}  ({time.perf_counter() - t0:.0f}s)")
        if dice >= a.dice_threshold:
            print(f"PASS: fg dice {dice:.4f} >= {a.dice_threshold} at iter {it} (pipeline check only, not a result)")
            return 0
    print(f"FAIL: peak fg dice {best:.4f} < {a.dice_threshold} after {a.max_iters} iters")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
