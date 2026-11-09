"""Forward FLOPs and parameter counts of every baseline config at its configured patch size (batch 1, no training).

    python scripts/flops_all.py --out results/timing/flops_all.json

One model at a time, torch.no_grad, thread cap 2. FLOPs come from torch FlopCounterMode (a count, not a time).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from src.eval.evaluate_shared import call_model  # noqa: E402
from src.training.run import build_model  # noqa: E402

CONFIGS = ["shared_b1_unet_resample", "shared_b2_dual_resample", "shared_b2b_dual_matched",
           "shared_b3_acf_zeroed", "shared_b4_acf_cond", "shared_b5_acf_evidential"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    rows = {}
    for name in CONFIGS:
        cfg = yaml.safe_load(Path(f"configs/{name}.yaml").read_text())
        p = cfg["patch"]["spatial_size"]
        model = build_model(cfg).eval()
        x = torch.randn(1, 2, *p)
        cond, sp = torch.zeros(1, 3), torch.tensor([[1.0, 1.0, 3.0]])
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            call_model(model, x, cond, sp)
        rows[name] = {"patch": p, "params": sum(q.numel() for q in model.parameters()),
                      "flops_forward_per_patch": int(fc.get_total_flops())}
        print(name, rows[name], flush=True)
        del model, x
        gc.collect()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
