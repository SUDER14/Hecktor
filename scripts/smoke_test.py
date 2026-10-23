"""End-to-end CPU smoke test: synthetic cohort -> cache -> train 2 epochs -> checkpoint ->
resume -> sliding-window inference with NIfTI output -> metrics, for B1, B2, ACF and ACF+evidential.

    python scripts/smoke_test.py            # prints PASS/FAIL per stage and wall-clock

It checks that the pipeline runs and resumes; it produces no result worth reporting.
Typically completes in a few minutes on a laptop CPU (the time is printed, not assumed).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.data.conditioning import load_stats  # noqa: E402
from src.data.dual_cache import build_dual_datasets  # noqa: E402
from src.eval.evaluate_cases import evaluate_dataset, summarize  # noqa: E402
from src.training.run import build_model, train  # noqa: E402
from src.training.smoke import SMOKE_KINDS, smoke_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--kinds", nargs="*", default=list(SMOKE_KINDS))
    a = ap.parse_args()
    work = Path(a.workdir or tempfile.mkdtemp(prefix="acf_smoke_"))
    t_all, failed = time.perf_counter(), []
    for kind in a.kinds:
        t0 = time.perf_counter()
        try:
            cfg = smoke_config(work, kind, n_epochs=1)
            run = train(cfg)
            ckpts = sorted((run / "checkpoints").glob("epoch_*.pt"))
            assert ckpts and (run / "metrics.csv").exists() and (run / "env.json").exists()
            cfg["train"]["n_epochs"] = 2                      # resume must continue at epoch 2, not restart
            train(cfg, resume=True)
            rows = (run / "metrics.csv").read_text().strip().splitlines()
            assert len(rows) == 1 + 2, f"expected 2 epoch rows after resume, got {len(rows) - 1}"
            ds = build_dual_datasets(cfg)
            model = build_model(cfg)
            ck = torch.load(sorted((run / "checkpoints").glob("epoch_*.pt"))[-1], map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model_state"])
            out = run / "pred"
            df = evaluate_dataset(model, ds["test"], cfg, load_stats(cfg["conditioning"]["stats_path"]), "cpu", out,
                                  save_maps=True, max_cases=2)
            assert len(list(out.glob("*_pred.nii.gz"))) == len(df) and len(list(out.glob("*_uncertainty.nii.gz"))) == len(df)
            print(f"[PASS] {kind:15s} {time.perf_counter() - t0:6.1f}s  cases={summarize(df)['n_cases']}")
        except Exception as e:  # noqa: BLE001
            failed.append(kind)
            print(f"[FAIL] {kind}: {type(e).__name__}: {e}")
    print(f"total {time.perf_counter() - t_all:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
