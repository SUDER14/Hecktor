"""Loader dry run: sha256 of the tensors the training loader yields for the first N training cases.

Used to prove that a refactor or cleanup did not change what the model is fed: run it before and after,
the two outputs must be byte-identical. Reads the cache only (never the zip), writes one small JSON.

    python scripts/loader_hash.py --config configs/shared_b4_acf_cond.yaml --n 3 --out results/cleanup/loader_hash_before.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.acf_dataset import build_acf_datasets  # noqa: E402


def _sha(t) -> str:
    a = np.ascontiguousarray(t.numpy())
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ds = build_acf_datasets(cfg)["train"]
    ds.set_epoch(0)
    rows = []
    for i in range(args.n):
        for j, item in enumerate(ds[i]):
            rows.append({"patient_id": item["patient_id"], "patch": j, "shape": list(item["image"].shape),
                         "image": _sha(item["image"]), "label": _sha(item["label"]), "cond": _sha(item["cond"]),
                         "spacing": _sha(item["spacing"])})
    out = {"config": args.config, "n_cases": args.n, "items": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
