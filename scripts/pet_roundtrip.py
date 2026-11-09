"""How much PET detail does the down/up round trip lose? (pet_mode "rederived", 20 cases)

    python scripts/pet_roundtrip.py --data_dir "<zip>" --n 20 --pet_spacing 4 4 4 --out results/pet_roundtrip

For each sampled case: released PET (on the CT grid) -> re-derived coarse grid -> linearly resampled back
to the CT grid, then compared with the released PET (Pearson r, RMSE, max error, and the same inside the
GTV, where SUVmax change matters most). Cases are stratified by centre (>= 1 each, rest proportional),
chosen with a fixed seed. Everything printed is measured on those cases only.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.hecktor2026 import Hecktor2026Adapter  # noqa: E402
from src.data.pet_rederive import rederive_pet, roundtrip_metrics, upsample_to_ct  # noqa: E402
from src.data.zip_reader import load_nifti  # noqa: E402


def stratified_sample(samples: list[dict], n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_c: dict[str, list[dict]] = {}
    for s in samples:
        by_c.setdefault(s["centre"], []).append(s)
    centres = sorted(by_c)
    take = {c: 1 for c in centres}
    extra = n - len(centres)
    weights = np.array([len(by_c[c]) for c in centres], float)
    for c, k in zip(centres, np.floor(weights / weights.sum() * extra).astype(int)):
        take[c] += int(k)
    while sum(take.values()) < n:  # largest remainder goes to the biggest centre
        take[max(centres, key=lambda c: len(by_c[c]) - take[c])] += 1
    out = []
    for c in centres:
        idx = rng.choice(len(by_c[c]), size=min(take[c], len(by_c[c])), replace=False)
        out += [by_c[c][i] for i in sorted(idx)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pet_spacing", type=float, nargs=3, default=[4.0, 4.0, 4.0])
    ap.add_argument("--out", default="results/pet_roundtrip")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "pet_roundtrip.csv"
    done = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()   # resumable: skip finished cases
    rows = done.to_dict("records")
    seen = set(done.get("patient_id", []))
    for s in stratified_sample(Hecktor2026Adapter(a.data_dir).training_samples(), a.n, a.seed):
        if s["patient_id"] in seen:
            continue
        pet_img = nib.as_closest_canonical(load_nifti(s["image"][1]))
        pet = np.asarray(pet_img.dataobj, dtype=np.float32)
        lab = np.asarray(nib.as_closest_canonical(load_nifti(s["label"])).dataobj)
        coarse, caff, info = rederive_pet(pet, pet_img.affine, a.pet_spacing)
        back = upsample_to_ct(coarse, caff, pet.shape, pet_img.affine)
        rows.append({"patient_id": s["patient_id"], "centre": s["centre"],
                     "ct_spacing": [round(float(x), 4) for x in np.diag(pet_img.affine)[:3]],
                     "coarse_spacing": info["coarse_spacing"], "coarse_shape": list(coarse.shape),
                     **roundtrip_metrics(pet, back, lab)})
        print(rows[-1]["patient_id"], {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
        del pet, back, coarse, lab, pet_img
        gc.collect()
        if len(rows) % 5 == 0:   # 20 cases in total, so checkpoint more often than every 20
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pet_roundtrip.csv", index=False)
    num = df.select_dtypes("number")
    summary = {"n_cases": len(df), "pet_spacing": a.pet_spacing, "seed": a.seed,
               "mean": num.mean().to_dict(), "min": num.min().to_dict(), "max": num.max().to_dict()}
    (out / "pet_roundtrip_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("n_cases", "pet_spacing")}), "\n", num.describe().T[["mean", "min", "max"]].round(4))


if __name__ == "__main__":
    main()
