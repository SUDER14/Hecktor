"""Writes a synthetic PET/CT cohort in the HECKTOR 2026 layout, for pipeline and timing checks.

    python scripts/make_synthetic_cohort.py --out data/synthetic_hecktor2026            # 64 mm box
    python scripts/make_synthetic_cohort.py --out data/synthetic_production --synthetic_production

--synthetic_production sizes the cohort for the production patch (configs/b*.yaml, patch 96^3): the
field of view is large enough that every volume holds a 96^3-voxel CT patch (1 mm) and its 24^3 PET
patch (4 mm). The z extent is 288 mm so the 3 mm-slice centres still have 96 slices. Content is
blobs, not anatomy: the output is for measuring data-loading / GPU time and VRAM at real tensor
sizes, never for Dice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import SynthConfig, write_synthetic_hecktor2026  # noqa: E402

PRODUCTION_PATCH_MM = 96.0
# Margin beyond the patch so the sampler has room to place crops; the 1x1x3 mm centres need
# 3x the z extent in mm to reach 96 slices (the fov is per-axis, so z is set separately).
PRODUCTION_FOV_MM = (128.0, 128.0, 288.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n_per_centre", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fov_mm", type=float, nargs=3, default=(96.0, 96.0, 96.0))
    ap.add_argument("--synthetic_production", action="store_true",
                    help="use a field of view that holds the production 96^3 patch (CT 1 mm / PET 4 mm)")
    a = ap.parse_args()
    fov = PRODUCTION_FOV_MM if a.synthetic_production else tuple(a.fov_mm)
    ids = write_synthetic_hecktor2026(a.out, SynthConfig(fov_mm=fov, n_per_centre=a.n_per_centre, seed=a.seed))
    print(f"wrote {len(ids)} cases to {a.out} (fov_mm={fov})")


if __name__ == "__main__":
    main()
