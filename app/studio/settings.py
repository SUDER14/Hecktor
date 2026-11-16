"""Where the Studio reads from. Paths only; nothing here opens a patient file.

The host is a constant, not a setting: the HECKTOR agreement limits use to the VIT site and bars third-party
access, so the server must never be reachable from another machine. Tests assert the value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOST = "127.0.0.1"
DEFAULT_PORT = 8765


@dataclass(frozen=True)
class Settings:
    cache_dir: Path      # <case>.acf files and pet_scale.json (src/data/crop_cache.py), outside the repo
    split_path: Path     # train / val / test patient lists
    meta_csv: Path       # per-case scan table written by scripts/crop_scan.py (whole-scan PET maximum)
    port: int = DEFAULT_PORT


def load_settings(env=None) -> Settings:
    """Defaults follow docs/HANDOVER.md; each can be overridden by an environment variable."""
    env = os.environ if env is None else env
    return Settings(
        cache_dir=Path(env.get("ACF_STUDIO_CACHE_DIR", "D:/hecktor_cache")),
        split_path=Path(env.get("ACF_STUDIO_SPLIT", ROOT / "configs/splits/hecktor2026_seed42.json")),
        meta_csv=Path(env.get("ACF_STUDIO_META_CSV", ROOT / "results/crop/crop_scan.csv")),
        port=int(env.get("ACF_STUDIO_PORT", DEFAULT_PORT)),
    )
