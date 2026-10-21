"""Resample + normalise every volume once and save it to a local cache.

    python scripts/precompute_cache.py --config configs/train_baseline_decathlon_lung.yaml \
        --data_dir data/Task06_Lung --cache_dir cache/decathlon_lung_1mm

Put --cache_dir on the machine's LOCAL disk (Colab: /content/..., not Drive) -- reading
the cache from Drive would reintroduce the I/O cost this removes. Uses the committed
split and conditioning stats; it never creates or changes them (it refuses to run if
either file is missing). Idempotent: valid entries are skipped, so a dead Colab session
can just rerun it. --native skips resampling (ACF's native-spacing path); the original
spacing is stored in every entry either way. Training picks the cache up via the
config's `cache: {dir: ...}` block.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache import build_cache_entry, entry_is_valid, preprocess_signature  # noqa: E402
from src.data.dataset import _expand_to_modality_items, build_samples, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data_dir", required=True, type=Path)
    ap.add_argument("--cache_dir", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--native", action="store_true", help="no resampling (native spacing)")
    ap.add_argument("--image_dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (smoke checks)")
    ap.add_argument("--overwrite", action="store_true", help="rebuild entries even if valid")
    args = ap.parse_args()

    config = load_config(args.config)
    config["data_dir"] = str(args.data_dir)
    for key in (config["split"]["path"], config["conditioning"]["stats_path"]):
        if not Path(key).exists():
            raise SystemExit(f"{key} is missing -- refusing to create splits/stats from here; "
                             "restore the committed file (git checkout) instead")
    samples = build_samples(config)

    items = [it for s in args.splits for it in _expand_to_modality_items(samples[s])]
    items = items[: args.limit]
    t_start, written, nbytes = time.time(), 0, 0
    for i, item in enumerate(items, 1):
        sig = preprocess_signature(config, item["modality"], args.native)
        if not args.overwrite and entry_is_valid(args.cache_dir, item, sig, args.image_dtype):
            print(f"[{i}/{len(items)}] {item['patient_id']}: cached, skipped")
            continue
        t0 = time.time()
        meta = build_cache_entry(item, config, args.cache_dir, native=args.native,
                                 image_dtype=args.image_dtype, seed=config["seed"])
        size = sum(f.stat().st_size for f in args.cache_dir.glob(f"{item['patient_id']}__*"))
        written, nbytes = written + 1, nbytes + size
        print(f"[{i}/{len(items)}] {item['patient_id']}: shape {meta['shape']} "
              f"spacing {meta['original_spacing']} -> {meta['cache_spacing']}  {size / 2**20:.0f} MiB  "
              f"{time.time() - t0:.1f}s", flush=True)
    print(f"done: {written} written, {len(items) - written} skipped, {nbytes / 2**30:.2f} GiB in "
          f"{time.time() - t_start:.0f}s -> {args.cache_dir}")


if __name__ == "__main__":
    main()
