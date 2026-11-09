"""GPU timing of B1 and B4 on SYNTHETIC random tensors at the real patch size and batch. No HECKTOR data, no cache,
no Drive. Prints ONE plain-text block to paste back.

    python scripts/colab_timing.py 

Per variant (B1 unet3d, B4 ACF; each in fp32 and with mixed precision): 5 warm-up micro-steps, then 20 timed
micro-steps of forward + backward (+ AdamW step every `accum_steps` micro-steps, exactly as src/training/run.py),
batch = train.micro_batch_size at patch.spatial_size from configs/shared_b1_unet_resample.yaml and
configs/shared_b4_acf_cond.yaml. Peak VRAM comes from torch.cuda.max_memory_allocated / reserved.

Hours per 100 epochs = 100 x micro-steps per epoch x max(GPU seconds per step, loader seconds per fetch) / 3600,
which assumes the loader overlaps the GPU. LOADER_S_PER_FETCH holds seconds per fetch (2 patches = one micro-batch)
MEASURED on the local laptop (scripts/measure_data_rate.py, real .acf cache, 24 timed fetches per setting): the "1 worker"
entry is the measured 0-worker (main-process) rate used as a proxy, "2" and "3" were measured with 2 and 3 DataLoader
workers. The Colab CPU may be faster or slower. Micro-steps per epoch = train cases x patches per case / batch.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.models.acf_shared import build_acf  # noqa: E402
from src.models.unet3d import build_unet3d  # noqa: E402
from src.training.losses import dice_ce_loss  # noqa: E402

# seconds per fetch measured locally: B1 = on-the-fly resampled path, B4 = native path with the PET-scale sidecar
LOADER_S_PER_FETCH = {"b1": {1: 1.990, 2: 0.742, 3: 0.568}, "b4": {1: 0.800, 2: 0.304, 3: 0.213}}
N_TRAIN_CASES = 592   # configs/splits/hecktor2026_seed42.json (train)
VARIANTS = [("B1 (CT-only U-Net, resampled)", "shared_b1_unet_resample", "b1"),
            ("B4 (ACF, native spacing)", "shared_b4_acf_cond", "b4")]


def time_variant(cfg: dict, kind: str, amp: bool, warmup: int, steps: int, dev) -> dict:
    t, patch = cfg["train"], cfg["patch"]["spatial_size"]
    batch, accum, k = t["micro_batch_size"], t["accum_steps"], cfg["model"]["out_channels"]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    model = (build_unet3d(cfg) if kind == "b1" else build_acf(cfg)).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    image = torch.rand(batch, 2, *patch, device=dev)
    label = torch.randint(0, k, (batch, 1, *patch), device=dev).float()
    cond = torch.randn(batch, 3, device=dev)
    spacing = torch.tensor([[0.9766, 0.9766, 3.27]], device=dev).repeat(batch, 1)
    out = {"params": sum(p.numel() for p in model.parameters()), "batch": batch, "patch": list(patch)}
    times = []
    try:
        for i in range(warmup + steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", enabled=amp):
                logits = model(image[:, :1]) if kind == "b1" else model(image, cond, spacing)
            loss = dice_ce_loss(logits.float(), label, 1.0, 1.0, 1.0)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - t0)
        out.update(step_mean=statistics.mean(times), step_median=statistics.median(times), step_min=min(times),
                   step_std=statistics.pstdev(times), finite=bool(torch.isfinite(loss).item()))
    except torch.cuda.OutOfMemoryError:
        out["oom"] = True
    out["peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 2**20
    out["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
    del model, opt, image, label
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    L = ["=== PASTE FROM HERE ==="]
    if not torch.cuda.is_available():
        L += ["NO CUDA GPU AVAILABLE in this runtime (torch %s). Nothing was timed." % torch.__version__, "=== END ==="]
        print("\n".join(L))
        return
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        drv = "n/a"
    L += [f"GPU: {props.name} | VRAM total {total / 2**30:.1f} GiB, free at start {free / 2**30:.1f} GiB | driver {drv} | "
          f"torch {torch.__version__} | CUDA {torch.version.cuda}",
          f"synthetic random tensors, {a.warmup} warm-up + {a.steps} timed micro-steps per variant; no data, no cache, no Drive", ""]
    results = []
    for title, name, kind in VARIANTS:
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml").read_text())
        n_micro = N_TRAIN_CASES * cfg["patch"]["num_samples"] / cfg["train"]["micro_batch_size"]
        rates = LOADER_S_PER_FETCH[kind]
        for amp in (False, True):
            r = time_variant(cfg, kind, amp, a.warmup, a.steps, dev)
            tag = f"{title}, {'mixed precision' if amp else 'fp32'}"
            if r.get("oom"):
                L += [f"{tag}: OUT OF MEMORY at batch {r['batch']}, patch {r['patch']} (peak allocated {r['peak_alloc_mb']:.0f} MB)"]
                continue
            L += [f"{tag}: batch {r['batch']}, patch {r['patch']}, params {r['params']:,}",
                  f"  seconds per micro-step: mean {r['step_mean']:.3f}, median {r['step_median']:.3f}, min {r['step_min']:.3f}, std {r['step_std']:.3f}",
                  f"  peak VRAM: allocated {r['peak_alloc_mb']:.0f} MB, reserved {r['peak_reserved_mb']:.0f} MB; loss finite: {r['finite']}"]
            hrs = lambda s: 100 * n_micro * s / 3600  # noqa: E731
            L += [f"  hours per 100 epochs ({n_micro:.0f} micro-steps per epoch): GPU only {hrs(r['step_mean']):.2f} h"]
            for w in (1, 2, 3):
                eff = max(r["step_mean"], rates[w])
                L += [f"    {w} worker(s), loader {rates[w]:.3f} s/fetch (measured locally) -> {hrs(eff):.2f} h "
                      f"({'data-bound' if rates[w] > r['step_mean'] else 'GPU-bound'})"]
            results.append((tag, r))
    L += ["", "Assumptions: the loader overlaps the GPU; loader rates were measured on the local laptop, not on this runtime "
              "(the 1-worker figure is the measured 0-worker rate).", "=== END ==="]
    print("\n".join(L))


if __name__ == "__main__":
    main()
