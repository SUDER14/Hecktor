"""Timing probe for one baseline config: ONE training step and one epoch estimate, on whatever device exists.

    python scripts/timing_probe.py --config configs/shared_b1_unet_resample.yaml --out results/timing/b1.json

What is measured (nothing is assumed):
  * data: seconds to produce one training fetch (num_samples patches of one patient) from the real .acf cache,
    and one whole validation volume;
  * model: parameter count, FLOPs of one micro-batch forward and forward+backward (torch FlopCounterMode),
    bytes of tensors saved for backward during that step (autograd saved_tensors_hooks), exact bytes of
    parameters, gradients and AdamW state;
  * one full training step (forward, backward, optimizer step) wall time on the device present, and its
    peak process memory (CPU RSS) -- CUDA peak VRAM only if CUDA exists, otherwise reported as NOT MEASURED;
  * epoch arithmetic from the config: fetches per epoch = number of training cases / loader batch size.
GPU time per epoch is NOT measured on a CPU-only machine; the output gives FLOPs per epoch so a measured GPU
throughput can be applied later. One case at a time, no worker processes.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from src.data.acf_dataset import build_acf_datasets  # noqa: E402
from src.eval.evaluate_shared import call_model  # noqa: E402
from src.training.run import build_model, compute_loss  # noqa: E402


def saved_bytes_during(fn) -> int:
    total = 0

    def pack(t):
        nonlocal total
        total += t.element_size() * t.nelement()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--micro_batch", type=int, default=None, help="override train.micro_batch_size (memory); results are per this size")
    ap.add_argument("--steps", type=int, default=3, help="timed steps after one warm-up step")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.micro_batch:
        cfg["train"]["micro_batch_size"] = a.micro_batch
    t = cfg["train"]
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))   # thread cap
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = psutil.Process()
    res: dict = {"config": a.config, "device": str(dev), "cuda_available": torch.cuda.is_available(),
                 "patch": cfg["patch"]["spatial_size"], "model": cfg["model"]["name"],
                 "threads": torch.get_num_threads()}

    ds = build_acf_datasets(cfg)
    ds["train"].set_epoch(1)
    t0 = time.perf_counter()
    fetch = ds["train"][0]
    res["data_s_per_train_fetch"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    fetch = ds["train"][1]
    res["data_s_per_train_fetch_second"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    vol = ds["val"][0]
    res["data_s_per_val_volume"] = time.perf_counter() - t0
    res["val_volume_shape"] = list(vol["image"].shape[1:])
    res["n_train_cases"], res["n_val_cases"] = len(ds["train"]), len(ds["val"])
    del vol
    gc.collect()

    mb = t["micro_batch_size"]
    res["micro_batch_measured"], res["micro_batch_in_config"] = mb, yaml.safe_load(Path(a.config).read_text())["train"]["micro_batch_size"]
    batch = {k: torch.stack([f[k] for f in fetch[:mb]]) for k in ("image", "label", "cond", "spacing")}
    batch = {k: v.to(dev) for k, v in batch.items()}
    model = build_model(cfg).to(dev)
    res["params"] = sum(p.numel() for p in model.parameters())
    res["param_bytes_fp32"] = res["params"] * 4
    res["adamw_state_bytes_fp32"] = res["params"] * 8
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))

    def step():
        opt.zero_grad(set_to_none=True)
        out = call_model(model, batch["image"], batch["cond"], batch["spacing"])
        loss, _ = compute_loss(out, batch["label"], cfg, epoch=1)
        loss.backward()
        opt.step()
        return loss

    model.train()
    with FlopCounterMode(display=False) as fc:
        out = call_model(model, batch["image"], batch["cond"], batch["spacing"])
    res["flops_forward_microbatch"] = int(fc.get_total_flops())
    res["saved_for_backward_bytes_fp32"] = saved_bytes_during(
        lambda: compute_loss(call_model(model, batch["image"], batch["cond"], batch["spacing"]), batch["label"], cfg, 1)[0].backward())
    del out
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step()   # warm-up
    times = []
    for _ in range(a.steps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    res["step_s_each"] = times
    res["step_s_mean"] = sum(times) / len(times)
    res["final_loss_finite"] = bool(torch.isfinite(loss).item())
    res["process_rss_gb_after"] = proc.memory_info().rss / 2**30
    res["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else "NOT MEASURED (no CUDA on this machine)"

    fetches = res["n_train_cases"] / t["loader_batch_size"]
    patches_per_epoch = fetches * cfg["patch"]["num_samples"]
    res["patches_per_epoch"] = patches_per_epoch
    res["epoch_steps_at_measured_batch"] = patches_per_epoch / mb
    res["epoch_step_time_s_on_this_device"] = res["epoch_steps_at_measured_batch"] * res["step_s_mean"]
    res["epoch_data_time_s_single_process"] = fetches * res["data_s_per_train_fetch"]
    res["epoch_flops_train_estimate"] = res["epoch_steps_at_measured_batch"] * res["flops_forward_microbatch"] * 3  # fwd + ~2x backward (rule of thumb, not measured)
    res["n_epochs_config"], res["val_every"] = t["n_epochs"], t.get("val_every", 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
