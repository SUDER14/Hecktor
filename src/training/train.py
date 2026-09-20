"""Training loop for Baseline 1 (src/models/baselines.py), and any other
single-modality model later registered in MODEL_BUILDERS below.

Per CLAUDE.md, training runs on Colab Pro (Colab sessions die without
warning), so this loop checkpoints after every epoch (src/training/
checkpoint.py) and can resume from the latest one with --resume. Checkpoints
carry model/optimizer/scaler/scheduler state, the epoch, the best metric so
far, and RNG state for python/numpy/torch -- everything needed for a resumed
run to draw the same sequence of augmentations/dropout as an uninterrupted
one would have. Only the last config["checkpoint"]["keep_last_n"] epoch
checkpoints are kept on disk plus a separate best.pt (checkpoint.py's
prune_checkpoints/save_best_checkpoint), so Drive quota doesn't grow
unbounded over 15-25 experiments.

Effective batch size: config["train"]["micro_batch_size"] patches go through
one forward/backward pass (the physical, VRAM-bound batch); gradients
accumulate over config["train"]["accum_steps"] such passes before the
optimizer steps -- effective batch = micro_batch_size * accum_steps. CLAUDE.md's
stated convention (96^3 patches, batch 2, effective 8) is micro_batch_size=2,
accum_steps=4. The DataLoader itself fetches config["train"]["loader_batch_size"]
volumes at a time, which MONAI's collate flattens to loader_batch_size *
config["patch"]["num_samples"] crops -- that number must divide evenly by
micro_batch_size (checked at runtime, not silently rounded).

Validation runs whole-volume inference via MONAI's SlidingWindowInferer (a
patch-trained model evaluated on patches gives meaningless numbers) and logs
Dice + HD95 (95th-percentile Hausdorff distance, in mm -- computed with the
baseline's resample spacing so it isn't silently 1.0 by default) per epoch.
When config["wandb"]["enabled"] is true, both plus the full run config are
logged to Weights & Biases so 15-25 experiments stay comparable; wandb is
imported lazily so environments without it (and the --overfit-one-batch
smoke test, which never logs) don't need it installed.

--overfit-one-batch is a pipeline smoke test, not a training run: it fetches
one batch once and trains on only that batch for up to --max-iters steps,
with no validation and no checkpointing. It defaults to an aggressive lr
(1e-2, overridable with --lr) rather than the config's training lr, since the
goal is fast memorization to prove the pipeline works, not a usable model --
at the config's real training lr a healthy-but-slow loss curve is easy to
mistake for a broken pipeline simply because it wasn't given enough steps to
converge. It draws a batch, keeps only crops that contain foreground, and passes
when foreground Dice on that batch reaches --dice-threshold (default 0.9): the
raw loss is a poor pass signal because an empty-target crop has a Dice-loss
floor. If the model still cannot fit a single batch at this lr,
something upstream is broken (wrong loss reduction, a shape mismatch that
broadcast silently, a detached gradient) and a real run isn't worth starting
until this passes. Per CLAUDE.md, this script only ever reports numbers it
actually measured in this run.

Usage:
    python -m src.training.train --config configs/train_baseline_decathlon_lung.yaml --overfit-one-batch
    python -m src.training.train --config configs/train_baseline_decathlon_lung.yaml
    python -m src.training.train --config configs/train_baseline_decathlon_lung.yaml --resume
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import yaml
from monai.data import DataLoader as MonaiDataLoader
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.utils import one_hot
from torch.amp import GradScaler, autocast

from src.data.dataset import build_monai_datasets, load_config
from src.data.seed import seed_everything
from src.data.transforms import build_baseline_train_transforms, build_baseline_val_transforms
from src.models.baselines import build_baseline_unet
from src.training.checkpoint import (
    load_latest_valid_checkpoint,
    prune_checkpoints,
    save_best_checkpoint,
    save_checkpoint,
)

MODEL_BUILDERS = {"baseline_unet": build_baseline_unet}
TRANSFORM_BUILDERS = {
    "baseline_unet": (build_baseline_train_transforms, build_baseline_val_transforms),
}


def build_model(config: dict) -> torch.nn.Module:
    name = config["model"]["name"]
    if name not in MODEL_BUILDERS:
        raise ValueError(f"unknown model {name!r}; registered: {list(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](config)


def build_datasets(config: dict) -> dict:
    name = config["model"]["name"]
    if name not in TRANSFORM_BUILDERS:
        raise ValueError(f"unknown model {name!r}; registered: {list(TRANSFORM_BUILDERS)}")
    train_fn, val_fn = TRANSFORM_BUILDERS[name]
    return build_monai_datasets(config, train_fn, val_fn)


def build_loss(config: dict) -> DiceCELoss:
    """Softmax multi-class Dice + CE over raw logits. to_onehot_y=True/softmax=True
    is the combination verified against this MONAI version's DiceCELoss: it
    one-hots the (B,1,...) integer label for the Dice term and squeezes the
    same label to (B,...) class indices for the CE term internally -- the
    label tensor coming out of the transforms pipeline needs no reshaping.

    smooth (default 1.0, applied to both smooth_nr and smooth_dr) matters for
    small targets: with MONAI's 1e-5 default, a crop with no foreground scores
    a Dice loss of ~1 unless the predicted foreground probability is below
    ~1e-11, so the loss has a large floor that no realistic model reaches
    (measured: perfect logits still gave 0.124 on a 4-crop batch with one
    empty crop)."""
    loss_cfg = config.get("loss", {})
    smooth = loss_cfg.get("smooth", 1.0)
    return DiceCELoss(
        include_background=loss_cfg.get("include_background", True),
        to_onehot_y=True,
        softmax=True,
        lambda_dice=loss_cfg.get("lambda_dice", 1.0),
        lambda_ce=loss_cfg.get("lambda_ce", 1.0),
        smooth_nr=smooth,
        smooth_dr=smooth,
    )


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    t = config["train"]
    name = t.get("optimizer", "adamw")
    if name != "adamw":
        raise ValueError(f"unknown optimizer {name!r}; only 'adamw' is implemented")
    return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.01))


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Config-driven, optional. "none" (or an absent "scheduler" block) holds
    train.lr constant, matching the previous behaviour. Stepped once per
    epoch in train(), after the optimizer steps for that epoch -- not
    per-iteration."""
    sched_cfg = config.get("scheduler", {"name": "none"})
    name = sched_cfg.get("name", "none")
    if name in (None, "none"):
        return None
    if name == "cosine":
        t_max = sched_cfg.get("t_max", config["train"]["n_epochs"])
        eta_min = sched_cfg.get("eta_min", 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    raise ValueError(f"unknown scheduler {name!r}")


def resolve_device(config: dict) -> torch.device:
    requested = config["train"].get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _micro_batches(batch: dict, micro_batch_size: int):
    """Splits one DataLoader fetch (loader_batch_size volumes x patch.num_samples
    crops, flattened by MONAI's list_data_collate) into physical micro-batches
    of micro_batch_size patches each, for gradient accumulation."""
    images, labels = batch["image"], batch["label"]
    n = images.shape[0]
    if n % micro_batch_size != 0:
        raise ValueError(
            f"fetched batch of {n} crops is not divisible by "
            f"train.micro_batch_size={micro_batch_size} -- set "
            "train.loader_batch_size and patch.num_samples so their product "
            "divides evenly, rather than silently dropping the remainder."
        )
    for start in range(0, n, micro_batch_size):
        sl = slice(start, start + micro_batch_size)
        yield images[sl], labels[sl]


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, config) -> float:
    t = config["train"]
    micro_batch_size = t["micro_batch_size"]
    accum_steps = t["accum_steps"]
    use_amp = t.get("amp", True) and device.type == "cuda"

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss, n_micro, micro_step = 0.0, 0, 0

    for batch in loader:
        for images, labels in _micro_batches(batch, micro_batch_size):
            images, labels = images.to(device), labels.to(device)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels) / accum_steps
            scaler.scale(loss).backward()
            running_loss += loss.item() * accum_steps
            n_micro += 1
            micro_step += 1
            if micro_step % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

    if micro_step % accum_steps != 0:
        # leftover partial accumulation window at epoch end -- still apply it
        # rather than silently dropping those gradients.
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / max(n_micro, 1)


@torch.no_grad()
def validate(model, val_dataset, loss_fn, device, config) -> dict:
    """Whole-volume inference via SlidingWindowInferer -- a patch-trained
    model evaluated on patches gives meaningless numbers. HD95 is computed in
    mm using config["baseline"]["resample_spacing"], the isotropic spacing
    every baseline volume (train and val) is resampled to; passing it
    explicitly (rather than relying on the metric's spacing=None default,
    which is numerically identical only because that spacing happens to be
    1.0mm) keeps this correct if resample_spacing ever changes."""
    model.eval()
    roi_size = config["patch"]["spatial_size"]
    sw_batch_size = config["train"].get("sw_batch_size", 1)
    n_classes = config["model"]["out_channels"]
    spacing = config["baseline"]["resample_spacing"]
    inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    losses = []
    for item in val_dataset:
        image = item["image"].unsqueeze(0).to(device)
        label = item["label"].unsqueeze(0).to(device)
        logits = inferer(inputs=image, network=model)
        losses.append(loss_fn(logits, label).item())

        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1, keepdim=True)
        pred_oh, label_oh = one_hot(pred, n_classes), one_hot(label, n_classes)
        dice_metric(y_pred=pred_oh, y=label_oh)
        hd95_metric(y_pred=pred_oh, y=label_oh, spacing=spacing)

    mean_dice = dice_metric.aggregate().item()
    mean_hd95 = hd95_metric.aggregate().item()
    dice_metric.reset()
    hd95_metric.reset()
    return {"loss": sum(losses) / len(losses), "dice": mean_dice, "hd95": mean_hd95}


def get_git_info() -> dict:
    """Commit hash of the code producing this run, so every result traces back
    to it. dirty=True means tracked files differ from that commit (untracked
    files are ignored on purpose: a stray notebook or scratch file doesn't
    change what the training code did). commit is None outside a git checkout
    or before the first commit -- treat such a run as untraceable."""
    repo = Path(__file__).resolve().parents[2]

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if commit is not None else None}


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Applies "a.b.c=value" overrides in place (value parsed as YAML, so
    numbers/bools/lists work). Keys must already exist in the config: a typo
    like train.n_epoch would otherwise silently create a new key and the run
    would use the unchanged default. Overrides land in the config that is
    logged to wandb and saved in checkpoints, so they stay traceable."""
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"bad --override {item!r}; expected dotted.key=value")
        dotted, raw = item.split("=", 1)
        *parents, leaf = dotted.split(".")
        node = config
        for k in parents:
            if not isinstance(node, dict) or k not in node:
                raise SystemExit(f"--override {dotted!r}: no such config section {k!r}")
            node = node[k]
        if not isinstance(node, dict) or leaf not in node:
            raise SystemExit(f"--override {dotted!r}: no such config key {leaf!r}")
        node[leaf] = yaml.safe_load(raw)
    return config


def _init_wandb(config: dict, resume_run_id: str | None):
    """Lazily imports wandb so nothing outside this function needs it
    installed unless config["wandb"]["enabled"] is true. resume_run_id, when
    given (read back from a resumed checkpoint's "wandb_run_id"), continues
    logging to the same run instead of starting a new one -- a Colab session
    dying mid-run shouldn't fragment its wandb history."""
    import wandb

    wb_cfg = config.get("wandb", {})
    run = wandb.init(
        project=wb_cfg.get("project", "acf"),
        name=wb_cfg.get("run_name"),
        mode=wb_cfg.get("mode", "online"),
        config={**config, "git": get_git_info()},
        id=resume_run_id,
        resume="allow" if resume_run_id is not None else None,
    )
    return wandb, run


def train(config: dict, resume: bool) -> None:
    seed_everything(config["seed"])
    device = resolve_device(config)
    print(f"device: {device}")
    git_info = get_git_info()
    print(f"git commit: {git_info['commit']}  dirty: {git_info['dirty']}")
    if git_info["commit"] is None or git_info["dirty"]:
        print("! this run is not traceable to a clean commit -- commit (and push) before a run you intend to report")

    datasets = build_datasets(config)
    if len(datasets["train"]) == 0:
        raise SystemExit("train split is empty -- check the split file / config")

    loader = MonaiDataLoader(
        datasets["train"],
        batch_size=config["train"]["loader_batch_size"],
        shuffle=True,
        num_workers=config["train"].get("num_workers", 0),
    )

    model = build_model(config).to(device)
    loss_fn = build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    use_amp = config["train"].get("amp", True) and device.type == "cuda"
    scaler = GradScaler(device=device.type, enabled=use_amp)

    ckpt_cfg = config["checkpoint"]
    ckpt_dir = Path(ckpt_cfg["dir"])
    keep_last_n = ckpt_cfg.get("keep_last_n", 3)
    start_epoch, best_dice, wandb_run_id = 1, -1.0, None
    if resume:
        ckpt_path, ckpt = load_latest_valid_checkpoint(
            ckpt_dir, model, optimizer, scaler, scheduler, map_location=device
        )
        if ckpt_path is not None:
            start_epoch = ckpt["epoch"] + 1
            best_dice = ckpt.get("best_metric", -1.0)
            wandb_run_id = ckpt.get("wandb_run_id")
            print(f"resumed from {ckpt_path} at epoch {start_epoch}")
        else:
            print(f"--resume passed but no checkpoint found in {ckpt_dir}; starting fresh")

    wandb_mod, wandb_run = (None, None)
    if config.get("wandb", {}).get("enabled", False):
        wandb_mod, wandb_run = _init_wandb(config, resume_run_id=wandb_run_id)
        wandb_run_id = wandb_run.id

    n_epochs = config["train"]["n_epochs"]
    val_every = config["train"].get("val_every", 1)

    for epoch in range(start_epoch, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, config)
        if scheduler is not None:
            scheduler.step()
        msg = f"epoch {epoch}/{n_epochs}  train_loss={train_loss:.4f}  ({time.time() - t0:.1f}s)"

        log = {"epoch": epoch, "train_loss": train_loss, "lr": optimizer.param_groups[0]["lr"]}
        is_best = False
        if epoch % val_every == 0 or epoch == n_epochs:
            metrics = validate(model, datasets["val"], loss_fn, device, config)
            is_best = metrics["dice"] > best_dice
            best_dice = max(best_dice, metrics["dice"])
            msg += f"  val_loss={metrics['loss']:.4f}  val_dice={metrics['dice']:.4f}  val_hd95={metrics['hd95']:.4f}"
            log.update({"val_loss": metrics["loss"], "val_dice": metrics["dice"], "val_hd95": metrics["hd95"]})
        print(msg)

        if wandb_mod is not None:
            wandb_mod.log(log, step=epoch)

        epoch_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
        save_checkpoint(
            epoch_path, epoch, model, optimizer, scaler, best_dice, config,
            scheduler=scheduler, extra={"wandb_run_id": wandb_run_id, "git": git_info},
        )
        if is_best:
            save_best_checkpoint(ckpt_dir, epoch_path)
        prune_checkpoints(ckpt_dir, keep_last_n)

    if wandb_run is not None:
        wandb_run.finish()


def overfit_one_batch(
    config: dict, max_iters: int, dice_threshold: float, lr: float | None = None, early_stop: bool = True,
    batch_size: int = 1, diag_path: str = "results/logs/overfit_diag.jsonl",
) -> bool:
    """Pipeline smoke test: fetch one batch, train on only it, and check the
    foreground Dice can approach 1. See module docstring for why this gates a real run.

    lr overrides config["train"]["lr"] for this run only. The config's own lr
    is tuned for real training (conservative, stable over many epochs); an
    overfit-one-batch check wants the opposite -- an aggressive lr that forces
    fast memorization, so a slow-but-healthy loss curve doesn't get mistaken
    for a broken pipeline just because it wasn't given enough steps to
    converge at a training-scale learning rate.
    """
    seed_everything(config["seed"])
    device = resolve_device(config)
    print(f"device: {device}")

    datasets = build_datasets(config)
    if len(datasets["train"]) == 0:
        raise SystemExit("train split is empty -- check the split file / config")

    loader = MonaiDataLoader(
        datasets["train"],
        batch_size=config["train"]["loader_batch_size"],
        shuffle=True,
        num_workers=0,
    )
    images = labels = None
    for attempt in range(1, 21):
        batch = next(iter(loader))
        has_fg = (batch["label"] > 0).flatten(1).any(dim=1)
        if int(has_fg.sum()) >= batch_size:
            images = batch["image"][has_fg][:batch_size].to(device)
            labels = batch["label"][has_fg][:batch_size].to(device)
            print(
                f"drew batch on attempt {attempt}: kept {batch_size} of {int(has_fg.sum())}/{len(has_fg)} "
                "crops containing foreground (empty crops dropped so the test "
                "measures fitting the lesion, not the empty-target Dice floor)"
            )
            break
    if images is None:
        raise SystemExit(f"could not draw a batch with >={batch_size} foreground crops in 20 attempts")
    n_fg = int((labels > 0).sum())
    print(f"batch: {tuple(images.shape)} images, foreground voxels={n_fg}/{labels.numel()}")
    print(
        f"label dtype={labels.dtype} shape={tuple(labels.shape)} unique={labels.unique().tolist()} | "
        f"image min={images.min().item():.3f} max={images.max().item():.3f}"
    )
    label_fg_frac = n_fg / labels.numel()
    Path(diag_path).parent.mkdir(parents=True, exist_ok=True)
    diag_file = open(diag_path, "w")

    model = build_model(config).to(device)
    loss_fn = build_loss(config)
    optimizer_config = dict(config, train={**config["train"], "lr": lr}) if lr is not None else config
    optimizer = build_optimizer(model, optimizer_config)
    print(f"lr: {optimizer_config['train']['lr']}")
    n_classes = config["model"]["out_channels"]
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    model.train()
    final_loss, fg_dice = None, 0.0
    peak_dice, peak_iter = 0.0, 0
    for it in range(1, max_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        log_now = it == 1 or it % 10 == 0 or it == max_iters
        if log_now:
            grad_norms = {
                name: (None if p.grad is None else p.grad.detach().norm().item())
                for name, p in model.named_parameters()
            }
        optimizer.step()
        final_loss = loss.item()

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1, keepdim=True)
            dice_metric(y_pred=one_hot(pred, n_classes), y=one_hot(labels, n_classes))
            fg_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            pred_fg_frac = (pred == 1).float().mean().item()
        if fg_dice > peak_dice:
            peak_dice, peak_iter = fg_dice, it

        if log_now:
            names = list(grad_norms)
            present = {n: g for n, g in grad_norms.items() if g is not None}
            n_none = len(names) - len(present)
            # weights only: a conv bias feeding an InstanceNorm has ~0 gradient by construction
            weights = {n: g for n, g in present.items() if n.endswith("weight")}
            lo = min(weights, key=weights.get) if weights else None
            print(
                f"  iter {it:4d}  loss={final_loss:.6f}  fg_dice={fg_dice:.4f}  "
                f"pred_fg_frac={pred_fg_frac:.6f} (label {label_fg_frac:.6f})  "
                f"gnorm first={grad_norms[names[0]]} last={grad_norms[names[-1]]} "
                f"min_w={weights[lo] if lo else None} ({lo}) no_grad_params={n_none}"
            )
            diag_file.write(json.dumps({
                "iter": it, "loss": final_loss, "fg_dice": fg_dice,
                "pred_fg_frac": pred_fg_frac, "label_fg_frac": label_fg_frac,
                "grad_norm": grad_norms,
            }) + "\n")
            diag_file.flush()

        if fg_dice >= dice_threshold and early_stop:
            print(
                f"\nPASS: foreground dice {fg_dice:.4f} >= {dice_threshold} "
                f"at iter {it} (loss={final_loss:.6f})"
            )
            return True

    print(
        f"\nsummary: final fg_dice={fg_dice:.4f} (iter {max_iters}), "
        f"peak fg_dice={peak_dice:.4f} at iter {peak_iter}, final loss={final_loss:.6f}"
    )
    if peak_dice >= dice_threshold:
        print(f"PASS: peak foreground dice {peak_dice:.4f} >= {dice_threshold}")
        return True
    print(
        f"FAIL: foreground dice never reached {dice_threshold} within {max_iters} "
        "iters. Investigate loss reduction, silently-broadcast shape mismatches, "
        "detached gradients and lr before starting a real training run."
    )
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--overfit-one-batch", action="store_true")
    ap.add_argument("--max-iters", type=int, default=200, help="overfit-one-batch only")
    ap.add_argument("--dice-threshold", type=float, default=0.9, help="overfit-one-batch only: foreground dice needed to pass")
    ap.add_argument(
        "--lr", type=float, default=None,
        help="overfit-one-batch only: overrides config train.lr for this run "
             "(defaults to an aggressive value, not the config's training lr, "
             "since this mode wants fast memorization, not stable training)",
    )
    ap.add_argument(
        "--no-early-stop", action="store_true",
        help="overfit-one-batch only: run all --max-iters and report final and peak dice, for comparing settings",
    )
    ap.add_argument("--overfit-batch-size", type=int, default=1, help="overfit-one-batch only: foreground crops to memorize")
    ap.add_argument(
        "--diag-out", default="results/logs/overfit_diag.jsonl",
        help="overfit-one-batch only: per-layer grad norms every 10 iters, one JSON line each",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --override checkpoint.dir=/content/drive/MyDrive/x "
             "(repeatable; the key must already exist in the config)",
    )
    args = ap.parse_args()

    config = apply_overrides(load_config(args.config), args.override)

    if args.overfit_one_batch:
        lr = args.lr if args.lr is not None else 1e-2
        passed = overfit_one_batch(
            config, args.max_iters, args.dice_threshold, lr=lr, early_stop=not args.no_early_stop,
            batch_size=args.overfit_batch_size, diag_path=args.diag_out,
        )
        raise SystemExit(0 if passed else 1)

    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
