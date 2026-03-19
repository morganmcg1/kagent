"""Train a CFD surrogate model.

Template — fill in your model, loss, optimizer, and training loop.
The data loading, W&B setup, and validation metric logging are provided.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from viz import visualize


# ---------------------------------------------------------------------------
# YOUR MODEL, LOSS, OPTIMIZER — everything is up to you.
#
# Model contract:
#   Input:  {"x": tensor [B, N, 24]}  (normalized features)
#   Output: {"preds": tensor [B, N, 3]}  (predicted Ux, Uy, p in normalized space)
# ---------------------------------------------------------------------------

raise NotImplementedError("Write your model, loss, and training loop. Remove this line.")


# ---------------------------------------------------------------------------
# Config + data loading (you can modify the config)
# ---------------------------------------------------------------------------

MAX_TIMEOUT = 30.0  # minutes — do not increase


@dataclass
class Config:
    batch_size: int = 4
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, **loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}


# ---------------------------------------------------------------------------
# W&B setup (do not remove — required for consistent metric tracking)
# ---------------------------------------------------------------------------

# model = ...  # your model here
n_params = sum(p.numel() for p in model.parameters())

run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
    project=os.environ.get("WANDB_PROJECT", "kagent-v1"),
    group=cfg.wandb_group,
    name=cfg.wandb_name,
    tags=[cfg.agent] if cfg.agent else [],
    config={**asdict(cfg), "n_params": n_params,
            "train_samples": len(train_ds),
            "val_samples": {k: len(v) for k, v in val_splits.items()}},
    mode=os.environ.get("WANDB_MODE", "online"),
)

wandb.define_metric("global_step")
wandb.define_metric("train/*", step_metric="global_step")
wandb.define_metric("val/*", step_metric="global_step")
for _name in VAL_SPLIT_NAMES:
    wandb.define_metric(f"{_name}/*", step_metric="global_step")

model_dir = Path(f"models/model-{run.id}")
model_dir.mkdir(parents=True)
model_path = model_dir / "checkpoint.pt"


# ---------------------------------------------------------------------------
# YOUR TRAINING LOOP HERE
#
# You get: train_loader, val_loaders, stats, model, device
# You must: log the required W&B metrics (see README.md)
#
# Data from loader: (x, y, is_surface, mask)  — all [B, N, ...]
#   x:          [B, N, 24] float32 — raw features (normalize with stats)
#   y:          [B, N, 3]  float32 — targets in physical units
#   is_surface: [B, N]     bool    — surface node mask
#   mask:       [B, N]     bool    — valid node mask (padding is False)
#
# Normalization:
#   x_norm = (x - stats["x_mean"]) / stats["x_std"]
#   y_norm = (y - stats["y_mean"]) / stats["y_std"]
#   pred_phys = pred_norm * stats["y_std"] + stats["y_mean"]
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation (do not modify — ensures consistent metrics across all kagglers)
# Call this after each epoch (or however often you validate).
# ---------------------------------------------------------------------------

def validate(model, val_loaders, stats, device, global_step, surf_weight=10.0):
    """Run validation across all splits, log to W&B. Returns mean val/loss."""
    model.eval()
    val_loss_sum = 0.0
    split_metrics: dict[str, dict] = {}

    for split_name, vloader in val_loaders.items():
        val_vol = val_surf = 0.0
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = n_vol = n_vb = 0

        with torch.no_grad():
            for x, y, is_surface, mask in vloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                x = (x - stats["x_mean"]) / stats["x_std"]
                y_norm = (y - stats["y_mean"]) / stats["y_std"]

                pred = model({"x": x})["preds"]
                sq_err = (pred - y_norm) ** 2

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += (sq_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item()
                val_surf += (sq_err * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item()
                n_vb += 1

                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                err = (pred_orig - y).abs()
                mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_surf += surf_mask.sum().item()
                n_vol += vol_mask.sum().item()

        val_vol /= max(n_vb, 1)
        val_surf /= max(n_vb, 1)
        split_loss = val_vol + surf_weight * val_surf
        mae_surf /= max(n_surf, 1)
        mae_vol /= max(n_vol, 1)

        split_metrics[split_name] = {
            f"{split_name}/vol_loss": val_vol,
            f"{split_name}/surf_loss": val_surf,
            f"{split_name}/loss": split_loss,
            f"{split_name}/mae_vol_Ux": mae_vol[0].item(),
            f"{split_name}/mae_vol_Uy": mae_vol[1].item(),
            f"{split_name}/mae_vol_p": mae_vol[2].item(),
            f"{split_name}/mae_surf_Ux": mae_surf[0].item(),
            f"{split_name}/mae_surf_Uy": mae_surf[1].item(),
            f"{split_name}/mae_surf_p": mae_surf[2].item(),
        }
        val_loss_sum += split_loss

    mean_val_loss = val_loss_sum / len(val_loaders)

    metrics = {"val/loss": mean_val_loss, "global_step": global_step}
    for sm in split_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    return mean_val_loss, split_metrics


# --- Final cleanup (call at the end of your training) ---
# wandb.finish()
