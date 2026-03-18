"""Score predictions against hidden test ground truth.

Organizer-only. Reads from .test_gt/ directory.

Run:
  uv run score.py --predictions <path>/predictions.pt
"""

from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch

SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits")


@dataclass
class Config:
    """Score predictions against hidden test ground truth."""
    predictions: str  # path to kaggler's predictions.pt
    gt_dir: str = str(SPLITS_DIR / ".test_gt")


cfg = sp.parse(Config)

preds = torch.load(cfg.predictions, map_location="cpu", weights_only=True)
gt_files = sorted(Path(cfg.gt_dir).glob("*.pt"))
assert len(preds) == len(gt_files), f"Count mismatch: {len(preds)} vs {len(gt_files)}"

# Accumulate per-domain metrics
domains: dict[str, dict] = {}

for i, gt_file in enumerate(gt_files):
    gt = torch.load(gt_file, map_location="cpu", weights_only=False)
    pred_y = preds[i]
    true_y = gt["y"]
    is_surface = gt["is_surface"]
    domain = gt["domain"]

    assert pred_y.shape == true_y.shape, f"Sample {i}: {pred_y.shape} vs {true_y.shape}"

    err = (pred_y - true_y).abs()
    if domain not in domains:
        domains[domain] = {"mae_surf": torch.zeros(3), "n_surf": 0,
                           "mae_vol": torch.zeros(3), "n_vol": 0}
    d = domains[domain]
    d["mae_surf"] += (err * is_surface.unsqueeze(-1)).sum(0)
    d["n_surf"] += is_surface.sum().item()
    d["mae_vol"] += (err * (~is_surface).unsqueeze(-1)).sum(0)
    d["n_vol"] += (~is_surface).sum().item()

# Print results
total_surf = torch.zeros(3)
total_vol = torch.zeros(3)
total_n_surf = total_n_vol = 0

header = "| Domain | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p | mae_vol_Ux | mae_vol_Uy |"
sep = "|--------|-----------|-------------|-------------|----------|-----------|-----------|"
print(header)
print(sep)

for domain in sorted(domains):
    d = domains[domain]
    s = d["mae_surf"] / max(d["n_surf"], 1)
    v = d["mae_vol"] / max(d["n_vol"], 1)
    print(f"| {domain} | {s[2]:.2f} | {s[0]:.2f} | {s[1]:.2f} | {v[2]:.2f} | {v[0]:.2f} | {v[1]:.2f} |")
    total_surf += d["mae_surf"]
    total_vol += d["mae_vol"]
    total_n_surf += d["n_surf"]
    total_n_vol += d["n_vol"]

s = total_surf / max(total_n_surf, 1)
v = total_vol / max(total_n_vol, 1)
print(f"| **OVERALL** | {s[2]:.2f} | {s[0]:.2f} | {s[1]:.2f} | {v[2]:.2f} | {v[0]:.2f} | {v[1]:.2f} |")
