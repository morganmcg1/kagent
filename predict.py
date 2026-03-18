"""Generate predictions on the hidden test set.

Run:
  uv run predict.py --checkpoint models/model-<id>/checkpoint.pt
"""

import json
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import yaml
from tqdm import tqdm

from data import X_DIM

PREDICTIONS_DIR = Path("/mnt/new-pvc/predictions")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits")


@dataclass
class Config:
    """Generate test predictions from a trained checkpoint."""
    checkpoint: str  # path to best model checkpoint
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None  # kaggler name for output path
    batch_size: int = 4


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

# Load model
checkpoint_dir = Path(cfg.checkpoint).parent
with open(checkpoint_dir / "config.yaml") as f:
    model_config = yaml.safe_load(f)

from train import Transolver
model = Transolver(**model_config).to(device)
model.load_state_dict(torch.load(cfg.checkpoint, map_location=device, weights_only=True))
model.eval()
print(f"Loaded model from {cfg.checkpoint}")

# Load stats
with open(splits_dir / "stats.json") as f:
    stats_data = json.load(f)
x_mean = torch.tensor(stats_data["x_mean"], dtype=torch.float32, device=device)
x_std = torch.tensor(stats_data["x_std"], dtype=torch.float32, device=device)
y_mean = torch.tensor(stats_data["y_mean"], dtype=torch.float32, device=device)
y_std = torch.tensor(stats_data["y_std"], dtype=torch.float32, device=device)

# Load test inputs
test_files = sorted((splits_dir / "test").glob("*.pt"))
print(f"Test samples: {len(test_files)}")

# Run inference
predictions = []
with torch.no_grad():
    for i in tqdm(range(0, len(test_files), cfg.batch_size), desc="Predicting"):
        batch_files = test_files[i:i + cfg.batch_size]
        samples = [torch.load(f, weights_only=True) for f in batch_files]
        xs = [s["x"] for s in samples]

        max_n = max(x.shape[0] for x in xs)
        B = len(xs)
        x_pad = torch.zeros(B, max_n, X_DIM, device=device)
        for j, x in enumerate(xs):
            x_pad[j, :x.shape[0]] = x.to(device)

        pred_norm = model({"x": (x_pad - x_mean) / x_std})["preds"]
        pred = pred_norm * y_std + y_mean

        for j, x in enumerate(xs):
            predictions.append(pred[j, :x.shape[0]].cpu())

# Save
agent_name = cfg.agent or "unknown"
run_id = checkpoint_dir.name
output_dir = PREDICTIONS_DIR / agent_name / run_id
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "predictions.pt"
torch.save(predictions, output_path)
print(f"Saved {len(predictions)} predictions to {output_path}")
