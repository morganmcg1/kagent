"""Score predictions against hidden test ground truth.

Organizer-only. Scores a submission and logs results to W&B.

Predictions live at: /mnt/new-pvc/predictions/<agent>/<commit>/predictions.pt
The agent name and commit hash are inferred from the path.

Run:
  uv run score.py --predictions /mnt/new-pvc/predictions/frieren/abc1234/predictions.pt

Score all pending submissions:
  uv run score.py --score_all
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import wandb

SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits")
PREDICTIONS_ROOT = Path("/mnt/new-pvc/predictions")
SCORES_FILE = PREDICTIONS_ROOT / "scores.json"
CHANNELS = ["Ux", "Uy", "p"]


@dataclass
class Config:
    """Score predictions against hidden test ground truth."""
    predictions: str = ""  # path to a single predictions.pt
    score_all: bool = False  # score all unscored submissions in predictions dir
    gt_dir: str = str(SPLITS_DIR / ".test_gt")


def score_predictions(predictions_path: Path, gt_dir: Path) -> dict:
    """Score a predictions.pt file. Returns metrics dict."""
    preds = torch.load(predictions_path, map_location="cpu", weights_only=True)
    gt_files = sorted(gt_dir.glob("*.pt"))
    assert len(preds) == len(gt_files), f"Count mismatch: {len(preds)} vs {len(gt_files)}"

    domains: dict[str, dict] = {}
    for i, gt_file in enumerate(gt_files):
        gt = torch.load(gt_file, map_location="cpu", weights_only=False)
        pred_y, true_y = preds[i], gt["y"]
        is_surface, domain = gt["is_surface"], gt["domain"]
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

    # Compute final metrics
    results = {}
    total_surf, total_vol = torch.zeros(3), torch.zeros(3)
    total_n_surf = total_n_vol = 0

    for domain in sorted(domains):
        d = domains[domain]
        s = d["mae_surf"] / max(d["n_surf"], 1)
        v = d["mae_vol"] / max(d["n_vol"], 1)
        results[f"{domain}/mae_surf_Ux"] = s[0].item()
        results[f"{domain}/mae_surf_Uy"] = s[1].item()
        results[f"{domain}/mae_surf_p"] = s[2].item()
        results[f"{domain}/mae_vol_Ux"] = v[0].item()
        results[f"{domain}/mae_vol_Uy"] = v[1].item()
        results[f"{domain}/mae_vol_p"] = v[2].item()
        total_surf += d["mae_surf"]
        total_vol += d["mae_vol"]
        total_n_surf += d["n_surf"]
        total_n_vol += d["n_vol"]

    s = total_surf / max(total_n_surf, 1)
    v = total_vol / max(total_n_vol, 1)
    results["overall/mae_surf_Ux"] = s[0].item()
    results["overall/mae_surf_Uy"] = s[1].item()
    results["overall/mae_surf_p"] = s[2].item()
    results["overall/mae_vol_Ux"] = v[0].item()
    results["overall/mae_vol_Uy"] = v[1].item()
    results["overall/mae_vol_p"] = v[2].item()
    return results


def print_table(results: dict, agent: str, commit: str):
    """Print markdown results table."""
    print(f"\n## {agent} @ {commit}")
    header = "| Domain | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p | mae_vol_Ux | mae_vol_Uy |"
    sep = "|--------|-----------|-------------|-------------|----------|-----------|-----------|"
    print(header)
    print(sep)
    # Extract domain names
    domain_names = sorted({k.split("/")[0] for k in results if "/" in k})
    for domain in domain_names:
        sp_ = results.get(f"{domain}/mae_surf_p", 0)
        su = results.get(f"{domain}/mae_surf_Ux", 0)
        sv = results.get(f"{domain}/mae_surf_Uy", 0)
        vp = results.get(f"{domain}/mae_vol_p", 0)
        vu = results.get(f"{domain}/mae_vol_Ux", 0)
        vv = results.get(f"{domain}/mae_vol_Uy", 0)
        name = f"**{domain}**" if domain == "overall" else domain
        print(f"| {name} | {sp_:.2f} | {su:.2f} | {sv:.2f} | {vp:.2f} | {vu:.2f} | {vv:.2f} |")


def log_to_wandb(results: dict, agent: str, commit: str):
    """Log scores as a W&B run in the shared project."""
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "kagent-v1"),
        name=f"score/{agent}/{commit}",
        tags=["score", agent],
        config={"agent": agent, "commit": commit},
        job_type="scoring",
    )
    wandb.log({f"test/{k}": v for k, v in results.items()})
    wandb.summary.update({f"test/{k}": v for k, v in results.items()})
    wandb.finish()


def load_scores() -> dict:
    """Load existing scores from PVC."""
    if SCORES_FILE.exists():
        return json.loads(SCORES_FILE.read_text())
    return {}


def save_scores(scores: dict):
    """Save scores to PVC."""
    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCORES_FILE.write_text(json.dumps(scores, indent=2))


def update_leaderboard(scores: dict, repo_dir: str = "/workspace/kagent"):
    """Update leaderboard.md in the repo and push to main."""
    if not scores:
        return

    # Build rows: best submission per agent (lowest overall/mae_surf_p)
    best_per_agent: dict[str, tuple[str, dict]] = {}
    for key, results in scores.items():
        agent, commit = key.split("/", 1)
        surf_p = results.get("overall/mae_surf_p", float("inf"))
        if agent not in best_per_agent or surf_p < best_per_agent[agent][1].get("overall/mae_surf_p", float("inf")):
            best_per_agent[agent] = (commit, results)

    # Sort by overall surface pressure MAE (lower is better)
    ranked = sorted(best_per_agent.items(), key=lambda x: x[1][1].get("overall/mae_surf_p", float("inf")))

    lines = [
        "# Leaderboard",
        "",
        "Ranked by **overall surface pressure MAE** (lower is better).",
        "",
        "| Rank | Agent | Commit | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p | mae_vol_Ux | mae_vol_Uy |",
        "|------|-------|--------|-----------|-------------|-------------|----------|-----------|-----------|",
    ]

    for rank, (agent, (commit, r)) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {agent} | `{commit[:7]}` "
            f"| {r.get('overall/mae_surf_p', 0):.2f} "
            f"| {r.get('overall/mae_surf_Ux', 0):.2f} "
            f"| {r.get('overall/mae_surf_Uy', 0):.2f} "
            f"| {r.get('overall/mae_vol_p', 0):.2f} "
            f"| {r.get('overall/mae_vol_Ux', 0):.2f} "
            f"| {r.get('overall/mae_vol_Uy', 0):.2f} |"
        )

    lines.extend(["", f"*Last updated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*", ""])

    leaderboard_path = Path(repo_dir) / "leaderboard.md"
    leaderboard_path.write_text("\n".join(lines))

    # Commit and push to main
    import subprocess
    git = lambda *args: subprocess.run(["git", "-C", repo_dir] + list(args), capture_output=True, text=True)
    git("config", "user.name", "kagent-organizer")
    git("config", "user.email", "kagent-organizer@kagent")
    git("checkout", "main")
    git("pull", "origin", "main")
    git("add", "leaderboard.md")
    result = git("diff", "--cached", "--quiet")
    if result.returncode != 0:  # there are changes
        git("commit", "-m", "Update leaderboard")
        push = git("push", "origin", "main")
        if push.returncode == 0:
            print(f"  Leaderboard pushed to main ({len(ranked)} agents)")
        else:
            print(f"  Leaderboard push failed: {push.stderr.strip()}")
    else:
        print("  Leaderboard unchanged")


def score_one(predictions_path: Path, gt_dir: Path):
    """Score a single submission."""
    # Infer agent and commit from path: .../predictions/<agent>/<commit>/predictions.pt
    parts = predictions_path.parts
    pred_idx = parts.index("predictions")
    agent = parts[pred_idx + 1]
    commit = parts[pred_idx + 2]

    print(f"Scoring: {agent} @ {commit}")
    results = score_predictions(predictions_path, gt_dir)
    print_table(results, agent, commit)
    log_to_wandb(results, agent, commit)

    # Save to scores.json
    scores = load_scores()
    scores[f"{agent}/{commit}"] = results
    save_scores(scores)
    print(f"Saved to {SCORES_FILE}")


cfg = sp.parse(Config)
gt_dir = Path(cfg.gt_dir)

if cfg.score_all:
    scores = load_scores()
    new_scored = False
    for pred_file in sorted(PREDICTIONS_ROOT.glob("*/*/predictions.pt")):
        parts = pred_file.parts
        pred_idx = parts.index("predictions")
        key = f"{parts[pred_idx + 1]}/{parts[pred_idx + 2]}"
        if key in scores:
            print(f"  Already scored: {key}")
            continue
        score_one(pred_file, gt_dir)
        new_scored = True
    if new_scored:
        update_leaderboard(load_scores())
elif cfg.predictions:
    score_one(Path(cfg.predictions), gt_dir)
else:
    print("Specify --predictions <path> or --score_all")
