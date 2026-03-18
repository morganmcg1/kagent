"""One-time: split raw TandemFoilSet pickles into per-sample .pt files on PVC.

Run on organizer pod before launching kagglers:
  uv run prepare_splits.py

Output on PVC:
  /mnt/new-pvc/datasets/tandemfoil/splits/
  ├── train/000000.pt ...           {x, y, is_surface}
  ├── val_in_dist/000000.pt ...     {x, y, is_surface}
  ├── val_tandem_transfer/...       {x, y, is_surface}
  ├── val_ood_cond/...              {x, y, is_surface}
  ├── val_ood_re/...                {x, y, is_surface}
  ├── test/000000.pt ...            {x, is_surface}  (no y)
  ├── .test_gt/000000.pt ...        {y, is_surface, domain}  (hidden)
  ├── stats.json
  └── meta.json
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing as sp
import torch
from rich.console import Console
from rich.panel import Panel

console = Console()

# --- Constants ---
SEED = 42
SAMPLE_FRACTION = 0.85
DATA_ROOT = Path("/mnt/new-pvc/datasets/tandemfoil")
SURFACE_IDS = (5, 6, 7)  # foil 1 upper, foil 1 lower, foil 2
X_DIM = 24

PICKLE_FILES = [
    "raceCar_single_randomFields.pickle",         # 0: single foil
    "raceCar_randomFields_mgn_Part1.pickle",       # 1: tandem, front=NACA2412
    "raceCar_randomFields_mgn_Part2.pickle",       # 2: tandem, front=NACA6416 → val
    "raceCar_randomFields_mgn_Part3.pickle",       # 3: tandem, front=NACA9412
    "cruise_randomFields_mgn_Part1.pickle",        # 4: cruise, Re=1.475M
    "cruise_randomFields_mgn_Part2.pickle",        # 5: cruise, Re=4.445M → val
    "cruise_randomFields_mgn_Part3.pickle",        # 6: cruise, Re=802K
]

VAL_SPLITS = ["val_in_dist", "val_tandem_transfer", "val_ood_cond", "val_ood_re"]


@dataclass
class Args:
    """Prepare TandemFoilSet competition splits."""
    data_root: str = str(DATA_ROOT)  # directory containing raw pickle files
    out_dir: str = str(DATA_ROOT / "splits")  # output directory for splits


# --- Raw data helpers ---

def load_pickle(path: Path) -> list:
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_naca(s: str) -> tuple[float, float, float]:
    if len(s) == 4 and s.isdigit():
        return int(s[0]) / 9.0, int(s[1]) / 9.0, int(s[2:]) / 24.0
    return 0.0, 0.0, 0.0


def preprocess(sample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw PyG sample → (x [N,24], y [N,3], is_surface [N]).

    x layout: [pos(2), saf(2), dsdf(8), is_surface(1), log_Re(1),
               AoA0_rad(1), NACA0(3), AoA1_rad(1), NACA1(3), gap(1), stagger(1)]
    """
    n = sample.pos.shape[0]

    is_surface = torch.zeros(n, dtype=torch.bool)
    for sid in SURFACE_IDS:
        is_surface |= sample.boundary == sid

    aoa = sample.AoA
    aoa0 = float(aoa[0]) if isinstance(aoa, list) else float(aoa)
    aoa1 = float(aoa[1]) if isinstance(aoa, list) else 0.0

    naca0 = parse_naca(sample.NACA[0])
    naca1 = parse_naca(sample.NACA[1]) if len(sample.NACA) > 1 else (0.0, 0.0, 0.0)

    gap_val = getattr(sample, "gap", None)
    stagger_val = getattr(sample, "stagger", None)

    x = torch.cat([
        sample.pos.float(),                                             # 2
        sample.saf.float(),                                             # 2
        sample.dsdf.float(),                                            # 8
        is_surface.float().unsqueeze(1),                                # 1
        torch.full((n, 1), math.log(float(sample.flowState["Re"]))),    # 1
        torch.full((n, 1), aoa0 * math.pi / 180.0),                    # 1
        torch.tensor(naca0, dtype=torch.float32).expand(n, 3),          # 3
        torch.full((n, 1), aoa1 * math.pi / 180.0),                    # 1
        torch.tensor(naca1, dtype=torch.float32).expand(n, 3),          # 3
        torch.full((n, 1), float(gap_val) if gap_val is not None else 0.0),        # 1
        torch.full((n, 1), float(stagger_val) if stagger_val is not None else 0.0),  # 1
    ], dim=1)

    return x, sample.y.float(), is_surface


# --- Subsampling ---

def subsample(idxs: list[int], fraction: float, rng=None) -> tuple[list[int], list[int]]:
    """(kept, excluded) with even spacing. Pass rng to shuffle first."""
    n = max(1, round(len(idxs) * fraction))
    if n >= len(idxs):
        return idxs, []
    if rng is not None:
        arr = np.array(idxs)
        rng.shuffle(arr)
        return arr[:n].tolist(), arr[n:].tolist()
    step = len(idxs) / n
    kept = [idxs[round(i * step)] for i in range(n)]
    excluded = [i for i in idxs if i not in set(kept)]
    return kept, excluded


# --- Split assignment ---

def scan_metadata(pickle_paths: list[Path]):
    """Lightweight metadata scan. Returns (by_file, file_sizes)."""
    by_file: dict[int, list[dict]] = {}
    file_sizes: list[int] = []
    offset = 0

    for fi, path in enumerate(pickle_paths):
        console.print(f"  [{fi}] {path.name}", end="")
        raw = load_pickle(path)
        n = len(raw)
        file_sizes.append(n)
        by_file[fi] = []
        for li, sample in enumerate(raw):
            aoa = sample.AoA
            by_file[fi].append({
                "global_idx": offset + li,
                "aoa0": float(aoa[0]) if isinstance(aoa, list) else float(aoa),
                "gap": float(sample.gap) if getattr(sample, "gap", None) is not None else None,
                "stagger": float(sample.stagger) if getattr(sample, "stagger", None) is not None else None,
            })
        console.print(f" → {n} samples")
        offset += n
        del raw

    return by_file, file_sizes


def assign_splits(by_file: dict[int, list[dict]]):
    """Assign every sample to exactly one split.

    Returns (splits, domain_groups, test_domains).
    """
    rng = np.random.default_rng(SEED)

    splits: dict[str, list[int]] = {k: [] for k in ["train"] + VAL_SPLITS + ["test"]}
    groups: dict[str, list[int]] = {"racecar_single": [], "racecar_tandem": [], "cruise": []}
    test_domains: dict[int, str] = {}

    def add_test(idxs, domain):
        splits["test"].extend(idxs)
        for i in idxs:
            test_domains[i] = domain

    # File 0: raceCar single → 90/10 train/val_in_dist
    single = [r["global_idx"] for r in by_file[0]]
    kept, excl = subsample(single, SAMPLE_FRACTION, rng=rng)
    add_test(excl, "single")
    n_val = max(1, round(len(kept) * 0.10))
    splits["val_in_dist"].extend(kept[:n_val])
    splits["train"].extend(kept[n_val:])
    groups["racecar_single"].extend(kept[n_val:])

    # Files 1,3: raceCar tandem → train
    for fi in (1, 3):
        idxs = [r["global_idx"] for r in by_file[fi]]
        kept, excl = subsample(idxs, SAMPLE_FRACTION)
        splits["train"].extend(kept)
        groups["racecar_tandem"].extend(kept)
        add_test(excl, "tandem_known")

    # File 2: raceCar tandem Part2 → val_tandem_transfer
    idxs = [r["global_idx"] for r in by_file[2]]
    kept, excl = subsample(idxs, SAMPLE_FRACTION)
    splits["val_tandem_transfer"].extend(kept)
    add_test(excl, "tandem_transfer")

    # Files 4,6: cruise Part1+3 → frontier 20% to val_ood_cond, rest to train
    cruise_recs = by_file[4] + by_file[6]
    cruise_all = list(range(len(cruise_recs)))
    keep_idxs, excl_idxs = subsample(cruise_all, SAMPLE_FRACTION)
    kept_recs = [cruise_recs[i] for i in keep_idxs]
    for i in excl_idxs:
        add_test([cruise_recs[i]["global_idx"]], "cruise_known")

    feats = np.array([[r["aoa0"], r["gap"], r["stagger"]] for r in kept_recs], dtype=np.float64)
    feat_min, feat_max = feats.min(0), feats.max(0)
    feat_range = np.where(feat_max - feat_min > 0, feat_max - feat_min, 1.0)
    normed = (feats - feat_min) / feat_range
    dists = np.linalg.norm(normed - normed.mean(0), axis=1)
    n_frontier = max(1, round(len(dists) * 0.20))
    frontier = set(np.argsort(-dists)[:n_frontier].tolist())

    for i, rec in enumerate(kept_recs):
        gidx = rec["global_idx"]
        if i in frontier:
            splits["val_ood_cond"].append(gidx)
        else:
            splits["train"].append(gidx)
            groups["cruise"].append(gidx)

    # File 5: cruise Part2 → val_ood_re
    idxs = [r["global_idx"] for r in by_file[5]]
    kept, excl = subsample(idxs, SAMPLE_FRACTION)
    splits["val_ood_re"].extend(kept)
    add_test(excl, "cruise_ood_re")

    return splits, groups, test_domains


# --- Save helpers ---

def global_to_file_local(global_idx: int, file_sizes: list[int]) -> tuple[int, int]:
    offset = 0
    for fi, n in enumerate(file_sizes):
        if global_idx < offset + n:
            return fi, global_idx - offset
        offset += n
    raise ValueError(f"global_idx {global_idx} out of range")


def save_samples(
    out_dir: Path,
    split_name: str,
    global_indices: list[int],
    pickle_paths: list[Path],
    file_sizes: list[int],
    include_y: bool = True,
    test_domains: dict[int, str] | None = None,
):
    """Preprocess raw samples and save as individual .pt files."""
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = None
    if not include_y:
        gt_dir = out_dir / ".test_gt"
        gt_dir.mkdir(parents=True, exist_ok=True)

    # Group by file for sequential I/O
    by_file: dict[int, list[tuple[int, int, int]]] = {}  # fi → [(seq_idx, local_idx, global_idx)]
    for seq_idx, gidx in enumerate(global_indices):
        fi, li = global_to_file_local(gidx, file_sizes)
        by_file.setdefault(fi, []).append((seq_idx, li, gidx))

    for fi in sorted(by_file):
        console.print(f"    {pickle_paths[fi].name} ({len(by_file[fi])} samples)")
        raw = load_pickle(pickle_paths[fi])
        for seq_idx, li, gidx in by_file[fi]:
            x, y, is_surface = preprocess(raw[li])
            fname = f"{seq_idx:06d}.pt"
            if include_y:
                torch.save({"x": x, "y": y, "is_surface": is_surface}, split_dir / fname)
            else:
                torch.save({"x": x, "is_surface": is_surface}, split_dir / fname)
                domain = test_domains[gidx] if test_domains else "unknown"
                torch.save({"y": y, "is_surface": is_surface, "domain": domain}, gt_dir / fname)
        del raw

    console.print(f"  {split_name}: {len(global_indices)} samples")


def compute_stats(train_dir: Path) -> dict:
    """Two-pass mean/std over training .pt files."""
    files = sorted(train_dir.glob("*.pt"))
    n = len(files)

    console.print(f"  Pass 1/2 (mean) — {n} samples")
    sum_x = torch.zeros(X_DIM, dtype=torch.float64)
    sum_y = torch.zeros(3, dtype=torch.float64)
    total = 0

    for i, f in enumerate(files):
        if i % 200 == 0:
            console.print(f"    {i}/{n}")
        s = torch.load(f, weights_only=True)
        sum_x += s["x"].double().sum(0)
        sum_y += s["y"].double().sum(0)
        total += s["x"].shape[0]

    mean_x = sum_x / total
    mean_y = sum_y / total

    console.print(f"  Pass 2/2 (std) — {n} samples")
    sq_x = torch.zeros(X_DIM, dtype=torch.float64)
    sq_y = torch.zeros(3, dtype=torch.float64)

    for i, f in enumerate(files):
        if i % 200 == 0:
            console.print(f"    {i}/{n}")
        s = torch.load(f, weights_only=True)
        sq_x += ((s["x"].double() - mean_x) ** 2).sum(0)
        sq_y += ((s["y"].double() - mean_y) ** 2).sum(0)

    std_x = (sq_x / (total - 1)).sqrt().clamp(min=1e-6)
    std_y = (sq_y / (total - 1)).sqrt().clamp(min=1e-6)

    return {
        "x_dim": X_DIM,
        "n_train_samples": n,
        "n_train_nodes": total,
        "x_mean": mean_x.float().tolist(),
        "x_std": std_x.float().tolist(),
        "y_mean": mean_y.float().tolist(),
        "y_std": std_y.float().tolist(),
    }


# --- Main ---

args = sp.parse(Args)
data_root = Path(args.data_root)
out_dir = Path(args.out_dir)
pickle_paths = [data_root / f for f in PICKLE_FILES]

console.rule("Phase 1: Metadata scan + split assignment")
by_file, file_sizes = scan_metadata(pickle_paths)
splits, domain_groups, test_domains = assign_splits(by_file)

console.print()
for k, v in splits.items():
    console.print(f"  {k:30s} {len(v):5d}")

console.rule("Phase 2: Preprocess and save")
for split_name in ["train"] + VAL_SPLITS:
    save_samples(out_dir, split_name, splits[split_name], pickle_paths, file_sizes)

# Shuffle test order so file indices don't reveal source
test_indices = splits["test"].copy()
np.random.default_rng(123).shuffle(test_indices)
save_samples(out_dir, "test", test_indices, pickle_paths, file_sizes,
             include_y=False, test_domains=test_domains)

console.rule("Phase 3: Normalization stats")
stats = compute_stats(out_dir / "train")
with open(out_dir / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)
console.print(f"  Wrote stats.json ({stats['n_train_nodes']} total nodes)")

console.rule("Phase 4: Metadata")
train_gidx_to_seq = {gidx: i for i, gidx in enumerate(splits["train"])}
meta = {
    "x_dim": X_DIM,
    "val_splits": VAL_SPLITS,
    "split_counts": {k: len(v) for k, v in splits.items()},
    "domain_groups": {
        name: sorted(train_gidx_to_seq[gidx] for gidx in idxs)
        for name, idxs in domain_groups.items()
    },
}
with open(out_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
console.print("  Wrote meta.json")

console.rule("Done")
console.print(Panel(
    f"Output: {out_dir}\n"
    f"Train: {stats['n_train_samples']} samples, {stats['n_train_nodes']} nodes\n"
    f"y_mean: {[f'{v:.2f}' for v in stats['y_mean']]}\n"
    f"y_std: {[f'{v:.2f}' for v in stats['y_std']]}",
    title="Summary",
))
