# kagent — Development Context

Autonomous ML competition on CFD surrogates, coordinated through GitHub PRs using Claude Code agents as kagglers.

## Coding guidelines

- Simple, readable code. No unnecessary abstractions. This is a research codebase.
- Avoid overly defensive coding. Let errors propagate so we can fix them.
- No placeholder CLI flags that gate real functionality.
- Python 3.12+ conventions.

## Codebase

### kagent infrastructure (the tool)

| File | Purpose |
|------|---------|
| `data.py` | Generic data loader for pre-processed splits |
| `score.py` | Score predictions against hidden ground truth |
| `k8s/` | Orchestration: launch, deployments, entrypoints |
| `instructions/` | Role-specific CLAUDE.md files for organizer/kaggler pods |

### Competition-specific (user provides per competition)

| File | Purpose |
|------|---------|
| `prepare_splits.py` | One-time data preprocessing + split (run before launch) |
| `train.py` | Baseline model for kagglers to iterate on |
| `predict.py` | Baseline prediction script |
| `viz.py` | Visualization helpers |
| `program.md` | Problem description, metrics, rules |

## Data layout on PVC

Pre-processed splits live at `/mnt/new-pvc/datasets/tandemfoil/splits/`:
```
splits/
├── train/000000.pt ...           {x, y, is_surface}
├── val_in_dist/000000.pt ...     {x, y, is_surface}
├── val_tandem_transfer/...
├── val_ood_cond/...
├── val_ood_re/...
├── test/000000.pt ...            {x, is_surface}  (no y)
├── .test_gt/000000.pt ...        {y, is_surface, domain}  (hidden)
├── stats.json                    normalization stats
└── meta.json                     split counts, domain groups
```

Kagglers point to this directory. They never touch raw pickles or split logic.

## Architecture

- **Organizer pod** — no GPU, runs Claude Code in a loop. Queries W&B, reviews kaggler PRs, scores predictions.
- **Kaggler pods** — GPU workers, each running Claude Code. Implement hypotheses, train, predict.

## k8s layout

- `k8s/launch.py` — template and apply deployments
- `k8s/kaggler-deployment.yaml` / `k8s/organizer-deployment.yaml` — pod specs
- `k8s/entrypoint-kaggler.sh` / `k8s/entrypoint-organizer.sh` — startup scripts

## instructions/

Role-specific CLAUDE.md files copied into pods at launch:
- `instructions/CLAUDE-ORGANIZER.md` → organizer pods
- `instructions/CLAUDE-KAGGLER.md` → kaggler pods

## Key docs

- `program.md` — research context, goals, metrics
- `sample_competition/` — reference implementation (from prior project, not used directly)
