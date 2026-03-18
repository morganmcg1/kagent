"""Launch kagent resources on Kubernetes."""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp

TEMPLATES_DIR = Path(__file__).parent
KAGGLER_TEMPLATE = TEMPLATES_DIR / "kaggler-deployment.yaml"
ORGANIZER_TEMPLATE = TEMPLATES_DIR / "organizer-deployment.yaml"
PREPARE_TEMPLATE = TEMPLATES_DIR / "prepare-splits-job.yaml"

KAGGLER_NAMES = [
    "frieren", "fern", "tanjiro", "nezuko", "alphonse", "edward",
    "thorfinn", "askeladd", "violet", "gilbert", "senku", "kohaku",
    "emma", "norman", "chihiro", "haku", "shoya", "shouko",
    "mitsuha", "taki", "shinji", "rei", "kaneda", "tetsuo",
]


@dataclass
class Args:
    """Launch kagent resources on Kubernetes."""
    tag: str  # research tag (e.g. mar18)
    names: str = ""  # comma-separated kaggler names (e.g. "frieren,fern")
    n_kagglers: int = 4  # number of kagglers to launch (ignored if --names is provided)
    repo_url: str = "https://github.com/tcapelle/kagent.git"  # git repo URL
    repo_branch: str = "main"  # git branch to clone
    image: str = "ghcr.io/tcapelle/dev_box:latest"  # container image
    wandb_entity: str = "wandb-applied-ai-team"  # W&B entity
    wandb_project: str = "kagent-v1"  # W&B project name
    organizer_branch: str = "jurgen"  # branch the organizer works on
    organizer: bool = False  # also deploy the organizer pod
    prepare: bool = False  # run prepare_splits.py as a one-shot Job
    dry_run: bool = False  # print manifests without applying


def render_template(template: str, replacements: dict[str, str]) -> str:
    out = template
    for key, value in replacements.items():
        out = out.replace(f"{{{{{key}}}}}", value)
    return out


def render_configmap(name: str, labels: dict[str, str], data: dict[str, str]) -> str:
    lines = ["apiVersion: v1", "kind: ConfigMap", "metadata:", f"  name: {name}", "  labels:"]
    for k, v in labels.items():
        lines.append(f"    {k}: {v}")
    lines.append("data:")
    for k, v in data.items():
        lines.append(f"  {k}: \"{v}\"")
    return "\n".join(lines)


def render_kaggler(template: str, kaggler_name: str, tag: str, args: Args) -> str:
    configmap = render_configmap(
        name=f"kagent-config-kaggler-{kaggler_name}",
        labels={"app": "kagent", "role": "kaggler", "research-tag": tag},
        data={
            "REPO_URL": args.repo_url,
            "REPO_BRANCH": args.repo_branch,
            "KAGGLER_NAME": kaggler_name,
            "RESEARCH_TAG": tag,
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_PROJECT": args.wandb_project,
            "ORGANIZER_BRANCH": args.organizer_branch,
            "WANDB_MODE": "online",
        },
    )
    deployment = render_template(template, {
        "KAGGLER_NAME": kaggler_name,
        "RESEARCH_TAG": tag,
        "IMAGE": args.image,
        "ORGANIZER_BRANCH": args.organizer_branch,
    })
    return configmap + "\n---\n" + deployment


def render_organizer(template: str, tag: str, kaggler_list: list[str], args: Args) -> str:
    configmap = render_configmap(
        name="kagent-config-organizer",
        labels={"app": "kagent", "role": "organizer", "research-tag": tag},
        data={
            "REPO_URL": args.repo_url,
            "REPO_BRANCH": args.repo_branch,
            "RESEARCH_TAG": tag,
            "KAGGLER_NAMES": ",".join(kaggler_list),
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_PROJECT": args.wandb_project,
            "ORGANIZER_BRANCH": args.organizer_branch,
        },
    )
    deployment = render_template(template, {"RESEARCH_TAG": tag})
    return configmap + "\n---\n" + deployment


def render_prepare(template: str, tag: str, args: Args) -> str:
    configmap = render_configmap(
        name="kagent-config-prepare",
        labels={"app": "kagent", "role": "prepare", "research-tag": tag},
        data={
            "REPO_URL": args.repo_url,
            "REPO_BRANCH": args.repo_branch,
        },
    )
    job = render_template(template, {
        "RESEARCH_TAG": tag,
        "IMAGE": args.image,
    })
    return configmap + "\n---\n" + job


def kubectl_apply(manifest: str, name: str):
    print(f"Launching: {name}")
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest, text=True, capture_output=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"  {result.stdout.strip()}")


def main():
    args = sp.parse(Args)

    # --- Prepare splits job ---
    if args.prepare:
        manifest = render_prepare(PREPARE_TEMPLATE.read_text(), args.tag, args)
        if args.dry_run:
            print("--- Prepare Splits Job ---")
            print(manifest)
        else:
            kubectl_apply(manifest, "prepare-splits")
            print(f"\nMonitor:")
            print(f"  kubectl logs -f job/kagent-prepare-splits")
            print(f"  kubectl get job kagent-prepare-splits")
        return

    # --- Resolve kaggler list ---
    if args.names:
        kaggler_list = [n.strip() for n in args.names.split(",")]
    else:
        if args.n_kagglers > len(KAGGLER_NAMES):
            print(f"ERROR: max {len(KAGGLER_NAMES)} kagglers (got {args.n_kagglers})", file=sys.stderr)
            sys.exit(1)
        kaggler_list = KAGGLER_NAMES[:args.n_kagglers]

    # --- Deploy kagglers ---
    kaggler_template = KAGGLER_TEMPLATE.read_text()
    for name in kaggler_list:
        manifest = render_kaggler(kaggler_template, name, args.tag, args)
        if args.dry_run:
            print(f"--- Kaggler: {name} ---")
            print(manifest)
            print()
        else:
            kubectl_apply(manifest, f"kaggler {name}")

    # --- Deploy organizer ---
    if args.organizer:
        organizer_template = ORGANIZER_TEMPLATE.read_text()
        manifest = render_organizer(organizer_template, args.tag, kaggler_list, args)
        if args.dry_run:
            print("--- Organizer ---")
            print(manifest)
            print()
        else:
            kubectl_apply(manifest, "organizer")

    if not args.dry_run:
        print(f"\nLaunched {len(kaggler_list)} kagglers: {', '.join(kaggler_list)}")
        if args.organizer:
            print("Launched organizer pod")
        print(f"\nMonitor:")
        print(f"  kubectl get deployments -l research-tag={args.tag}")
        print(f"  kubectl logs -f deployment/kagent-{kaggler_list[0]}")
        print(f"\nStop:")
        print(f"  kubectl delete deployments,configmaps -l research-tag={args.tag}")


if __name__ == "__main__":
    main()
