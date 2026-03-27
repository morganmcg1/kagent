set -e
set -o pipefail

: "${COMPETITION_DIR:?missing COMPETITION_DIR}"
: "${COMPETITION_NAME:?missing COMPETITION_NAME}"

WORKDIR="${ORGANIZER_WORKDIR:-/workspace/kagent/$COMPETITION_DIR/organizer}"
POLL_INTERVAL=300  # 5 minutes

echo "=== kagent Organizer ==="
echo "Competition: $COMPETITION_DIR"
echo "Repo:     $REPO_URL (branch: $REPO_BRANCH)"
echo "Polling:  /mnt/new-pvc/predictions/ every ${POLL_INTERVAL}s"

cd /workspace/kagent

# --- Organizer branch setup ---
ORGANIZER_BRANCH="${COMPETITION_NAME}/organizer"
git fetch origin
if git rev-parse --verify "origin/$ORGANIZER_BRANCH" >/dev/null 2>&1; then
    git checkout "$ORGANIZER_BRANCH"
    git pull origin "$ORGANIZER_BRANCH"
else
    git checkout -b "$ORGANIZER_BRANCH"
    git push -u origin "$ORGANIZER_BRANCH"
fi
git config user.name "kagent-${COMPETITION_NAME}-organizer"
git config user.email "kagent-${COMPETITION_NAME}-organizer@kagent"
export ORGANIZER_BRANCH

# Install deps from root pyproject
uv pip install --system -e .
cd "$WORKDIR"

# --- Inject WANDB_PROJECT into competition files ---
if [ -n "$WANDB_PROJECT" ] && [ "$WANDB_PROJECT" != "kagent-v2" ]; then
    sed -i "s/kagent-v2/$WANDB_PROJECT/g" \
        "$WORKDIR/score.py" \
        "$WORKDIR/README.md" \
        "$WORKDIR/train.py" \
        2>/dev/null || true
    echo "Injected WANDB_PROJECT=$WANDB_PROJECT into competition files"
fi

echo "=== Organizer ready, starting scoring loop ==="

while true; do
    echo "--- Scoring check $(date) ---"
    python score.py --score_all 2>&1 || echo "  scoring error (will retry)"
    echo "--- Sleeping ${POLL_INTERVAL}s ---"
    sleep "$POLL_INTERVAL"
done
