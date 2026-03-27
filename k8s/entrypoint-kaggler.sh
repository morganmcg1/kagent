set -e
set -o pipefail

: "${COMPETITION_DIR:?missing COMPETITION_DIR}"

BRANCH="kaggler/$KAGGLER_NAME"
AGENT_MODEL="${AGENT_MODEL:-}"
KAGGLER_WORKDIR="${KAGGLER_WORKDIR:-/workspace/kagent/$COMPETITION_DIR/kaggler}"
KAGGLER_PROMPT_FILE="${KAGGLER_PROMPT_FILE:-KAGGLER_AGENT.md}"

echo "=== kagent Kaggler: $KAGGLER_NAME ==="
echo "Competition: $COMPETITION_DIR"
echo "Model: ${AGENT_MODEL:-default}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd /workspace/kagent

# --- Branch setup ---
git fetch origin
if git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git checkout -b "$BRANCH"
    git push -u origin "$BRANCH"
fi

# --- Install ---
uv pip install --system -e .
git config user.name "kagent-$KAGGLER_NAME"
git config user.email "kagent-$KAGGLER_NAME@kagent"

require_gh() {
    if ! command -v gh >/dev/null 2>&1; then
        echo "GitHub CLI is not present in the image"
        exit 1
    fi
}

update_claude() {
    export PATH="$HOME/.claude/bin:$HOME/.local/bin:$PATH"
    if ! command -v claude >/dev/null 2>&1; then
        echo "Claude CLI is not present in the image"
        exit 1
    fi

    claude update >/dev/null 2>&1 || echo "Claude update failed, using baked version"
}

require_auth() {
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "Missing ANTHROPIC_API_KEY"
        exit 1
    fi
}

run_claude_iteration() {
    local logfile="$1"
    local args=(--model "$AGENT_MODEL" --output-format stream-json --verbose --dangerously-skip-permissions)

    if [ "$ITERATION" -eq 1 ]; then
        claude -p "$PROMPT" "${args[@]}" > "$logfile" 2>&1 || true
    else
        claude -c -p "$PROMPT" "${args[@]}" > "$logfile" 2>&1 || \
        claude -p "$PROMPT" "${args[@]}" > "$logfile" 2>&1 || true
    fi
}

update_claude
require_gh
require_auth

# --- Launch ---
cd "$KAGGLER_WORKDIR"
export IS_SANDBOX=1

if [ ! -f "$KAGGLER_PROMPT_FILE" ]; then
    echo "Missing prompt file: $KAGGLER_WORKDIR/$KAGGLER_PROMPT_FILE"
    exit 1
fi

PROMPT="You are $KAGGLER_NAME. Your branch is $BRANCH. Read $KAGGLER_PROMPT_FILE in the current directory and follow it. Go."
LOGDIR="/workspace/kagent/logs"
mkdir -p "$LOGDIR"

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    mkdir -p "$LOGDIR"
    LOGFILE="$LOGDIR/iter_${ITERATION}_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "=== Iteration $ITERATION ($(date)) → $LOGFILE ==="

    run_claude_iteration "$LOGFILE"

    echo "=== Restarting in 10s ==="
    sleep 10
done
