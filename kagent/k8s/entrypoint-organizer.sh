set -o pipefail

WORKDIR="/workspace/kagent"

echo "=== kagent Organizer ==="
echo "Repo:     $REPO_URL (branch: $REPO_BRANCH)"
echo "Tag:      $RESEARCH_TAG"
echo "Kagglers: $KAGGLER_NAMES"

# Repo already cloned by the deployment args block
cd "$WORKDIR"

# --- Install role instructions ---
cp "$WORKDIR/instructions/CLAUDE-ORGANIZER.md" "$WORKDIR/CLAUDE.md"

uv pip install --system -e .

# --- Git identity ---
git config user.name "kagent-organizer"
git config user.email "kagent-organizer@kagent"

# --- Create or checkout organizer branch ---
git fetch origin
if git rev-parse --verify "origin/$ORGANIZER_BRANCH" >/dev/null 2>&1; then
    git checkout "$ORGANIZER_BRANCH"
    git pull origin "$ORGANIZER_BRANCH"
else
    git checkout -b "$ORGANIZER_BRANCH"
    git push -u origin "$ORGANIZER_BRANCH"
fi

# --- Install Claude Code ---
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.claude/bin:$PATH"

# --- Install kubectl ---
curl -fsSL "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" -o /usr/local/bin/kubectl
chmod +x /usr/local/bin/kubectl

# --- Install gh CLI ---
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null
apt-get update && apt-get install -y gh
# gh uses GITHUB_TOKEN env var automatically, no explicit login needed
echo "=== gh auth ready (using GITHUB_TOKEN env var) ==="

# --- Build prompt (bash heredoc expansion — no envsubst needed) ---
PROMPT="$(eval "cat <<_PROMPT_EOF_
$(cat "$WORKDIR/instructions/prompt-organizer.md")
_PROMPT_EOF_")"

# --- Launch Claude Code in Ralph Loop ---
export IS_SANDBOX=1

LOGDIR="/workspace/kagent/organizer_logs"
mkdir -p "$LOGDIR"

# --- Start Weave thread logger in background ---
python3 "$WORKDIR/tools/weave_logger.py" --role organizer --agent-name organizer --workdir "$WORKDIR" &

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    LOGFILE="$LOGDIR/iteration_${ITERATION}_$(date +%Y%m%d_%H%M%S).jsonl"
    echo "=== Organizer Loop iteration $ITERATION ($(date)) ==="
    echo "=== Log: $LOGFILE ==="

    # Restore CLAUDE.md — branch checkouts clobber it
    cp "$WORKDIR/instructions/CLAUDE-ORGANIZER.md" "$WORKDIR/CLAUDE.md"

    if [ "$ITERATION" -eq 1 ]; then
        claude -p "$PROMPT" --model "claude-opus-4-6[1m]" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || true
    else
        claude -c -p "$PROMPT" --model "claude-opus-4-6[1m]" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || \
        claude -p "$PROMPT" --model "claude-opus-4-6[1m]" --output-format stream-json --verbose --dangerously-skip-permissions > "$LOGFILE" 2>&1 || true
    fi

    echo "=== Organizer exited at $(date), next check in 5 minutes ==="
    sleep 300
done
