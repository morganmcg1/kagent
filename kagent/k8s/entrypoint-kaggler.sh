set -e
set -o pipefail

WORKDIR="/workspace/repo/kagent"

echo "=== kagent Kaggler: $KAGGLER_NAME ==="
echo "Repo:   $REPO_URL (branch: $REPO_BRANCH)"
echo "GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Repo already cloned by the deployment args block
cd "$WORKDIR"
uv pip install --system -e .

# --- Git identity for commits ---
git config user.name "kagent-$KAGGLER_NAME"
git config user.email "kagent-$KAGGLER_NAME@kagent"

# --- Install Claude Code ---
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.claude/bin:$PATH"

# --- Install gh CLI ---
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null
apt-get update && apt-get install -y gh
# gh uses GITHUB_TOKEN env var automatically, no explicit login needed
echo "=== gh auth ready (using GITHUB_TOKEN env var) ==="

# --- Install role instructions ---
cp instructions/CLAUDE-KAGGLER.md "$WORKDIR/CLAUDE.md"

# --- Launch Claude Code in Ralph Loop ---
export IS_SANDBOX=1

PROMPT="$(eval "cat <<_PROMPT_EOF_
$(cat "$WORKDIR/instructions/prompt-kaggler.md")
_PROMPT_EOF_")"

# --- Start Weave thread logger in background ---
python3 "$WORKDIR/tools/weave_logger.py" --role kaggler --agent-name "$KAGGLER_NAME" --workdir "$WORKDIR" &

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo "=== Ralph Loop iteration $ITERATION ($(date)) ==="

    # Return to latest organizer branch so kaggler starts from the current baseline
    git checkout "$ORGANIZER_BRANCH" 2>/dev/null || true
    git pull origin "$ORGANIZER_BRANCH" 2>/dev/null || true

    # Restore CLAUDE.md — branch checkouts clobber it
    cp "$WORKDIR/instructions/CLAUDE-KAGGLER.md" "$WORKDIR/CLAUDE.md"

    if [ "$ITERATION" -eq 1 ]; then
        claude -p "$PROMPT" --dangerously-skip-permissions || true
    else
        claude -c -p "$PROMPT" --dangerously-skip-permissions || \
        claude -p "$PROMPT" --dangerously-skip-permissions || true
    fi

    echo "=== Claude exited at $(date), restarting in 5s ==="
    sleep 5
done
