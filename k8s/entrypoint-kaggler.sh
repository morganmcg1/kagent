set -e
set -o pipefail

BRANCH="kaggler/$KAGGLER_NAME"

echo "=== kagent Kaggler: $KAGGLER_NAME ==="
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

# --- Claude Code ---
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.claude/bin:$PATH"

# --- gh CLI ---
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null
chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null
apt-get update -qq && apt-get install -y -qq gh

# --- Launch ---
cd /workspace/kagent/cfd-competition/kaggler
export IS_SANDBOX=1

PROMPT="You are $KAGGLER_NAME. Your branch is $BRANCH. Follow the instructions in @CLAUDE-KAGGLER.md. Go."

ITERATION=0
while true; do
    ITERATION=$((ITERATION + 1))
    echo "=== Iteration $ITERATION ($(date)) ==="

    if [ "$ITERATION" -eq 1 ]; then
        claude -p "$PROMPT" --dangerously-skip-permissions || true
    else
        claude -c -p "$PROMPT" --dangerously-skip-permissions || \
        claude -p "$PROMPT" --dangerously-skip-permissions || true
    fi

    echo "=== Restarting in 10s ==="
    sleep 10
done
