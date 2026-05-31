#!/usr/bin/env bash
# One-time remote (A100) provisioning. Run on the box after SSH.
set -euo pipefail

# 1. uv
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# 2. GitHub SSH key (paste the printed pubkey into github.com/settings/keys)
if [ ! -f "$HOME/.ssh/id_ed25519_github" ]; then
  ssh-keygen -t ed25519 -N "" -C "a100-attn-pool-dom" -f "$HOME/.ssh/id_ed25519_github"
fi
cat >> "$HOME/.ssh/config" <<'EOF'
Host github.com
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
EOF
echo "=== ADD THIS PUBKEY TO GITHUB ==="
cat "$HOME/.ssh/id_ed25519_github.pub"
echo "================================="

# 3. clone + sync (after the key is added)
#   git clone git@github.com:jamie-stephenson/attn-pool-dom.git
#   cd attn-pool-dom && uv sync && uv run pytest
