#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Energie Delen — Dashboard launcher (macOS / Linux)
# Installs uv if not present, then starts the Panel dashboard.
# Safe to run multiple times: uv and Python are only downloaded once.
# ---------------------------------------------------------------------------

# Resolve uv — check PATH first, then the default install location
if command -v uv &>/dev/null; then
    UV=uv
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    echo "uv not found. Installing uv (one-time setup)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if [ ! -x "$HOME/.local/bin/uv" ]; then
        echo "ERROR: uv installation failed. Please install uv manually: https://docs.astral.sh/uv/"
        exit 1
    fi
    UV="$HOME/.local/bin/uv"
fi

# Move to the directory containing this script
cd "$(dirname "$0")"

echo "Starting Energie Delen dashboard..."
"$UV" run --group dashboard panel serve dashboard/app.py --show
