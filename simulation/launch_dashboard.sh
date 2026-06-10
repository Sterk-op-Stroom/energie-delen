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
    echo "Setting up (one-time, this may take a moment)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    if [ ! -x "$HOME/.local/bin/uv" ]; then
        echo "ERROR: Setup failed. Please install uv manually: https://docs.astral.sh/uv/"
        exit 1
    fi
    UV="$HOME/.local/bin/uv"
fi

# Move to the directory containing this script
cd "$(dirname "$0")"

echo "Starting Energie Delen dashboard..."
echo "Dashboard running. Your browser will open automatically. Press Ctrl+C to stop."
"$UV" run --quiet --group dashboard panel serve dashboard/app.py --show 2>/dev/null
