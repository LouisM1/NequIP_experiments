#!/bin/bash
# Run once after duplicating/moving this folder. Recreates only its local .venv.
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "Install uv first, then rerun this script."; exit 1; }
uv venv --clear --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
printf '\nReady: select this folder\x27s .venv/bin/python as your notebook kernel.\n'
