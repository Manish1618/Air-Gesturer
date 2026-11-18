#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first (see README.md)." >&2
  exit 1
fi
if ! command -v ydotool >/dev/null 2>&1; then
  echo "Note: ydotool is not installed, so scrolling is disabled (see README.md)." >&2
fi

if [[ ! -d .venv || requirements.txt -nt .venv ]]; then
  uv venv --allow-existing --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
  touch .venv
fi

exec .venv/bin/python air_scroll.py "$@"
