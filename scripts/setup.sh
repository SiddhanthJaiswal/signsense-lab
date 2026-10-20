#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
command -v uv >/dev/null 2>&1 || { echo 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ first.'; exit 1; }
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.10 .venv
fi
uv pip install --python .venv/bin/python -r services/motion/requirements.lock.txt
npm ci
.venv/bin/python scripts/download_support.py
.venv/bin/python scripts/download_soke.py
printf '\nSetup complete. Start with: npm run dev\n'
