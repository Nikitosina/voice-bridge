#!/usr/bin/env bash
set -euo pipefail

# Voice Bridge Server — run.sh
# Creates/activates venv, installs deps, starts uvicorn.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[run.sh] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "[run.sh] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "[run.sh] Installing/updating dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "[run.sh] Starting voice bridge server on ${VOICE_BRIDGE_HOST:-0.0.0.0}:${VOICE_BRIDGE_PORT:-8765}..."
uvicorn server:app \
    --host "${VOICE_BRIDGE_HOST:-0.0.0.0}" \
    --port "${VOICE_BRIDGE_PORT:-8765}" \
    --log-level info \
    --reload
