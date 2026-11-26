#!/usr/bin/env bash
# Local development: start the real API runtime and Vite. Demo is opt-in.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODE="real"
case "${1:-}" in
    --demo) MODE="demo" ;;
    "") ;;
    *) echo "Usage: $0 [--demo]" >&2; exit 2 ;;
esac
MOCK_PIPELINE=false
DEFAULT_DATA_DIR="$ROOT_DIR/data"
if [ "$MODE" = "demo" ]; then
    MOCK_PIPELINE=true
    DEFAULT_DATA_DIR="$ROOT_DIR/data/demo"
fi
TASK_DATA_DIR="${GSSTUDIO_DATA_DIR:-$DEFAULT_DATA_DIR}"

echo "=== GSStudio Local Dev ==="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

# Create virtualenv if needed
VENV="$ROOT_DIR/.venv"
if [ ! -d "$VENV" ]; then
    echo "Creating Python virtualenv..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Install server dependencies
echo "Installing server dependencies..."
pip install -q -r "$ROOT_DIR/requirements-dev.txt" websockets

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd "$ROOT_DIR/web"
npm ci

# Start API server in background
echo ""
echo "Starting API server ($MODE mode) on port 8000..."
if [ "$MODE" = "real" ]; then
    echo "Real tasks require the HY-World models, CUDA and VLM service. See /api/system/status for readiness."
else
    echo "DEMO MODE: synthetic task execution; no real reconstruction."
fi
cd "$ROOT_DIR"
GSSTUDIO_MOCK_PIPELINE="$MOCK_PIPELINE" \
GSSTUDIO_MANAGE_VLLM="${GSSTUDIO_MANAGE_VLLM:-false}" \
GSSTUDIO_DATA_DIR="$TASK_DATA_DIR" \
GSSTUDIO_PORT=8000 \
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload &
API_PID=$!

# Start Vite dev server
echo "Starting Vite dev server on port 5173..."
cd "$ROOT_DIR/web"
npx vite --host 127.0.0.1 --port 5173 &
VITE_PID=$!

echo ""
echo "=== Dev servers running ==="
echo "  API:      http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo "  API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

# Wait and cleanup
cleanup() { kill "$API_PID" "$VITE_PID" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait
