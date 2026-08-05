#!/usr/bin/env bash
# Local development: start API server (mock pipeline) + Vite dev server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

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
pip install -q fastapi uvicorn python-multipart aiofiles websockets

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd "$ROOT_DIR/web"
npm install

# Start API server in background
echo ""
echo "Starting API server (mock pipeline) on port 8000..."
cd "$ROOT_DIR"
GSSTUDIO_MOCK_PIPELINE=true \
GSSTUDIO_DATA_DIR="$ROOT_DIR/data" \
GSSTUDIO_PORT=8000 \
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Start Vite dev server
echo "Starting Vite dev server on port 5173..."
cd "$ROOT_DIR/web"
npx vite --host 0.0.0.0 --port 5173 &
VITE_PID=$!

echo ""
echo "=== Dev servers running ==="
echo "  API:      http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo "  API docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

# Wait and cleanup
trap "kill $API_PID $VITE_PID 2>/dev/null; exit" INT TERM
wait
