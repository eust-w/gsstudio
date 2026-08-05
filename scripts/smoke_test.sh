#!/usr/bin/env bash
# Smoke test after deployment: verify API is healthy and can create tasks.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
echo "=== GSStudio Smoke Test ==="
echo "Target: $BASE_URL"

# 1. Health check
echo ""
echo "[1/4] Health check..."
HEALTH=$(curl -sf "$BASE_URL/api/health" || echo "FAILED")
if [ "$HEALTH" = "FAILED" ]; then
    echo "FAIL: API not responding"
    exit 1
fi
echo "OK: $HEALTH"

# 2. GPU status
echo ""
echo "[2/4] GPU status..."
GPU=$(curl -sf "$BASE_URL/api/gpu/status" || echo "FAILED")
if [ "$GPU" = "FAILED" ]; then
    echo "WARN: GPU status endpoint not responding"
else
    echo "OK: $GPU"
fi

# 3. Create a test task (worldrecon with mock data)
echo ""
echo "[3/4] Creating test task..."
TASK_RESPONSE=$(curl -sf -X POST "$BASE_URL/api/tasks" \
    -F "type=worldrecon" \
    -F "name=smoke-test-$(date +%s)" \
    -F "config={}" || echo "FAILED")
if [ "$TASK_RESPONSE" = "FAILED" ]; then
    echo "FAIL: Could not create task"
    exit 1
fi
echo "OK: $TASK_RESPONSE"

# Extract task ID
TASK_ID=$(echo "$TASK_RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('task_id','unknown'))")

# 4. Check task status
echo ""
echo "[4/4] Checking task status..."
sleep 2
TASK=$(curl -sf "$BASE_URL/api/tasks/$TASK_ID" || echo "FAILED")
if [ "$TASK" = "FAILED" ]; then
    echo "FAIL: Could not get task"
    exit 1
fi
echo "OK: Task created and queryable"

echo ""
echo "=== Smoke Test PASSED ==="
