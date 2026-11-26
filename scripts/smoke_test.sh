#!/usr/bin/env bash
# Verify API contracts without creating a task or spending GPU resources.
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
REQUIRE_READY=false
if [ "${2:-}" = "--require-ready" ]; then
    REQUIRE_READY=true
elif [ -n "${2:-}" ]; then
    echo "Usage: $0 [base-url] [--require-ready]" >&2
    exit 2
fi
AUTH=(-H "Accept: application/json")
if [ -n "${GSSTUDIO_SERVICE_TOKEN:-}" ]; then
    AUTH+=(-H "Authorization: Bearer $GSSTUDIO_SERVICE_TOKEN")
elif [ -n "${GSSTUDIO_BASIC_AUTH_USER:-}" ] && [ -n "${GSSTUDIO_BASIC_AUTH_PASSWORD:-}" ]; then
    AUTH+=(-u "$GSSTUDIO_BASIC_AUTH_USER:$GSSTUDIO_BASIC_AUTH_PASSWORD")
fi
echo "=== GSStudio Smoke Test ==="
echo "Target: $BASE_URL"

# 1. Health check
echo ""
echo "[1/5] Health check..."
HEALTH=$(curl --max-time 10 -sf "$BASE_URL/api/health" || echo "FAILED")
if [ "$HEALTH" = "FAILED" ]; then
    echo "FAIL: API not responding"
    exit 1
fi
echo "OK: $HEALTH"

# 2. GPU status
echo ""
echo "[2/5] GPU status..."
GPU=$(curl --max-time 10 -sf "${AUTH[@]}" "$BASE_URL/api/gpu/status" || echo "FAILED")
if [ "$GPU" = "FAILED" ]; then
    echo "FAIL: GPU status endpoint not responding or authentication is required"
    exit 1
else
    echo "OK: $GPU"
fi

# 3. Runtime readiness is distinct from API liveness.
echo ""
echo "[3/5] Runtime preflight..."
RUNTIME=$(curl --max-time 30 -sf "${AUTH[@]}" "$BASE_URL/api/system/status" || echo "FAILED")
if [ "$RUNTIME" = "FAILED" ]; then
    echo "FAIL: Runtime status endpoint not responding"
    exit 1
fi
printf '%s' "$RUNTIME" | python3 -c 'import json,sys; value=json.load(sys.stdin); print("Mode:", value["mode"], "Real runtime ready:", value["ready"]); [print(item["task_type"]+": "+("ready" if item["ready"] else ", ".join(reason["code"] for reason in item["blockers"]))) for item in value["capabilities"]]'
if [ "$REQUIRE_READY" = "true" ]; then
    printf '%s' "$RUNTIME" | python3 -c 'import json,sys; value=json.load(sys.stdin); sys.exit(0 if value["mode"] == "real" and value["ready"] else 1)' || { echo "FAIL: No real task capability is ready"; exit 1; }
fi

# 4. Read existing task metadata only.
echo ""
echo "[4/5] Task list..."
TASKS=$(curl --max-time 10 -sf "${AUTH[@]}" "$BASE_URL/api/tasks?limit=1" || echo "FAILED")
if [ "$TASKS" = "FAILED" ]; then
    echo "FAIL: Could not list tasks"
    exit 1
fi
printf '%s' "$TASKS" | python3 -c 'import json,sys; value=json.load(sys.stdin); assert isinstance(value["tasks"], list); print("OK: task list accessible")'

# 5. Confirm an invalid request is rejected before any work is queued.
echo ""
echo "[5/5] Empty reconstruction is rejected..."
REJECT_CODE=$(curl --max-time 10 -s -o /dev/null -w '%{http_code}' "${AUTH[@]}" -X POST "$BASE_URL/api/tasks" \
    -F 'type=worldrecon' -F 'name=smoke-invalid-input' -F 'config={}')
if [ "$REJECT_CODE" != "400" ]; then
    echo "FAIL: Expected HTTP 400 for missing reconstruction input; received $REJECT_CODE"
    exit 1
fi
echo "OK: invalid input rejected"

echo ""
echo "=== API Smoke Test PASSED (no task created) ==="
