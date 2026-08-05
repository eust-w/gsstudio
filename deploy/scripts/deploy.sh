#!/usr/bin/env bash
# Deploy GSStudio to the POC K8s cluster.
set -euo pipefail

NAMESPACE="gsstudio"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
K8S_DIR="$(dirname "$SCRIPT_DIR")/k8s"

echo "=== GSStudio Deployment ==="

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo "ERROR: kubectl not found"
    exit 1
fi

# Create namespace
echo "[1/5] Creating namespace..."
kubectl apply -f "$K8S_DIR/namespace.yaml"

# Create PVCs
echo "[2/5] Creating PVCs..."
kubectl apply -f "$K8S_DIR/pvc.yaml"

# Wait for PVCs to be bound
echo "[3/5] Waiting for PVCs..."
kubectl -n "$NAMESPACE" wait --for=condition=Bound pvc/gsstudio-hf-cache --timeout=120s || true
kubectl -n "$NAMESPACE" wait --for=condition=Bound pvc/gsstudio-workdir --timeout=120s || true

# Check if model prefetch is needed
if kubectl -n "$NAMESPACE" get job gsstudio-prefetch-models &> /dev/null; then
    echo "[4/5] Model prefetch job already exists, skipping"
else
    echo "[4/5] Starting model prefetch job..."
    echo "     Run ./deploy/scripts/prefetch_models.sh --k8s to start the download"
    echo "     This will download ~235GB of model weights"
fi

# Deploy application
echo "[5/5] Deploying GSStudio..."
kubectl apply -f "$K8S_DIR/deployment.yaml"
kubectl apply -f "$K8S_DIR/service.yaml"

# Optional ingress
if [ -f "$K8S_DIR/ingress.yaml" ]; then
    kubectl apply -f "$K8S_DIR/ingress.yaml"
fi

echo ""
echo "=== Deployment Status ==="
kubectl -n "$NAMESPACE" get pods -l app=gsstudio
echo ""
echo "Access: kubectl -n $NAMESPACE port-forward svc/gsstudio 8000:80"
echo "Then open http://localhost:8000"
