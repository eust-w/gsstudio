#!/usr/bin/env bash
# Render the public-IP LoadBalancer Service from deploy/k8s/service-public-ip.yaml.tmpl.
#
# The elastic IP and listener port are deployment configuration and are deliberately
# not stored in the manifest. Required/optional inputs:
#
#   GSSTUDIO_PUBLIC_IP     (required) CCE elastic IP to bind, e.g. the POC's EIP
#   GSSTUDIO_PUBLIC_PORT   (optional) listener port, default 80
#
# Usage:
#   GSSTUDIO_PUBLIC_IP=<ip> deploy/scripts/render-public-service.sh            # print manifest
#   GSSTUDIO_PUBLIC_IP=<ip> deploy/scripts/render-public-service.sh --apply    # kubectl apply -f -
#   GSSTUDIO_PUBLIC_IP=<ip> deploy/scripts/render-public-service.sh --dry-run  # kubectl --dry-run=client
#
# The CCE listener annotations reference the same port, so one variable keeps the
# Service and the cloud listener consistent. Missing variables fail loudly instead
# of rendering an empty loadBalancerIP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="$SCRIPT_DIR/../k8s/service-public-ip.yaml.tmpl"

if [ -z "${GSSTUDIO_PUBLIC_IP:-}" ]; then
    echo "ERROR: GSSTUDIO_PUBLIC_IP is required (elastic IP for the LoadBalancer)" >&2
    exit 1
fi
export GSSTUDIO_PUBLIC_IP
export GSSTUDIO_PUBLIC_PORT="${GSSTUDIO_PUBLIC_PORT:-80}"

render() {
    python3 - "$TEMPLATE" <<'PY'
import os
import sys
from string import Template

source = open(sys.argv[1], encoding="utf-8").read()
sys.stdout.write(Template(source).substitute(os.environ))
PY
}

case "${1:-}" in
    --apply)
        command -v kubectl >/dev/null || { echo "ERROR: kubectl not found" >&2; exit 1; }
        render | kubectl apply -f -
        ;;
    --dry-run)
        command -v kubectl >/dev/null || { echo "ERROR: kubectl not found" >&2; exit 1; }
        render | kubectl apply --dry-run=client -f -
        ;;
    "")
        render
        ;;
    *)
        echo "Usage: $0 [--apply|--dry-run]" >&2
        exit 2
        ;;
esac
