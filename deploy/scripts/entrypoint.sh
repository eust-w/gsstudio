#!/usr/bin/env bash
set -euo pipefail

echo =======================================
echo  GSStudio Starting
echo =======================================
echo Data dir:    $GSSTUDIO_DATA_DIR
echo HF cache:    $GSSTUDIO_HF_HOME
echo HY-World:    $GSSTUDIO_HYWORLD_REPO
echo GPUs:        $GSSTUDIO_NUM_GPUS
echo API port:    $GSSTUDIO_PORT
echo vLLM port:   $VLLM_PORT
echo =======================================

mkdir -p $GSSTUDIO_DATA_DIR $GSSTUDIO_HF_HOME /var/log/supervisor

# HY-World's trajectory script currently resolves ZIM/GroundingDINO through
# ~/.cache explicitly. Point that legacy path at the same persistent hub used
# by Transformers so Pod restarts never trigger another model download.
UPSTREAM_HF_HOME="/root/.cache/huggingface"
mkdir -p "$GSSTUDIO_HF_HOME/hub" "$UPSTREAM_HF_HOME"
if [ -d "$UPSTREAM_HF_HOME/hub" ] && [ ! -L "$UPSTREAM_HF_HOME/hub" ]; then
    cp -a "$UPSTREAM_HF_HOME/hub/." "$GSSTUDIO_HF_HOME/hub/" 2>/dev/null || true
    rm -rf "$UPSTREAM_HF_HOME/hub"
fi
ln -sfn "$GSSTUDIO_HF_HOME/hub" "$UPSTREAM_HF_HOME/hub"

if command -v nvidia-smi &> /dev/null; then
    echo GPU status:
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
else
    echo WARNING: nvidia-smi not found
fi

if [ -d $GSSTUDIO_HF_HOME/hub ]; then
    CACHE_SIZE=$(du -sh $GSSTUDIO_HF_HOME 2>/dev/null | cut -f1 || echo unknown)
    echo HF cache size: $CACHE_SIZE
else
    echo WARNING: HF cache not populated
fi

exec supervisord -c /etc/supervisor/conf.d/gsstudio.conf
