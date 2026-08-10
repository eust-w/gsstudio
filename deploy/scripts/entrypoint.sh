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

# `traj_generate.py` currently catches a missing rtree dependency and exits
# successfully.  Fail before accepting work instead of creating a silent
# low-quality task.
python -c 'import rtree; print("rtree startup preflight OK")'

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

# The full WorldStereo preset uses UniPC, unlike the faster DMD preset. Verify
# its scheduler before accepting work so an incomplete offline cache cannot
# fail only after panorama, NavMesh, and trajectory rendering have completed.
shopt -s nullglob
WORLDSTEREO_SCHEDULER=(
    "$GSSTUDIO_HF_HOME"/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers/snapshots/*/scheduler/scheduler_config.json
)
if [ "${#WORLDSTEREO_SCHEDULER[@]}" -eq 0 ] || [ ! -f "${WORLDSTEREO_SCHEDULER[0]}" ]; then
    echo "ERROR: full WorldStereo offline preflight failed: missing Wan scheduler/scheduler_config.json" >&2
    exit 1
fi
for required_model_file in \
    "$GSSTUDIO_HF_HOME/hub/hanshanxue--WorldStereo/worldstereo-memory/config.json" \
    "$GSSTUDIO_HF_HOME/hub/hanshanxue--WorldStereo/worldstereo-memory/model.safetensors" \
    "$GSSTUDIO_HF_HOME/hub/hanshanxue--WorldStereo/worldstereo-memory-dmd/config.json" \
    "$GSSTUDIO_HF_HOME/hub/hanshanxue--WorldStereo/worldstereo-memory-dmd/model.safetensors"; do
    if [ ! -f "$required_model_file" ]; then
        echo "ERROR: WorldStereo offline preflight failed: missing $required_model_file" >&2
        exit 1
    fi
done
echo "WorldStereo offline preflight OK"

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
