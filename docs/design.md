# GSStudio Design Document

## Overview

GSStudio is a full-stack web platform for running HY-World 2.0 pipelines on a 5x A800-SXM4-80GB GPU node in the POC Kubernetes cluster.

## Architecture

Single-pod deployment: API server + Pipeline Worker + vLLM sidecar, sharing 5 GPUs via CUDA_VISIBLE_DEVICES.

### Components

| Component | Technology | GPU | Purpose |
|-----------|-----------|-----|---------|
| API Server | FastAPI + SQLite | None | REST API, WebSocket logs, static files |
| Pipeline Worker | asyncio | 0-4 | Stage orchestration, subprocess management |
| vLLM Server | vLLM | GPU 0 | Qwen3-VL-8B for trajectory planning |
| Web Frontend | React + Vite | None (browser) | Task management, 3DGS preview |

### GPU Allocation

- GPU 0: vLLM (60% VRAM limit) + Stage 1 (traj_generate) + worldrecon tasks
- GPU 0-4: Stage 0 (panogen), Stage 2-5 (traj_render, video_gen, gen_gs_data, gs_training)

### Task Types

1. **worldgen** - Text/image -> 80B panogen -> 5-stage worldgen pipeline
2. **worldgen_from_pano** - Upload panorama -> 5-stage worldgen pipeline (skip stage 0)
3. **worldrecon** - Multi-view images -> WorldMirror-2 reconstruction

### State Machine

Task: pending -> running -> completed | failed | cancelled
Stage: pending -> running -> completed | failed | skipped

For valid world-generation input, strict trajectory quality changes the
delivery tier rather than the task status: six accepted paths produce
`strict_3d`, four or five produce whitelist-only `degraded_3d`, and fewer than
four produce `panorama_only` with downstream 3D stages skipped. All three are
top-level `completed` results.

### Checkpoint Resume

Each stage writes a completion marker to the task scene directory. On retry, completed stages are skipped.
High-quality tasks additionally keep atomic `auto_recovery.json` version 2
journals with pending batches, source attempts, best-attempt scoring, budgets,
delivery tier, and transient retry counts. The global recovery ceiling is 56
WorldStereo paths, 180 minutes, and two deterministic regenerated sources.

## Model Weights (pre-downloaded to PVC)

| Model | Size | Purpose |
|-------|------|---------|
| HY-Pano-2.0 (80B) | 169.4 GB | Panorama generation |
| HY-WorldMirror-2.0 | 5.1 GB | 3D reconstruction |
| WorldStereo-2 (memory + memory-dmd) | 34.9 GB | High-quality and fast world expansion |
| Qwen3-VL-8B-Instruct | 17.5 GB | vLLM trajectory planning |
| facebook/sam3 | 6.9 GB | Semantic segmentation |
| Ruicheng/moge-2-vitl-normal | 1.3 GB | Monocular depth |

Total: ~235 GB

## Storage Layout

    /data/tasks/{task_id}/
    +-- input/                    # Uploaded input files
    +-- scene/                    # Pipeline scene directory
    +-- output/                   # Final artifacts (ply, spz, glb)
    +-- logs/                     # Per-stage log files
    +-- task.json                 # Task metadata snapshot

    /models/hf-cache/             # HuggingFace model cache (PVC)

## Docker Image

Based on pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel. All CUDA extensions are compiled at build time:

- gsplat_maskgaussian (custom gsplat fork with MaskGaussian)
- navmesh (recastnavigation)
- flash-attn (FlashAttention-2)
- nerfview, fused-ssim, spz, pytorch3d, MoGe

## Key Decisions

1. Single Pod with 5 GPUs - no multi-pod distribution complexity
2. Serial task execution for worldgen (one at a time), worldrecon can interleave
3. vLLM TP=1 on GPU 0 with 60% memory utilization limit
4. 3DGS training steps scaled by factor 1.6 (8 GPU -> 5 GPU adjustment)
5. Spark 2.1 viewer for browser-based 3DGS preview: the viewer mounts as soon
   as a task is opened and streams/decompresses splat data in a Web Worker with
   visible byte progress and dynamic LoD. Paged coarse-to-fine rendering can be
   added later by publishing prebuilt RAD/RADC artifacts.
6. vLLM isolated in dedicated venv (/opt/vllm): vLLM 0.11.0 is the first
   release with native Qwen3-VL support and requires torch 2.8.0 +
   transformers <5, which conflicts with the main HY-World env
   (torch 2.7.1 + transformers 5.2.0). Pipeline talks to vLLM only over HTTP.

## Deployment Status (verified 2026-08-11)

- Image: `ccr-29eug8s3-pub.cnc.bj.baidubce.com/dataflow-poc/gsstudio:a800-v42`
- Digest: `sha256:ddb65575de2b234edf6192faf909d6bf4a26932c3632ca2d08c6c3aefe1e5297`
- Pod: `gsstudio` namespace, 1/1 Running, 5x A800 on node 192.168.80.136
- Public IP URL: http://120.48.33.45 (CCE load balancer + Basic Auth)
- TLS URL: https://geneva-ingredients-expression-located.trycloudflare.com (Cloudflare Tunnel TLS + Basic Auth)
- One-time world generation prerequisite: approved access to gated
  `facebook/sam3` and `HF_TOKEN` via the optional `gsstudio-huggingface` Secret
- `facebook/sam3` only needs that token for its first download. Its complete
  snapshot is retained under the node's persistent `/models/hf-cache`; later
  tasks and Pod restarts use the local cache without requiring the token.
- vLLM 0.11.0 Qwen3-VL-8B: process and trajectory-caption path verified working
- worldrecon e2e via API verified: 8 images -> gaussians.ply (82MB, 1.21M
  vertices) + points.ply + depth/normal/camera in ~45s (warm) 
- frontend served at / (static build), artifacts download verified
- High-quality auto-recovery acceptance task `435d4b33616e` completed as
  `strict_3d`: seed 1025 best attempt, 6 strict trajectories, 2,000 training
  steps, SPZ/PLY output, and authenticated browser preview verification.
- Panorama preview explicitly selects the source `scene/panorama.png`; generated
  depth/normal maps are never used as the interactive panorama.
