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

1. **worldgen** - Four ranked 1216x832 text candidates (or one uploaded image) -> strict generated world
2. **worldgen_from_pano** - Uploaded panorama -> strict generated world
3. **worldrecon** - Recommended 12-48 selected multi-view frames/video -> WorldMirror-2 reconstruction

### State Machine

Task: pending -> running -> completed | failed | cancelled
Stage: pending -> running -> completed | failed | skipped

World generation has one 3D delivery tier: `strict_3d`. Fewer than six
alignment-approved trajectories terminate before GS data preparation as
`failed / insufficient_free_view_coverage`. Panorama and polar RGB-D anchors
may supplement a strict trajectory set but can never train a standalone
delivery. Rejected trajectories are excluded from GS data.

Before GPU production, OpenCV contributes 55% and zero-temperature Qwen3-VL
contributes 45% to a minimum 0.70 input score. Resolution, exposure, blur,
features, connected feature-match inliers, optical flow, non-pure-rotation
parallax, edge density, and two principal line directions are hard gates.
People, mirror/glass/reflection, curved walls, unusable floors, and conflicting
perspective are rejected. Reports and all candidates are written atomically.
The exact generated panorama is scored a second time before trajectory
planning, so content or geometry introduced during expansion cannot bypass the
source-candidate gate.

Strict data first enters an independent 500-step pilot. The pilot checkpoint is
evaluation-only: final 2,000-step training always starts from scratch. Content
failure adds four trajectories once, then switches ranked source; infrastructure
failure retries after 30 and 120 seconds without consuming source budget.

After final training, the delivery gate requires at least 12 translated camera
centers, both camera OBB half-extents greater than 0.1 human scale, six of the
eight 45-degree azimuth bins, and eight finite-metric validation renders with
PSNR >= 28, SSIM >= 0.90, and LPIPS <= 0.10. SPZ, PLY, checkpoint, Gaussian
finiteness, opacity, and bounds validation still run. A final fixed-seed stress
test exercises 1,000 bounded navigation steps, 32 views, and 16 micro
translations; any giant Gaussian or unstable/blank/clipped view blocks WASD.
If and only if every failure is caused by a sparse giant-Gaussian set (at most
500 points and 0.05% of the model), the worker may perform one automatic
repair. It archives the original synchronized PLY, SPZ, checkpoint shards, and
metadata, removes the proven offenders from every representation, and repeats
the complete stress test. Any other failure requires source replacement.

### Checkpoint Resume

Each stage writes a completion marker to the task scene directory. On retry, completed stages are skipped.
High-quality tasks additionally keep atomic `auto_recovery.json` version 4
journals with pending batches, source attempts, best-attempt scoring, budgets,
delivery tier, and transient retry counts. The task-wide ceiling is 24
WorldStereo paths, four hours, and three ranked source attempts; switching
source never resets the paid-work counters.

Prompt-only tasks append deterministic geometry guidance to the user's intent
and generate four fixed 1216x832 landscape candidates using `base_seed + 0..3`
instead of model-selected aspect ratio. If all eight initial trajectories contain guidance diagnostics and even
the best worst-frame value is below `0.60`, recovery records a source-guidance
collapse and regenerates/replaces the source before spending GPU budget on
camera-prefix rescue paths. This diagnostic shortcut never weakens the strict
`0.80` acceptance gate.

Within each trajectory class, the initial WorldStereo batch now prefers the
planned cameras with the highest estimated source overlap. The estimate uses
first-to-last translation and rotation from `camera.json`; missing or malformed
pose metadata falls back to deterministic trajectory-id ordering.

Each source starts with eight regular trajectories and adds four or bounded
safe-prefix rescue paths. Ten accepted paths is the recovery target; six is the
absolute training minimum. A source with 16 attempts and fewer than six passes
is abandoned. The A800 image rejects an empty accepted-trajectory whitelist.
There is no empty-validation, anchored, degraded, or panorama-only training
fallback.

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

## Deployment Status (verified 2026-08-13)

- Image: `ccr-29eug8s3-pub.cnc.bj.baidubce.com/dataflow-poc/gsstudio:a800-v66`
- Digest: `sha256:08f681ddec830d2fcffc19c339f035b28d0a783ab841532f12cf200003e3a9cc`
- Rollback image: `a800-v65`
- Pod: `gsstudio-54d57d77b-89jjv`, 1/1 Running, restart count 0, 5x A800
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
- A production-data repair replay removed 119 exact giant-Gaussian rows from
  665,188 synchronized PLY/SPZ/checkpoint records, preserved the originals, and
  passed the complete navigation stress test with 665,069 remaining records.
- Task `f857acb62a03` was revalidated through the v66 UI. Its generated panorama
  failed the post-panorama gate for curved walls, mirror/glass, a person, edge
  density, and perspective conflict, so the product requires source replacement
  and exposes neither the repair action nor 3D/WASD navigation.
- Panorama preview explicitly selects the source `scene/panorama.png`; generated
  depth/normal maps are never used as the interactive panorama.
