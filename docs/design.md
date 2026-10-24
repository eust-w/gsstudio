# GSStudio Design Document

## Overview

GSStudio is a full-stack web platform for running HY-World 2.0 pipelines. The
recorded POC deployment targets a 5x A800-SXM4-80GB GPU node; its last GPU/public
acceptance was on 2026-08-12. Current online availability is unverified as of
2026-10-25.

## Architecture

The GPU application runs in one container: supervisord manages the API and an
independent vLLM process, and the API starts the asyncio Pipeline Worker. They
share five GPUs via `CUDA_VISIBLE_DEVICES`; vLLM is not a separate sidecar
container. The optional Cloudflare tunnel is a separate deployment.

### Components

| Component | Technology | GPU | Purpose |
|-----------|-----------|-----|---------|
| API Server | FastAPI + SQLite | None | REST API, WebSocket logs, static files |
| Pipeline Worker | asyncio | 0-4 | Stage orchestration, subprocess management |
| vLLM Server | vLLM | GPU 0 | Qwen3-VL-8B for trajectory planning |
| Web Frontend | React + Vite | None (browser) | Task management, 3DGS preview |

### GPU Allocation

- GPU 0: vLLM (60% VRAM limit), Stage 1 (`traj_generate`), and worldrecon tasks
  by default; worldrecon can opt into multiple GPUs with `recon_gpus`.
- GPU 0-4: `panogen`, `traj_render`, `gen_gs_data`, and `gs_training` in the
  high-quality configuration.
- GPU 0-3: `video_gen`; WorldStereo sequence parallelism must divide its 16
  attention heads, so the five-GPU deployment uses four for this stage.
- The worker stops vLLM before `panogen` and `video_gen` to release memory and
  starts it when trajectory planning/rendering needs it. All task types run
  serially in the same queue.

### Task Types

1. **worldgen** - Non-empty text prompt or image -> source/panorama generation
   -> 5-stage worldgen pipeline. Prompt-only mode generates a source image with
   HunyuanImage-3, then expands it to a panorama using the same loaded model.
2. **worldgen_from_pano** - Upload panorama -> 5-stage worldgen pipeline (skip stage 0)
3. **worldrecon** - Multi-view images or video -> WorldMirror-2 reconstruction

### State Machine

Task: pending -> running -> completed | failed | cancelled
Stage: pending -> running -> completed | failed | skipped

For high-quality world generation, strict trajectory quality selects the
training delivery tier:

| Strictly accepted trajectories | Delivery tier | Training mode |
|-------------------------------|---------------|---------------|
| 6 or more | `strict_3d` | `accepted_trajectories` |
| 1-5 | `degraded_3d` | `accepted_plus_anchors` |
| 0 | `anchored_3d` | `panorama_anchored` |

All tiers continue through GS data preparation and training. Rejected
trajectories are excluded; `training_input_manifest.json` records accepted and
excluded IDs, RGB-D anchors, and point-cloud inputs. `panorama_only` is a legacy
journal value, migrated to the applicable current tier when read; migration
alone does not create missing 3D artifacts.

Real world-generation tasks become `completed` only after the delivery
validator finds an SPZ, Gaussian PLY, a trainer checkpoint under `output/ckpts`
whose step matches the SPZ, and parseable viewer pose metadata. SPZ decoding
must produce at least 50,000 Gaussians with finite attributes and decoded
opacities and a non-degenerate bounding box. The result is recorded in
`artifact_validation.json`. A valid export can survive a later auxiliary
evaluation failure; otherwise a bounded safe training retry reduces the
initialization cloud, freezes means, and disables densification. Failure to
produce valid output still fails the task.

### Checkpoint Resume

Each stage writes a `.stage_<name>_done` marker to the task directory (or its
recovery-attempt directory). Retry resumes completed stages and pending
recovery batches; a prior 3D delivery is revalidated before training is reused.
High-quality tasks additionally keep atomic `auto_recovery.json` version 2
journals with pending batches, source attempts, best-attempt scoring, budgets,
delivery tier, and transient retry counts. Automatic recovery allows 32
WorldStereo paths, 90 minutes of recorded video-generation time, and one
additional deterministic generated source. Counters are task-wide, not per
source; the time budget is checked between batches rather than being a total
task-duration timeout. Manual source retry retains accumulated path/time usage
and archives prior downstream output before queuing a new attempt. See
[the retry API](api.md#retry-source) for eligibility and upload requirements.

## Model Weights (persistent node cache)

| Model | Size | Purpose |
|-------|------|---------|
| HY-Pano-2.0 (80B) | 169.4 GB | Panorama generation |
| HY-WorldMirror-2.0 | 5.1 GB | 3D reconstruction |
| WorldStereo-2 (memory + memory-dmd) | 34.9 GB | High-quality and fast world expansion |
| Qwen3-VL-8B-Instruct | 17.5 GB | vLLM trajectory planning |
| facebook/sam3 | 6.9 GB | Semantic segmentation |
| Ruicheng/moge-2-vitl-normal | 1.3 GB | Monocular depth |

These core model weights total approximately 235 GB; auxiliary model and Torch
checkpoints also occupy the cache. The recorded GPU deployment mounts the
node's `/home/cce/gsstudio/models` at `/models` using `hostPath`. The generic
deployment template uses PVCs instead.

## Storage Layout

With `GSSTUDIO_DATA_DIR=/data/tasks` in the GPU manifest, the container layout is:

    /data/tasks/gsstudio.db       # SQLite task queue and metadata
    /data/tasks/tasks/{task_id}/
    +-- input/                    # Uploaded input files
    +-- scene/                    # Pipeline scene directory
    +-- output/                   # Final artifacts (ply, spz, glb)
    +-- logs/                     # Per-stage log files
    +-- auto_recovery.json        # Durable automatic-recovery journal
    +-- recovery_attempts/        # Per-source scenes, reports, archived output
    +-- training_input_manifest.json
    +-- artifact_validation.json

    /models/hf-cache/             # Hugging Face model cache
    /models/torch-cache/          # Torch checkpoints

`/data` maps to `/home/cce/gsstudio/data` on the selected node. These hostPath
directories persist across Pod replacement on that node; they are not portable
PVC storage.

## Docker Image

`Dockerfile.a800` inherits the prebuilt `gsstudio:a800-v22` runtime, preserving
the PyTorch 2.7.1 / CUDA 12.8 stack and compiled CUDA extensions. Release builds
apply Python dependency/upstream patches and copy the API and prebuilt frontend
without rebuilding the model runtimes. The inherited stack includes:

- gsplat_maskgaussian (custom gsplat fork with MaskGaussian)
- navmesh (recastnavigation)
- flash-attn (FlashAttention-2)
- nerfview, fused-ssim, spz, pytorch3d, MoGe

## Key Decisions

1. Single Pod with 5 GPUs - no multi-pod distribution complexity
2. Serial task execution for all task types, including worldrecon
3. vLLM TP=1 on GPU 0 with 60% memory utilization limit
4. High-quality 3DGS uses a fixed 2,000-step schedule on five GPUs; the fast
   preset retains resource-scaled steps (8 GPU -> 5 GPU factor 1.6).
5. Spark 2.1 viewer for browser-based 3DGS preview: the viewer mounts as soon
   as a task is opened and streams/decompresses splat data in a Web Worker with
   visible byte progress and dynamic LoD. Paged coarse-to-fine rendering can be
   added later by publishing prebuilt RAD/RADC artifacts.
6. vLLM isolated in dedicated venv (/opt/vllm): vLLM 0.11.0 is the first
   release with native Qwen3-VL support and requires torch 2.8.0 +
   transformers <5, which conflicts with the main HY-World env
   (torch 2.7.1 + transformers 5.2.0). Pipeline talks to vLLM only over HTTP.

## Recorded Deployment Acceptance (2026-08-12)

The following are historical acceptance facts, not current health claims. See
[the full v50 acceptance record](experiments/2026-08-12-guaranteed-3dgs-delivery.md).

- Image: `ccr-29eug8s3-pub.cnc.bj.baidubce.com/dataflow-poc/gsstudio:a800-v50`
- Digest: `sha256:31781e16e3b5f73f2badef91db24267f1d56a2b9d92ecfa37e58f718440308ef`
- Deployment: `gsstudio` namespace, five A800 GPUs on node 192.168.80.136;
  startup GPU/model preflight passed during acceptance.
- Recorded public IP URL: http://120.48.33.45 (CCE load balancer + Basic Auth)
- Recorded TLS URL: https://geneva-ingredients-expression-located.trycloudflare.com
  (ephemeral Cloudflare Quick Tunnel + Basic Auth)
- One-time world generation prerequisite: approved access to gated
  `facebook/sam3` and `HF_TOKEN` via the optional `gsstudio-huggingface` Secret
- `facebook/sam3` only needs that token for its first download. Its complete
  snapshot is retained under the node's persistent `/models/hf-cache`; later
  tasks and Pod restarts use the local cache without requiring the token.
- Historical task `3dc27ec897d9` completed as `anchored_3d`, with zero accepted
  trajectories, 188 panorama/polar RGB-D anchors, and 419,723 decoded Gaussians.
  The 2,000-step SPZ, Gaussian PLY, matching checkpoints, and viewer metadata
  passed the completion validator.
- Authenticated API/download and browser acceptance verified visible 3DGS,
  fullscreen, WASD movement, reset, and Pointer Lock behavior.
- Panorama preview explicitly selects the source `scene/panorama.png`; generated
  depth/normal maps are never used as the interactive panorama.

The [2026-10-25 investigation](experiments/2026-10-25-service-status-and-source-retry.md)
found a public HTTP connection timeout, HTTP/2 502 from the historical
Cloudflare URL even with resolved public IPs and certificate verification, and
TLS EOF before an HTTP/authentication response from the correct Kubernetes
context. Current Pod state and GPU/inference health remain unconfirmed.
