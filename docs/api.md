# GSStudio API Reference

Base URL: /api

## Tasks

### Create Task

POST /api/tasks

Content-Type: multipart/form-data

Long-running workflow clients should send a stable `Idempotency-Key` header,
for example `droboticflow:<run_id>:<node_id>`. Replaying the identical request
returns the original task with `idempotent_replay=true` and does not enqueue a
second GPU job. Reusing the key for different metadata or file bytes returns
`409 Conflict`.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| type | string | yes | worldgen, worldgen_from_pano, or worldrecon |
| name | string | yes | Task display name |
| prompt | string | no | Text prompt for panorama generation |
| panorama | file | conditional | Panorama image (required for worldgen_from_pano) |
| image | file | conditional | Single input image (required for worldgen) |
| images | file[] | conditional | At least 12 multi-view images for the recommended worldrecon lane |
| video | file | no | Video input; 24 candidates are sampled before frame/parallax filtering |
| config | JSON string | no | Optional safe config (`seed`, `recon_gpus=1..5`); v4 policy fields are ignored and executable paths, hosts, model storage, budgets, and quality thresholds cannot be overridden |

The server always writes `contract_version=4`,
`success_policy=strict_free_view`, and either
`input_lane=production_multiview` or `input_lane=generative_worldgen`.
There is no Fast or degraded Worldgen contract.

Response: 201 Created

    {
      "task_id": "uuid-string",
      "status": "pending",
      "created_at": "2026-08-06T00:00:00Z"
    }

### List Tasks

GET /api/tasks?status=&type=&limit=20&offset=0

Response: 200 OK

    {
      "tasks": [
        {
          "id": "uuid",
          "name": "My Task",
          "type": "worldgen",
          "status": "running",
          "current_stage": "video_gen",
          "progress": 0.45,
          "created_at": "2026-08-06T00:00:00Z"
        }
      ],
      "total": 1
    }

### Get Task Detail

GET /api/tasks/{task_id}

Response: 200 OK

    {
      "id": "uuid",
      "name": "My Task",
      "type": "worldgen",
      "status": "running",
      "stages": [
        {
          "name": "panogen",
          "status": "completed",
          "started_at": "...",
          "completed_at": "...",
          "duration_seconds": 1200
        }
      ],
      "config": {},
      "artifacts": [
        { "name": "panorama.png", "path": "input/panorama.png", "size": 1024000, "type": "image" }
      ],
      "created_at": "...",
      "error": null
    }

### Cancel Task

POST /api/tasks/{task_id}/cancel

Response: 200 OK

    { "status": "cancelled" }

### Get Delivery Quality

GET /api/tasks/{task_id}/quality

Every v4 task exposes the strict contract and its closed-loop gate reports.
Only input preflight, strict geometry, pilot, final validation, and navigation
stress all passing can set `free_view_eligible=true`. Insufficient coverage or
exhausted sources are failed explicitly; evidence remains downloadable.

    {
      "available": true,
      "contract_version": 4,
      "input_lane": "generative_worldgen",
      "state": "completed",
      "delivery_tier": "strict_3d",
      "free_view_eligible": true,
      "best_source_attempt": 0,
      "accepted_trajectories": 10,
      "required_trajectories": 6,
      "target_trajectories": 10,
      "source_pool": {"state": "passed", "accepted_candidate_count": 3},
      "input_preflight": {"state": "passed"},
      "pilot_validation": {
        "state": "passed",
        "steps": 500,
        "validation_render_count": 4,
        "metrics": {"psnr": 23.1, "ssim": 0.79, "lpips": 0.20}
      },
      "navigation_validation": {
        "free_view_eligible": true,
        "unique_translated_camera_centers": 12,
        "camera_obb_half_width": 2.1,
        "camera_obb_half_height": 1.7,
        "human_scale": 1.0,
        "azimuth_bin_count": 6,
        "validation_render_count": 8,
        "metrics": {"psnr": 28.4, "ssim": 0.94, "lpips": 0.07},
        "failure_reasons": []
      },
      "recovery_exhausted": false,
      "navigation_stress": {
        "state": "passed",
        "free_view_eligible": true,
        "walk": {"seed": 1024, "completed_steps": 1000, "accepted_steps": 731}
      },
      "generated_video_trajectories": 12,
      "video_seconds": 3200,
      "transient_retry_count": 0,
      "source_guidance_collapse": false,
      "guidance_diagnostics": {
        "diagnostic_cutoff": 0.6,
        "best_minimum_guidance_mask": 0.87
      },
      "limits": {
        "maximum_video_trajectories": 24,
        "maximum_video_seconds": 14400,
        "maximum_source_regenerations": 2,
        "maximum_source_attempts": 3,
        "maximum_trajectories_per_source": 16
      }
    }

For new Worldgen tasks, `delivery_tier` is either `strict_3d` or `null`.
Recovery targets ten accepted trajectories; six is the absolute training
minimum. Rejected trajectories never enter GS preparation or either training.
The 500-step pilot requires four renders, finite PSNR >= 22, SSIM >= 0.75,
LPIPS <= 0.25, at least 50,000 Gaussians, and non-degenerate bounds. Final
validation additionally requires
12 translated camera centers, navigable camera OBB half-extents, six 45-degree
azimuth bins, eight validation renders, PSNR at least 28, SSIM at least 0.90,
and LPIPS at most 0.10. All values must be finite. Panorama and polar anchors
may supplement strict data but cannot form a delivery by themselves.

The final navigation stress performs 1,000 bounded steps, 32 representative
views, and 16 micro-translation pairs. Worldrecon without a v4 report remains
rotation-only as `Legacy unvalidated`; WASD is disabled.

For generative lanes, `post_panorama_preflight` evaluates the exact panorama
after HY-Pano expansion and before trajectory planning. This closes the gap
where a clean source candidate could expand into a panorama containing people,
reflections, curved geometry, or conflicting perspective.

The quality response also includes stable product fields:
`product_state`, `user_message`, `next_action`, `repair_eligibility`,
`repair_report`, and `post_panorama_preflight`. Product states are
`processing`, `repair_available`, `repairing`,
`source_replacement_required`, or `ready`.

### Get Delivery Manifest

GET /api/tasks/{task_id}/delivery-manifest

Returns the cross-system `gsstudio.delivery/v1` contract. Every registered
artifact includes its task-relative path, byte size, MIME type, semantic role,
SHA-256 digest, and authenticated download path. The manifest repeats the task
status, input lane, strict delivery tier, product state, and
`free_view_eligible` decision so workflow systems do not infer delivery from a
PLY/SPZ file alone.

Terminal tasks also persist `delivery_manifest.json` in the task directory.
Artifact downloads reject absolute paths and symlinks that resolve outside the
owning task directory.

### Retry From Journal

POST /api/tasks/{task_id}/retry

Retries a failed or cancelled task from its exact durable checkpoint. Completed
WorldStereo batches and partial rescue cameras are not recomputed.

### Retry Source

POST /api/tasks/{task_id}/retry-source

Queues the next validated ranked source for generated-source tasks. A
generated-source task may instead provide multipart field `image` to replace a
weak generated view with a real perspective image. A panorama-based task must
provide multipart field `panorama`. This endpoint accepts failed or cancelled
Worldgen tasks with a recovery journal. For Worldrecon, send `images` (12 or
more) or `video`; the upload is validated and staged before the journal or
current source is changed. Quality-blocked historical tasks are
migrated to `failed` first; existing attempts, logs, and artifacts are archived
or preserved rather than overwritten.

### Repair Sparse Giant Gaussians

POST /api/tasks/{task_id}/repair

Queues one bounded repair for a failed navigation test. It is available only
when every failure is a giant-Gaussian view failure and the offender set is at
most 500 Gaussians and 0.05% of the model. Reports from older releases can be
queued for deterministic renderer re-analysis first. Original PLY, SPZ,
checkpoint shards, and metadata are archived; repaired artifacts are promoted
atomically and must pass the complete 1,000-step, 32-view, 16-pair test again.
There is no second repair attempt and no threshold relaxation.

### Quality Summary

GET /api/quality/summary

Returns per-lane preflight pass rate, strict completion rate, failure-stage
counts, trajectory/GPU time, and the rolling 20-task success rate against the
0.95 production target. `incorrect_wasd_openings` must remain zero.

### Task Logs (WebSocket)

GET /api/tasks/{task_id}/logs

WebSocket endpoint. Streams JSON messages:

    { "stage": "video_gen", "timestamp": "...", "line": "log line text" }

### List Artifacts

GET /api/tasks/{task_id}/artifacts

Response: 200 OK

    {
      "artifacts": [
        { "name": "output.ply", "path": "output/output.ply", "size": 50000000, "type": "3dgs" }
      ]
    }

### Download Artifact

GET /api/tasks/{task_id}/artifacts/{file_path}

Returns the file with appropriate Content-Type and Content-Disposition headers.

## Health & Status

### Health Check

GET /api/health

    { "status": "ok", "version": "0.1.0" }

### GPU Status

GET /api/gpu/status

    {
      "gpus": [
        { "index": 0, "name": "A800-SXM4-80GB", "used_by": "vllm", "vram_used_gb": 48.2, "vram_total_gb": 80 }
      ],
      "active_task": "uuid-or-null"
    }
