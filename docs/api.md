# GSStudio API Reference

Base URL: /api

## Tasks

### Create Task

POST /api/tasks

Content-Type: multipart/form-data

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| type | string | yes | worldgen, worldgen_from_pano, or worldrecon |
| name | string | yes | Task display name |
| prompt | string | no | Text prompt for panorama generation |
| panorama | file | conditional | Panorama image (required for worldgen_from_pano) |
| image | file | no | Optional source image for worldgen; without it, a non-empty prompt is required |
| images | file[] | conditional | Multi-view images for worldrecon, as an alternative to video |
| video | file | no | Video input (alternative to images for worldrecon) |
| config | JSON string | no | Optional pipeline config overrides |

`worldgen` accepts a prompt alone, an image alone, or both. Prompt-only mode
generates a source image before expanding the panorama. Prompts are trimmed
and limited to 2,000 characters; a task with neither an image nor a non-empty
prompt returns `400`. The prompt may also be supplied in `config.prompt`.
Single-image and panorama uploads must decode as supported images and fit the
32 MiB limit. Uploads are staged and normalized before the pending task is
inserted into the worker queue.

For world generation, `quality_preset` defaults to `high` and `recovery_policy`
defaults to `auto`; the `fast` preset forces recovery to `off`. The service
returns `503` if `facebook/sam3` is not cached and no approved `HF_TOKEN` is
available for its initial download.

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

High-quality world generation exposes a durable recovery journal. Quality
selects one of three training delivery tiers; completion still requires valid
3DGS output. An unavailable journal returns `available: false`, optionally with
`source_generation` progress.

    {
      "available": true,
      "state": "completed",
      "delivery_tier": "degraded_3d",
      "training_mode": "accepted_plus_anchors",
      "guaranteed_3d": true,
      "best_source_attempt": 0,
      "accepted_trajectories": 5,
      "required_trajectories": 6,
      "quality_warning": "Only 5 trajectories passed the strict quality gates; 3D was trained from the accepted whitelist with reduced coverage.",
      "recovery_exhausted": true,
      "generated_video_trajectories": 28,
      "video_seconds": 5100,
      "transient_retry_count": 0,
      "limits": {
        "maximum_video_trajectories": 32,
        "maximum_video_seconds": 5400,
        "maximum_source_regenerations": 1
      }
    }

| Strictly accepted trajectories | `delivery_tier` | `training_mode` |
|-------------------------------|-----------------|-----------------|
| 6 or more | `strict_3d` | `accepted_trajectories` |
| 1-5 | `degraded_3d` | `accepted_plus_anchors` |
| 0 | `anchored_3d` | `panorama_anchored` |

All current tiers proceed through `gen_gs_data` and `gs_training`. Rejected
trajectories are not training inputs. The response also includes
`strict_accepted_count`, `anchor_view_count`, `training_input_manifest`,
`artifact_validation`, and `source_generation` when available. Legacy
`panorama_only` journals are mapped to a current tier when read; reading a
journal does not retrain a historical task or validate missing artifacts.

The quality journal's `state: completed` means trajectory recovery has selected
a delivery tier; it does not by itself mean the task has finished training.
Likewise, `guaranteed_3d` describes the delivery policy. Confirm task
`status: completed` and `artifact_validation.valid: true` for a validated real
world-generation delivery. The validator requires an SPZ, Gaussian PLY,
same-step trainer checkpoint, and viewer metadata, and checks decoded Gaussian
count (at least 50,000), finite attributes/opacities, and non-degenerate bounds.

The 32-path and 90-minute budgets apply across source attempts. Time is recorded
for video generation and checked between batches; this is not a 90-minute
end-to-end task timeout. Automatic generated-source recovery permits one extra
source. Manual source retry does not reset cumulative path/time usage.

### Retry From Journal

POST /api/tasks/{task_id}/retry

Retries a `failed` or `cancelled` task using its existing input, configuration,
stage checkpoints, and recovery journal. Completed WorldStereo batches and
partial rescue cameras are reused. The endpoint sets the task to `pending`,
clears the current error/progress, and returns the full task with `200`.
Other statuses return `409`; an unknown task returns `404`.

### Retry Source

POST /api/tasks/{task_id}/retry-source

Queues a replacement source for a `worldgen` or `worldgen_from_pano` task. It
requires an existing automatic-recovery report, `quality_preset: high`, and
`recovery_policy: auto` (both default to these values for older configurations).
Eligible statuses are:

- `failed` or `cancelled`.
- `completed` with `recovery_exhausted: true` and delivery tier `degraded_3d`,
  `anchored_3d`, or legacy `panorama_only`.

Completed `strict_3d` tasks and active (`pending`/`running`) tasks are not
eligible. The frontend applies these requirements and, for `failed`/`cancelled`
tasks, only offers source retry when the recovery state is
`source_quality_blocked` or `quality_blocked`. The API permits other recovery
states for those two task statuses.

Optional uploads use `multipart/form-data`:

| Field | Task type | Behavior |
|-------|-----------|----------|
| `image` | `worldgen` | Optional replacement perspective source; switches to image input mode |
| `panorama` | `worldgen_from_pano` | Required replacement panorama |

A `worldgen` request without an upload reuses its source input mode and queues
the next deterministic seed (`config.seed + next source-attempt index`).
Replacement uploads must have an `image/*` content type, be non-empty,
decodable, and at most 32 MiB. Validation occurs before recovery state is
changed.

On success, downstream stage markers, output, and any promotion journal move
to `recovery_attempts/attempt-NNN/previous_downstream/`. Prior source attempts
remain available, the journal retains attempt history, and the new scene uses
`recovery_attempts/attempt-NNN/scene/`. The task returns to `pending` with its
current error/progress cleared; the response is the full task with `200`.
Candidate/batch state is reset while cumulative video-path/time usage remains.

- `400`: missing replacement panorama for a panorama task, invalid upload
  type/content/size, or a perspective `image` sent to a non-`worldgen` task.
- `409`: unsupported task type/status/configuration, missing recovery report,
  or conflicting downstream archive.
- `404`: task not found.

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

This endpoint is exempt from Basic Auth and reports API liveness only. It does
not check GPU, model, vLLM, or pipeline readiness.

### GPU Status

GET /api/gpu/status

    {
      "gpus": [
        { "index": 0, "name": "A800-SXM4-80GB", "used_by": "vllm", "vram_used_gb": 48.2, "vram_total_gb": 80 }
      ],
      "active_task": "uuid-or-null"
    }

This endpoint requires Basic Auth when configured. In real mode it reads GPU
memory/name data from `nvidia-smi`; `used_by: vllm` is a fixed label for GPU 0,
not process-level ownership detection. `active_task` is the worker's current
task ID. Query failures can return an empty `gpus` list with HTTP 200, and mock
mode returns synthetic GPU data. Verify GPU processes and real task delivery
separately when assessing production health.
