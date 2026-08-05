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
| image | file | conditional | Single input image (required for worldgen) |
| images | file[] | conditional | Multi-view images (required for worldrecon) |
| video | file | no | Video input (alternative to images for worldrecon) |
| config | JSON string | no | Optional pipeline config overrides |

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
