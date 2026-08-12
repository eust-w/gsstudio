"""Task CRUD and log streaming endpoints."""
from __future__ import annotations

from typing import Optional

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, WebSocket, WebSocketDisconnect

from ..config import Config
from ..db import create_task, delete_task, get_task, list_tasks, new_task_id, update_task
from ..models import TaskDetail, TaskListResponse, TaskResponse, TaskStatus, TaskUpdate
from ..pipeline.quality import normalize_worldgen_config
from ..pipeline.recovery import (
    AutoRecoveryError,
    archive_downstream_checkpoints,
    auto_recovery_summary,
    request_source_retry,
)
from ..pipeline.stage_runner import stream_log

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tasks"])
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_PROMPT_CHARS = 2000

# Will be set by main.py
_config: Optional[Config] = None
_worker = None


def init(config: Config, worker):
    global _config, _worker
    _config = config
    _worker = worker


def _get_config() -> Config:
    if _config is None:
        raise RuntimeError("Router not initialized")
    return _config


def cleanup_stale_incoming(config: Config, *, max_age_seconds: int = 86400) -> int:
    """Remove abandoned pre-commit uploads that have no database task."""
    root = config.tasks_dir / ".incoming"
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for candidate in root.iterdir():
        if not candidate.is_dir() or candidate.stat().st_mtime >= cutoff:
            continue
        if get_task(config.DB_PATH, candidate.name):
            continue
        shutil.rmtree(candidate)
        removed += 1
    return removed


async def _stage_retry_image(upload: UploadFile, staging_dir: Path) -> tuple[Path, Path]:
    """Validate a replacement before any durable recovery state is changed."""
    from PIL import Image, UnidentifiedImageError

    staging_dir.mkdir(parents=True, exist_ok=False)
    suffix = Path(upload.filename or "source.png").suffix.lower() or ".png"
    raw = staging_dir / f"source_upload{suffix}"
    total = 0
    with raw.open("wb") as destination:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise HTTPException(400, "Replacement source exceeds the 32 MiB limit")
            destination.write(chunk)
    if total == 0:
        raise HTTPException(400, "Replacement source is empty")
    normalized = staging_dir / "input.png"
    try:
        with Image.open(raw) as decoded:
            decoded.verify()
        with Image.open(raw) as decoded:
            decoded.convert("RGB").save(normalized, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, "Replacement source is corrupt or unsupported") from exc
    return raw, normalized


def _sam3_is_cached(hf_home: str) -> bool:
    snapshots = Path(hf_home) / "hub" / "models--facebook--sam3" / "snapshots"
    return any(
        (snapshot / "config.json").is_file()
        and (snapshot / "model.safetensors").is_file()
        for snapshot in snapshots.glob("*")
        if snapshot.is_dir()
    )


@router.post("/tasks", status_code=201)
async def create_task_endpoint(
    type: str = Form(...),
    name: str = Form(...),
    prompt: Optional[str] = Form(None),
    panorama: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    images: Optional[list[UploadFile]] = File(None),
    video: Optional[UploadFile] = File(None),
    config: str = Form("{}"),
):
    cfg = _get_config()

    # Validate task type
    if type not in ("worldgen", "worldgen_from_pano", "worldrecon"):
        raise HTTPException(400, f"Invalid task type: {type}")

    if (
        type in ("worldgen", "worldgen_from_pano")
        and not os.environ.get("HF_TOKEN")
        and not _sam3_is_cached(cfg.HF_HOME)
    ):
        raise HTTPException(
            503,
            "World generation requires a one-time download of facebook/sam3. "
            "Approve access and configure the gsstudio-huggingface Kubernetes "
            "Secret; after the model is cached, the token is no longer required.",
        )

    # Parse config
    try:
        config_dict = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in config field")
    if not isinstance(config_dict, dict):
        raise HTTPException(400, "config field must contain a JSON object")
    if type in ("worldgen", "worldgen_from_pano"):
        try:
            config_dict = normalize_worldgen_config(config_dict, form_prompt=prompt)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    normalized_prompt = str(config_dict.get("prompt") or "").strip()
    if len(normalized_prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"Prompt must be at most {MAX_PROMPT_CHARS} characters")
    if type == "worldgen" and image is None and not normalized_prompt:
        raise HTTPException(400, "Text to World requires an image or a non-empty prompt")
    if type == "worldgen":
        config_dict["input_mode"] = "image" if image is not None else "prompt"

    # The SQLite pending row is the worker queue. Stage and validate every
    # byte before inserting that row so a worker can never observe half an
    # upload.
    task_id = new_task_id()
    incoming_dir = cfg.tasks_dir / ".incoming" / task_id
    input_dir = incoming_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)

    saved_files = []

    async def save_upload(upload: UploadFile, filename: Optional[str] = None, *, max_bytes: int | None = None):
        fname = Path(filename or upload.filename or "unnamed").name
        fpath = input_dir / fname
        total = 0
        with open(fpath, "wb") as f:
            while chunk := await upload.read(1024 * 1024):  # 1MB chunks
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise HTTPException(400, f"Image exceeds the {max_bytes // (1024 * 1024)} MiB limit")
                f.write(chunk)
        if total == 0:
            raise HTTPException(400, "Uploaded file is empty")
        saved_files.append(str(fpath))
        return fpath

    def normalize_image(source: Path, destination: Path) -> None:
        try:
            from PIL import Image, UnidentifiedImageError
            with Image.open(source) as decoded:
                decoded.verify()
            with Image.open(source) as decoded:
                decoded.convert("RGB").save(destination, format="PNG")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(400, "Uploaded image is corrupt or unsupported") from exc

    try:
        if panorama:
            suffix = Path(panorama.filename or "panorama.png").suffix.lower() or ".png"
            original = await save_upload(panorama, f"source_upload{suffix}", max_bytes=MAX_IMAGE_BYTES)
            normalized = input_dir / "panorama.png"
            normalize_image(original, normalized)
            if type == "worldgen_from_pano":
                scene_dir = incoming_dir / "scene"
                scene_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(normalized, scene_dir / "panorama.png")

        if image:
            suffix = Path(image.filename or "input.png").suffix.lower() or ".png"
            original = await save_upload(image, f"source_upload{suffix}", max_bytes=MAX_IMAGE_BYTES)
            normalize_image(original, input_dir / "input.png")

        if images:
            for index, img in enumerate(images):
                await save_upload(img, f"image_{index:04d}_{Path(img.filename or 'image').name}")

        if video:
            await save_upload(video, "input_video.mp4")

        final_dir = cfg.task_dir(task_id)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        incoming_dir.replace(final_dir)
        try:
            create_task(cfg.DB_PATH, name=name, task_type=type, config=config_dict, task_id=task_id)
        except Exception:
            shutil.rmtree(final_dir, ignore_errors=True)
            raise
    except HTTPException:
        shutil.rmtree(incoming_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(incoming_dir, ignore_errors=True)
        logger.exception("Atomic task creation failed")
        raise HTTPException(500, "Task input could not be stored safely") from exc

    return {"task_id": task_id, "status": "pending", "created_at": get_task(cfg.DB_PATH, task_id)["created_at"]}


@router.get("/tasks")
async def list_tasks_endpoint(
    status: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    cfg = _get_config()
    tasks, total = list_tasks(cfg.DB_PATH, status=status, task_type=type, limit=limit, offset=offset)
    return {"tasks": tasks, "total": total}


@router.get("/tasks/{task_id}")
async def get_task_endpoint(task_id: str):
    cfg = _get_config()
    task = get_task(cfg.DB_PATH, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task


@router.get("/tasks/{task_id}/quality")
async def get_task_quality_endpoint(task_id: str):
    """Poll the compact automatic-recovery state without downloading reports."""
    cfg = _get_config()
    if not get_task(cfg.DB_PATH, task_id):
        raise HTTPException(404, "Task not found")
    return auto_recovery_summary(cfg.task_dir(task_id))


@router.patch("/tasks/{task_id}")
async def update_task_endpoint(task_id: str, payload: TaskUpdate):
    cfg = _get_config()
    task = get_task(cfg.DB_PATH, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    name = payload.name.strip()
    if not name:
        raise HTTPException(400, "Task name cannot be blank")

    update_task(cfg.DB_PATH, task_id, name=name)
    return get_task(cfg.DB_PATH, task_id)


@router.delete("/tasks/{task_id}", status_code=204)
async def delete_task_endpoint(task_id: str):
    cfg = _get_config()
    task = get_task(cfg.DB_PATH, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["status"] in (TaskStatus.PENDING.value, TaskStatus.RUNNING.value):
        raise HTTPException(409, "Cancel the active task before deleting it")

    task_dir = cfg.task_dir(task_id)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    delete_task(cfg.DB_PATH, task_id)
    return Response(status_code=204)


@router.post("/tasks/{task_id}/cancel")
async def cancel_task_endpoint(task_id: str):
    cfg = _get_config()
    task = get_task(cfg.DB_PATH, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    if task["status"] not in ("pending", "running"):
        raise HTTPException(400, f"Cannot cancel task in status: {task['status']}")

    if _worker:
        await _worker.cancel_task(task_id)
    else:
        update_task(cfg.DB_PATH, task_id, status=TaskStatus.CANCELLED.value)

    return {"status": "cancelled"}


@router.post("/tasks/{task_id}/retry")
async def retry_task_endpoint(task_id: str):
    cfg = _get_config()
    task = get_task(cfg.DB_PATH, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["status"] not in (TaskStatus.FAILED.value, TaskStatus.CANCELLED.value):
        raise HTTPException(409, f"Cannot retry task in status: {task['status']}")

    update_task(
        cfg.DB_PATH,
        task_id,
        status=TaskStatus.PENDING.value,
        progress=0.0,
        current_stage=None,
        error=None,
        error_code=None,
        error_summary=None,
    )
    return get_task(cfg.DB_PATH, task_id)


@router.post("/tasks/{task_id}/retry-source")
async def retry_task_source_endpoint(
    task_id: str,
    panorama: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
):
    """Queue a user-approved new source without weakening the quality gate."""
    cfg = _get_config()
    task = get_task(cfg.DB_PATH, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["type"] not in ("worldgen", "worldgen_from_pano"):
        raise HTTPException(409, "Only World Generation tasks support source retry")
    task_dir = cfg.task_dir(task_id)
    current = auto_recovery_summary(task_dir)
    completed_recovery_retry = (
        task["status"] == TaskStatus.COMPLETED.value
        and current.get("recovery_exhausted") is True
        and current.get("delivery_tier") in {"degraded_3d", "panorama_only"}
    )
    if task["status"] not in (TaskStatus.FAILED.value, TaskStatus.CANCELLED.value) and not completed_recovery_retry:
        raise HTTPException(409, f"Cannot replace the source in status: {task['status']}")
    config = dict(task.get("config") or {})
    if config.get("quality_preset", "high") != "high" or config.get("recovery_policy", "auto") != "auto":
        raise HTTPException(409, "Source retry is available only for high-quality tasks with automatic recovery enabled")
    if task["type"] == "worldgen_from_pano" and panorama is None:
        raise HTTPException(400, "Upload a replacement panorama to retry this panorama-based task")
    if panorama is not None and not (panorama.content_type or "").startswith("image/"):
        raise HTTPException(400, "Replacement panorama must be an image")
    if image is not None and task["type"] != "worldgen":
        raise HTTPException(400, "Replacement perspective images are supported only for Text to World")
    if image is not None and not (image.content_type or "").startswith("image/"):
        raise HTTPException(400, "Replacement source must be an image")

    retry_staging: Path | None = None
    staged_raw: Path | None = None
    staged_normalized: Path | None = None
    try:
        if not current.get("available"):
            raise AutoRecoveryError("Source retry requires an automatic recovery report")
        if image is not None or panorama is not None:
            retry_staging = task_dir / ".incoming-retry" / uuid.uuid4().hex
            staged_raw, staged_normalized = await _stage_retry_image(image or panorama, retry_staging)
        source_attempt = int(current.get("source_attempt", 0))
        base_seed = int(config.get("seed", 1024))
        # Archive downstream state before mutating the journal. If the archive
        # cannot be made safely, the failed task remains exactly as it was.
        archive_downstream_checkpoints(task_dir, source_attempt=source_attempt + 1)
        report = request_source_retry(
            task_dir,
            task_type=task["type"],
            seed=base_seed + source_attempt + 1 if task["type"] == "worldgen" else None,
        )
    except HTTPException:
        if retry_staging is not None:
            shutil.rmtree(retry_staging, ignore_errors=True)
        raise
    except (AutoRecoveryError, TypeError, ValueError) as exc:
        if retry_staging is not None:
            shutil.rmtree(retry_staging, ignore_errors=True)
        raise HTTPException(409, str(exc)) from exc

    if panorama is not None:
        assert staged_raw is not None and staged_normalized is not None
        source_scene = task_dir / str(report["active_scene_relpath"])
        source_scene.mkdir(parents=True, exist_ok=True)
        input_dir = cfg.task_input_dir(task_id)
        input_dir.mkdir(parents=True, exist_ok=True)
        attempt = int(report["source_attempt"])
        upload_copy = input_dir / f"replacement-panorama-attempt-{attempt:03d}{staged_raw.suffix}"
        shutil.copy2(staged_raw, upload_copy)
        shutil.copy2(staged_normalized, source_scene / "panorama.png")

    if image is not None:
        assert staged_raw is not None and staged_normalized is not None
        source_scene = task_dir / str(report["active_scene_relpath"])
        attempt_input = source_scene.parent / "input"
        attempt_input.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged_raw, attempt_input / staged_raw.name)
        shutil.copy2(staged_normalized, attempt_input / "input.png")
        config["input_mode"] = "image"

    update_task(
        cfg.DB_PATH,
        task_id,
        status=TaskStatus.PENDING.value,
        progress=0.0,
        current_stage=None,
        error=None,
        error_code=None,
        error_summary=None,
        config=config,
    )
    if retry_staging is not None:
        shutil.rmtree(retry_staging, ignore_errors=True)
    return get_task(cfg.DB_PATH, task_id)


@router.websocket("/tasks/{task_id}/logs")
async def task_logs_websocket(websocket: WebSocket, task_id: str):
    await websocket.accept()
    cfg = _get_config()

    task = get_task(cfg.DB_PATH, task_id)
    if not task:
        await websocket.close(code=4004, reason="Task not found")
        return

    # Stream logs from all stage log files
    logs_dir = cfg.task_logs_dir(task_id)
    current_stage = task.get("current_stage")

    if current_stage:
        log_file = logs_dir / f"{current_stage}.log"
    else:
        # Find the most recent log file
        log_files = sorted(logs_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
        log_file = log_files[0] if log_files else None

    if not log_file or not log_file.exists():
        await websocket.send_json({"stage": "", "timestamp": "", "line": "No logs available yet."})
        await websocket.close()
        return

    try:
        async for line in stream_log(log_file):
            await websocket.send_json({
                "stage": current_stage or "",
                "timestamp": "",
                "line": line,
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.close()
        except Exception:
            pass
