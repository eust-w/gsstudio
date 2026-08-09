"""Task CRUD and log streaming endpoints."""
from __future__ import annotations

from typing import Optional

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, WebSocket, WebSocketDisconnect

from ..config import Config
from ..db import create_task, delete_task, get_task, list_tasks, update_task
from ..models import TaskDetail, TaskListResponse, TaskResponse, TaskStatus, TaskUpdate
from ..pipeline.stage_runner import stream_log

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tasks"])

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

    # Create task in DB
    task_id = create_task(cfg.DB_PATH, name=name, task_type=type, config=config_dict)

    # Save uploaded files
    input_dir = cfg.task_input_dir(task_id)
    input_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    async def save_upload(upload: UploadFile, filename: Optional[str] = None):
        fname = filename or upload.filename or "unnamed"
        fpath = input_dir / fname
        with open(fpath, "wb") as f:
            while chunk := await upload.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)
        saved_files.append(str(fpath))
        return fpath

    if panorama:
        p = await save_upload(panorama, "panorama.png")
        # For worldgen_from_pano, also copy to scene dir
        if type == "worldgen_from_pano":
            scene_dir = cfg.task_scene_dir(task_id)
            scene_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(p, scene_dir / "panorama.png")

    if image:
        await save_upload(image, "input.png")

    if images:
        for img in images:
            await save_upload(img)

    if video:
        await save_upload(video, "input_video.mp4")

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
    )
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
