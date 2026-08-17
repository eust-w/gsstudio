"""Artifact listing and download endpoints."""
from __future__ import annotations

from typing import Optional

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import Config
from ..db import get_task
from ..delivery_manifest import ArtifactPathError, resolve_task_artifact

router = APIRouter(tags=["artifacts"])

_config: Optional[Config] = None


def init(config: Config):
    global _config
    _config = config


def _get_config() -> Config:
    if _config is None:
        raise RuntimeError("Router not initialized")
    return _config


@router.get("/tasks/{task_id}/artifacts")
async def list_artifacts(task_id: str):
    cfg = _get_config()
    task = get_task(cfg.DB_PATH, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return {"artifacts": task.get("artifacts", [])}


@router.get("/tasks/{task_id}/artifacts/{file_path:path}")
async def download_artifact(task_id: str, file_path: str):
    cfg = _get_config()
    task = get_task(cfg.DB_PATH, task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    task_dir = cfg.task_dir(task_id)
    try:
        artifact = resolve_task_artifact(task_dir, file_path)
    except ArtifactPathError as exc:
        raise HTTPException(400, "Invalid artifact path") from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "Artifact not found") from exc
    return FileResponse(
        path=str(artifact),
        filename=artifact.name,
        media_type="application/octet-stream",
    )
