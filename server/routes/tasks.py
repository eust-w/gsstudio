"""Task CRUD and log streaming endpoints."""
from __future__ import annotations

from typing import Any, Optional

import asyncio
import hashlib
import json
import logging
import math
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, Query, Response, UploadFile, WebSocket, WebSocketDisconnect

from ..config import Config
from ..db import (
    IdempotencyConflictError,
    create_task_idempotent,
    delete_task,
    get_task,
    list_tasks,
    new_task_id,
    update_task,
)
from ..delivery_manifest import build_delivery_manifest
from ..models import TaskDetail, TaskListResponse, TaskResponse, TaskStatus, TaskUpdate
from ..pipeline.quality import normalize_worldgen_config
from ..pipeline.preflight import (
    CONTRACT_VERSION,
    MINIMUM_MULTIVIEW_FRAMES,
    normalize_task_contract,
    strict_contract,
    write_json_atomic,
)
from ..pipeline.gaussian_repair import (
    GaussianRepairError,
    assess_repair_eligibility,
    request_gaussian_repair,
)
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
MAX_IDEMPOTENCY_KEY_CHARS = 200
GENERATIVE_MINIMUM_SHORT_EDGE = 512

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


def normalize_uploaded_generation_image(source: Path, destination: Path) -> dict[str, Any]:
    """Decode and deterministically normalize a generative source image.

    The strict preflight still evaluates blur, features, exposure, edges, and
    perspective.  This bounded normalization only prevents an otherwise good
    image from failing because a browser or chat client resized it slightly
    below the 512-pixel transport contract.  Original bytes are retained next
    to the normalized PNG for audit.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(source) as decoded:
            decoded.verify()
        with Image.open(source) as decoded:
            image = decoded.convert("RGB")
            original_width, original_height = image.size
            short_edge = min(original_width, original_height)
            if short_edge <= 0:
                raise ValueError("Image dimensions must be positive")
            upscale_applied = short_edge < GENERATIVE_MINIMUM_SHORT_EDGE
            if upscale_applied:
                scale = GENERATIVE_MINIMUM_SHORT_EDGE / short_edge
                normalized_size = (
                    math.ceil(original_width * scale),
                    math.ceil(original_height * scale),
                )
                image = image.resize(normalized_size, Image.Resampling.LANCZOS)
            image.save(destination, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, "Uploaded image is corrupt or unsupported") from exc

    normalized_width, normalized_height = image.size
    return {
        "version": 1,
        "original_name": source.name,
        "original_width": original_width,
        "original_height": original_height,
        "original_short_edge": min(original_width, original_height),
        "normalized_name": destination.name,
        "normalized_width": normalized_width,
        "normalized_height": normalized_height,
        "normalized_short_edge": min(normalized_width, normalized_height),
        "upscale_applied": upscale_applied,
        "resampling": "lanczos" if upscale_applied else "none",
        "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "normalized_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


async def _stage_retry_image(upload: UploadFile, staging_dir: Path) -> tuple[Path, Path]:
    """Validate a replacement before any durable recovery state is changed."""
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
        report = normalize_uploaded_generation_image(raw, normalized)
    except HTTPException as exc:
        raise HTTPException(400, "Replacement source is corrupt or unsupported") from exc
    write_json_atomic(staging_dir / "input_normalization.json", report)
    return raw, normalized


def _sam3_is_cached(hf_home: str) -> bool:
    snapshots = Path(hf_home) / "hub" / "models--facebook--sam3" / "snapshots"
    return any(
        (snapshot / "config.json").is_file()
        and (snapshot / "model.safetensors").is_file()
        for snapshot in snapshots.glob("*")
        if snapshot.is_dir()
    )


def _normalized_idempotency_key(value: str | None) -> str:
    key = str(value or "").strip()
    if not key:
        return ""
    if len(key) > MAX_IDEMPOTENCY_KEY_CHARS:
        raise HTTPException(400, f"Idempotency-Key must be at most {MAX_IDEMPOTENCY_KEY_CHARS} characters")
    if not all(character.isalnum() or character in "._:-" for character in key):
        raise HTTPException(400, "Idempotency-Key contains unsupported characters")
    return key


def _request_fingerprint(
    *,
    task_type: str,
    name: str,
    prompt: str,
    config: dict[str, Any],
    input_dir: Path,
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "type": task_type,
        "name": name,
        "prompt": prompt,
        "config": config,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(input_dir)).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _sanitize_user_config(config: dict[str, Any], task_type: str) -> dict[str, Any]:
    """Reject client overrides of executable paths, hosts, and model storage."""
    public_keys = {"prompt", "seed", "quality_preset", "recovery_policy"}
    if task_type == "worldrecon":
        public_keys.add("recon_gpus")
    ignored_contract_keys = set(strict_contract(task_type))
    unsupported = sorted(set(config) - public_keys - ignored_contract_keys)
    if unsupported:
        raise HTTPException(
            400,
            "config contains unsupported server-managed keys: " + ", ".join(unsupported),
        )
    sanitized = dict(config)
    if "recon_gpus" in sanitized:
        try:
            recon_gpus = int(sanitized["recon_gpus"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "recon_gpus must be an integer from 1 to 5") from exc
        if recon_gpus < 1 or recon_gpus > 5:
            raise HTTPException(400, "recon_gpus must be an integer from 1 to 5")
        sanitized["recon_gpus"] = recon_gpus
    return sanitized


@router.post("/tasks", status_code=201)
async def create_task_endpoint(
    response: Response,
    type: str = Form(...),
    name: str = Form(...),
    prompt: Optional[str] = Form(None),
    panorama: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    images: Optional[list[UploadFile]] = File(None),
    video: Optional[UploadFile] = File(None),
    config: str = Form("{}"),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    cfg = _get_config()
    request_key = _normalized_idempotency_key(idempotency_key)

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
    final_dir: Path | None = None
    try:
        config_dict = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in config field")
    if not isinstance(config_dict, dict):
        raise HTTPException(400, "config field must contain a JSON object")
    config_dict = _sanitize_user_config(config_dict, type)
    if type in ("worldgen", "worldgen_from_pano"):
        try:
            config_dict = normalize_worldgen_config(config_dict, form_prompt=prompt)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    config_dict = normalize_task_contract(config_dict, type)

    normalized_prompt = str(config_dict.get("prompt") or "").strip()
    if len(normalized_prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"Prompt must be at most {MAX_PROMPT_CHARS} characters")
    if type == "worldgen" and image is None and not normalized_prompt:
        raise HTTPException(400, "Text to World requires an image or a non-empty prompt")
    if type == "worldgen_from_pano" and panorama is None:
        raise HTTPException(400, "Panorama to World requires a panorama image")
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

    try:
        if panorama:
            suffix = Path(panorama.filename or "panorama.png").suffix.lower() or ".png"
            original = await save_upload(panorama, f"source_upload{suffix}", max_bytes=MAX_IMAGE_BYTES)
            normalized = input_dir / "panorama.png"
            normalization = normalize_uploaded_generation_image(original, normalized)
            write_json_atomic(input_dir / "panorama_normalization.json", normalization)
            if type == "worldgen_from_pano":
                scene_dir = incoming_dir / "scene"
                scene_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(normalized, scene_dir / "panorama.png")

        if image:
            suffix = Path(image.filename or "input.png").suffix.lower() or ".png"
            original = await save_upload(image, f"source_upload{suffix}", max_bytes=MAX_IMAGE_BYTES)
            normalization = normalize_uploaded_generation_image(original, input_dir / "input.png")
            write_json_atomic(input_dir / "input_normalization.json", normalization)

        if images:
            for index, img in enumerate(images):
                await save_upload(img, f"image_{index:04d}_{Path(img.filename or 'image').name}")

        if video:
            await save_upload(video, "input_video.mp4")

        final_dir = cfg.task_dir(task_id)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        incoming_dir.replace(final_dir)
        write_json_atomic(final_dir / "task_contract.json", strict_contract(type))
        fingerprint = _request_fingerprint(
            task_type=type,
            name=name,
            prompt=normalized_prompt,
            config=config_dict,
            input_dir=final_dir / "input",
        )
        try:
            created_task_id, replayed = create_task_idempotent(
                cfg.DB_PATH,
                name=name,
                task_type=type,
                config=config_dict,
                task_id=task_id,
                idempotency_key=request_key,
                request_fingerprint=fingerprint,
            )
        except Exception:
            shutil.rmtree(final_dir, ignore_errors=True)
            raise
        if replayed:
            shutil.rmtree(final_dir, ignore_errors=True)
            existing = get_task(cfg.DB_PATH, created_task_id)
            if existing is None:
                raise HTTPException(409, "Idempotent task no longer exists")
            response.status_code = 200
            return {
                "task_id": created_task_id,
                "status": existing["status"],
                "created_at": existing["created_at"],
                "idempotent_replay": True,
            }
    except IdempotencyConflictError as exc:
        if final_dir is not None:
            shutil.rmtree(final_dir, ignore_errors=True)
        raise HTTPException(409, str(exc)) from exc
    except HTTPException:
        shutil.rmtree(incoming_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(incoming_dir, ignore_errors=True)
        if final_dir is not None:
            shutil.rmtree(final_dir, ignore_errors=True)
        logger.exception("Atomic task creation failed")
        raise HTTPException(500, "Task input could not be stored safely") from exc

    return {
        "task_id": task_id,
        "status": "pending",
        "created_at": get_task(cfg.DB_PATH, task_id)["created_at"],
        "idempotent_replay": False,
    }


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


@router.get("/tasks/{task_id}/delivery-manifest")
async def get_task_delivery_manifest_endpoint(task_id: str):
    """Return a checksum-backed cross-system delivery contract."""
    cfg = _get_config()
    task = get_task(cfg.DB_PATH, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    try:
        return build_delivery_manifest(cfg.task_dir(task_id), task)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(409, f"Delivery manifest is incomplete: {exc}") from exc


@router.get("/quality/summary")
async def get_quality_summary_endpoint():
    """Aggregate the strict v4 funnel and rolling-20 SLO by input lane."""
    cfg = _get_config()
    tasks: list[dict] = []
    offset = 0
    while True:
        page, total = list_tasks(cfg.DB_PATH, limit=100, offset=offset)
        tasks.extend(page)
        offset += len(page)
        if not page or offset >= total:
            break
    lanes: dict[str, dict[str, Any]] = {}
    eligible_total = 0
    incorrectly_opened = 0
    for task in tasks:
        quality = auto_recovery_summary(cfg.task_dir(task["id"]))
        contract_version = int(quality.get("contract_version", 0) or 0)
        if contract_version < CONTRACT_VERSION:
            continue
        lane = str(quality.get("input_lane") or (task.get("config") or {}).get("input_lane") or "unknown")
        stats = lanes.setdefault(lane, {
            "tasks": 0,
            "preflight_passed": 0,
            "preflight_resolved": 0,
            "strict_completed": 0,
            "failure_stages": {},
            "worldstereo_trajectories": 0,
            "gpu_seconds": 0.0,
            "recent": [],
        })
        stats["tasks"] += 1
        if (quality.get("input_preflight") or {}).get("state") == "passed":
            stats["preflight_passed"] += 1
        resolved = task.get("status") in {"completed", "failed"}
        if resolved and (quality.get("input_preflight") or {}).get("state") == "passed":
            stats["preflight_resolved"] += 1
        if task.get("status") == "completed" and quality.get("free_view_eligible") is True:
            stats["strict_completed"] += 1
        if quality.get("free_view_eligible") is True:
            eligible_total += 1
        if quality.get("free_view_eligible") is True and (quality.get("navigation_stress") or {}).get("state") != "passed":
            incorrectly_opened += 1
        error_code = str(task.get("error_code") or "")
        if task.get("status") == "failed":
            stats["failure_stages"][error_code or "pipeline_failed"] = stats["failure_stages"].get(error_code or "pipeline_failed", 0) + 1
        budget = quality.get("global_budget") or {}
        stats["worldstereo_trajectories"] += int(budget.get("worldstereo_trajectories_used", 0) or 0)
        stats["gpu_seconds"] += float(budget.get("worldstereo_seconds_used", 0.0) or 0.0)
        stats["recent"].append({
            "id": task["id"],
            "created_at": task.get("created_at"),
            "preflight_passed": (quality.get("input_preflight") or {}).get("state") == "passed",
            "resolved": resolved,
            "strict_completed": task.get("status") == "completed" and quality.get("free_view_eligible") is True,
        })
    for stats in lanes.values():
        stats["preflight_pass_rate"] = stats["preflight_passed"] / stats["tasks"] if stats["tasks"] else 0.0
        stats["strict_completion_rate"] = stats["strict_completed"] / stats["preflight_resolved"] if stats["preflight_resolved"] else None
        eligible_recent = sorted(
            [
                item for item in stats.pop("recent")
                if item["preflight_passed"] and item["resolved"]
            ],
            key=lambda item: str(item.get("created_at") or ""),
            reverse=True,
        )[:20]
        stats["recent_20_slo"] = {
            "sample_size": len(eligible_recent),
            "successes": sum(1 for item in eligible_recent if item["strict_completed"]),
            "success_rate": (
                sum(1 for item in eligible_recent if item["strict_completed"]) / len(eligible_recent)
                if eligible_recent else None
            ),
            "target": 0.95,
        }
    return {
        "contract_version": CONTRACT_VERSION,
        "success_policy": "strict_free_view",
        "lanes": lanes,
        "free_view_eligible_tasks": eligible_total,
        "incorrect_wasd_openings": incorrectly_opened,
    }


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


@router.post("/tasks/{task_id}/repair")
async def repair_task_model_endpoint(task_id: str):
    """Queue one bounded model repair without rerunning paid upstream stages."""
    cfg = _get_config()
    task = get_task(cfg.DB_PATH, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["status"] not in (TaskStatus.FAILED.value, TaskStatus.CANCELLED.value):
        raise HTTPException(409, f"Cannot repair task in status: {task['status']}")
    if task["type"] not in ("worldgen", "worldgen_from_pano", "worldrecon"):
        raise HTTPException(409, "This task type does not support model repair")
    task_dir = cfg.task_dir(task_id)
    post_panorama = {}
    try:
        post_panorama = json.loads((task_dir / "post_panorama_preflight.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    if post_panorama.get("state") == "failed":
        raise HTTPException(409, "Generated panorama preflight failed; replace the source instead")
    assessment = assess_repair_eligibility(task_dir)
    if not (assessment.get("eligible") or assessment.get("requires_reanalysis")):
        raise HTTPException(409, str(assessment.get("reason") or "Model repair is not available"))
    try:
        request = request_gaussian_repair(task_dir)
    except GaussianRepairError as exc:
        raise HTTPException(409, str(exc)) from exc
    navigation_marker = task_dir / ".stage_navigation_stress_done"
    if navigation_marker.exists():
        navigation_marker.unlink()
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
    return {
        "task_id": task_id,
        "status": "pending",
        "repair_request": request,
    }


@router.post("/tasks/{task_id}/retry-source")
async def retry_task_source_endpoint(
    task_id: str,
    panorama: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    images: Optional[list[UploadFile]] = File(None),
    video: Optional[UploadFile] = File(None),
):
    """Queue a user-approved new source without weakening the quality gate."""
    cfg = _get_config()
    task = get_task(cfg.DB_PATH, task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["type"] not in ("worldgen", "worldgen_from_pano", "worldrecon"):
        raise HTTPException(409, "This task type does not support source retry")
    task_dir = cfg.task_dir(task_id)
    current = auto_recovery_summary(task_dir)
    if task["status"] not in (TaskStatus.FAILED.value, TaskStatus.CANCELLED.value):
        raise HTTPException(409, f"Cannot replace the source in status: {task['status']}")
    config = dict(task.get("config") or {})
    if task["type"] != "worldrecon" and (
        config.get("quality_preset", "high") != "high" or config.get("recovery_policy", "auto") != "auto"
    ):
        raise HTTPException(409, "Source retry is available only for high-quality tasks with automatic recovery enabled")
    if task["type"] == "worldrecon":
        if not video and not images:
            raise HTTPException(400, "Upload a replacement video or at least 12 multi-view images")
        if images and len(images) < MINIMUM_MULTIVIEW_FRAMES:
            raise HTTPException(400, "At least 12 replacement multi-view images are required")
        retry_id = uuid.uuid4().hex
        staging = task_dir / ".incoming-retry" / retry_id
        staged_input = staging / "input"
        staged_input.mkdir(parents=True, exist_ok=False)
        total_limit = 2 * 1024 * 1024 * 1024
        total = 0
        uploads = images or []
        try:
            for index, upload in enumerate(uploads):
                if not (upload.content_type or "").startswith("image/"):
                    raise HTTPException(400, "Every replacement multi-view file must be an image")
                raw = staged_input / f"image_{index:04d}_{Path(upload.filename or 'image').name}"
                with raw.open("wb") as handle:
                    while chunk := await upload.read(1024 * 1024):
                        total += len(chunk)
                        if total > total_limit:
                            raise HTTPException(400, "Replacement upload exceeds the 2 GiB limit")
                        handle.write(chunk)
                if raw.stat().st_size == 0:
                    raise HTTPException(400, "Replacement image is empty")
            if video:
                if not (video.content_type or "").startswith("video/"):
                    raise HTTPException(400, "Replacement video must be a video file")
                raw = staged_input / "input_video.mp4"
                with raw.open("wb") as handle:
                    while chunk := await video.read(1024 * 1024):
                        total += len(chunk)
                        if total > total_limit:
                            raise HTTPException(400, "Replacement upload exceeds the 2 GiB limit")
                        handle.write(chunk)
                if raw.stat().st_size == 0:
                    raise HTTPException(400, "Replacement video is empty")
            if uploads:
                from PIL import Image, UnidentifiedImageError
                for path in sorted(staged_input.glob("image_*")):
                    try:
                        with Image.open(path) as decoded:
                            decoded.verify()
                    except (UnidentifiedImageError, OSError, ValueError) as exc:
                        raise HTTPException(400, f"Replacement image is corrupt: {path.name}") from exc
            if video:
                import cv2
                capture = cv2.VideoCapture(str(staged_input / "input_video.mp4"))
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                ok, _ = capture.read()
                capture.release()
                if not ok or frame_count < 24:
                    raise HTTPException(400, "Replacement video must contain at least 24 decodable frames")
            replacement_root = task_dir / "source_replacements"
            replacement_root.mkdir(parents=True, exist_ok=True)
            attempt = len([item for item in replacement_root.glob("attempt-*") if item.is_dir()]) + 1
            archive = replacement_root / f"attempt-{attempt:03d}"
            archive.mkdir(parents=True, exist_ok=False)
            moved: list[tuple[Path, Path]] = []
            replacement_input = task_dir / "input"
            try:
                for name in ("input", "output"):
                    current_path = task_dir / name
                    if current_path.exists():
                        destination = archive / name
                        os.replace(current_path, destination)
                        moved.append((destination, current_path))
                for name in (
                    "input_preflight.json", "post_panorama_preflight.json", "source_pool.json",
                    "navigation_stress.json", "viewer_metadata_validation.json",
                    "artifact_validation.json", "gaussian_repair.json",
                    "gaussian_repair_request.json",
                ):
                    current_path = task_dir / name
                    if current_path.exists():
                        destination = archive / name
                        os.replace(current_path, destination)
                        moved.append((destination, current_path))
                for marker in sorted(task_dir.glob(".stage_*_done")):
                    destination = archive / marker.name
                    os.replace(marker, destination)
                    moved.append((destination, marker))
                os.replace(staged_input, replacement_input)
                write_json_atomic(task_dir / "source_retry.json", {
                    "version": CONTRACT_VERSION,
                    "state": "queued",
                    "source_attempt": attempt,
                    "archive_path": str(archive.relative_to(task_dir)),
                    "input_mode": "video" if video else "images",
                })
                update_task(
                    cfg.DB_PATH, task_id, status=TaskStatus.PENDING.value, progress=0.0,
                    current_stage=None, error=None, error_code=None, error_summary=None,
                    config=normalize_task_contract(config, "worldrecon"),
                )
            except Exception:
                # All validation happened before this transaction. If any
                # same-filesystem promotion fails, restore every archived path
                # in reverse order so the task never observes a half retry.
                if replacement_input.exists() and not staged_input.exists():
                    staged_input.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(replacement_input, staged_input)
                for archived, original in reversed(moved):
                    if archived.exists() and not original.exists():
                        os.replace(archived, original)
                retry_report = task_dir / "source_retry.json"
                if retry_report.exists():
                    retry_report.unlink()
                try:
                    archive.rmdir()
                except OSError:
                    pass
                raise
            shutil.rmtree(staging, ignore_errors=True)
            return get_task(cfg.DB_PATH, task_id)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
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
        staged_report = staged_normalized.parent / "input_normalization.json"
        if staged_report.is_file():
            shutil.copy2(staged_report, attempt_input / "input_normalization.json")
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
