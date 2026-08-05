"""Async pipeline worker - serial task execution with checkpoint resume."""
from __future__ import annotations

from typing import Optional

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..db import get_task, list_tasks, set_task_stages, update_stage, update_task
from ..models import StageStatus, TaskStatus
from .stage_runner import run_stage
from .worldgen import get_stage_names, get_worldgen_commands
from .worldrecon import get_worldrecon_commands

logger = logging.getLogger(__name__)

# Artifact type classification
ARTIFACT_TYPES = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    ".mp4": "video", ".mov": "video", ".avi": "video", ".webm": "video",
    ".ply": "3dgs", ".spz": "3dgs", ".splat": "3dgs",
    ".pcd": "pointcloud", ".xyz": "pointcloud",
    ".glb": "mesh", ".gltf": "mesh", ".obj": "mesh",
    ".json": "data", ".npy": "data", ".pt": "data",
}


class PipelineWorker:
    def __init__(self, config: Config):
        self.config = config
        self._running = False
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._current_task_id: Optional[str] = None

    async def start(self):
        """Main worker loop: poll for pending tasks and execute serially."""
        self._running = True
        logger.info("Pipeline worker started (mock=%s)", self.config.MOCK_PIPELINE)

        while self._running:
            try:
                # Find next pending task
                tasks, _ = list_tasks(self.config.DB_PATH, status="pending", limit=1)
                if tasks:
                    task = tasks[0]
                    await self.execute_task(task["id"])
                else:
                    await asyncio.sleep(2)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Worker loop error")
                await asyncio.sleep(5)

        logger.info("Pipeline worker stopped")

    async def stop(self):
        self._running = False

    async def cancel_task(self, task_id: str):
        """Signal cancellation for a running task."""
        if task_id in self._cancel_events:
            self._cancel_events[task_id].set()
        update_task(self.config.DB_PATH, task_id, status=TaskStatus.CANCELLED.value)

    async def execute_task(self, task_id: str):
        """Execute a single task through its pipeline stages."""
        task = get_task(self.config.DB_PATH, task_id)
        if not task:
            logger.error("Task %s not found", task_id)
            return

        self._current_task_id = task_id
        cancel_event = asyncio.Event()
        self._cancel_events[task_id] = cancel_event

        task_dir = self.config.task_dir(task_id)
        scene_dir = self.config.task_scene_dir(task_id)
        input_dir = self.config.task_input_dir(task_id)
        output_dir = self.config.task_output_dir(task_id)
        logs_dir = self.config.task_logs_dir(task_id)

        for d in (task_dir, scene_dir, input_dir, output_dir, logs_dir):
            d.mkdir(parents=True, exist_ok=True)

        task_type = task["type"]
        task_config = task.get("config", {})
        task_config.setdefault("hf_home", self.config.HF_HOME)
        task_config.setdefault("num_gpus", self.config.NUM_GPUS)

        # Set up stage list
        stage_names = get_stage_names(task_type)
        set_task_stages(self.config.DB_PATH, task_id, stage_names)
        update_task(self.config.DB_PATH, task_id, status=TaskStatus.RUNNING.value)

        logger.info("Executing task %s (%s): %d stages", task_id, task_type, len(stage_names))

        try:
            if self.config.MOCK_PIPELINE:
                await self._execute_mock(task_id, stage_names, cancel_event, task_dir, logs_dir)
            else:
                await self._execute_real(task_id, task_type, task_config, task_dir, scene_dir, input_dir, logs_dir, cancel_event)

            # Check if cancelled
            if cancel_event.is_set():
                update_task(self.config.DB_PATH, task_id, status=TaskStatus.CANCELLED.value)
                return

            # Collect artifacts
            artifacts = self._collect_artifacts(output_dir, scene_dir)
            update_task(
                self.config.DB_PATH, task_id,
                status=TaskStatus.COMPLETED.value,
                progress=1.0,
                artifacts=artifacts,
            )
            logger.info("Task %s completed", task_id)

        except Exception as e:
            logger.exception("Task %s failed", task_id)
            update_task(
                self.config.DB_PATH, task_id,
                status=TaskStatus.FAILED.value,
                error=str(e),
            )
        finally:
            self._cancel_events.pop(task_id, None)
            self._current_task_id = None

    async def _execute_real(
        self, task_id, task_type, task_config, task_dir, scene_dir, input_dir, logs_dir, cancel_event
    ):
        """Execute real pipeline stages."""
        if task_type in ("worldgen", "worldgen_from_pano"):
            commands = get_worldgen_commands(
                task_dir=task_dir,
                scene_dir=scene_dir,
                input_dir=input_dir,
                config=task_config,
                hyworld_repo=self.config.HYWORLD_REPO,
                vllm_host=self.config.VLLM_HOST,
                vllm_port=self.config.VLLM_PORT,
                vllm_model=self.config.VLLM_MODEL,
                task_type=task_type,
            )
        elif task_type == "worldrecon":
            commands = get_worldrecon_commands(
                task_dir=task_dir,
                input_dir=input_dir,
                config=task_config,
                hyworld_repo=self.config.HYWORLD_REPO,
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        for stage_name, cmd, env, cwd in commands:
            if cancel_event.is_set():
                return

            # Checkpoint resume: skip completed stages
            marker = task_dir / f".stage_{stage_name}_done"
            if marker.exists():
                logger.info("Stage %s already completed, skipping", stage_name)
                update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.SKIPPED)
                continue

            log_file = logs_dir / f"{stage_name}.log"
            update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.RUNNING)

            success, error = await run_stage(
                stage_name=stage_name,
                cmd=cmd,
                env=env,
                log_file=log_file,
                cwd=cwd,
                cancel_event=cancel_event,
            )

            if success:
                marker.touch()
                update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.COMPLETED)
            else:
                update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.FAILED, error=error)
                raise RuntimeError(f"Stage {stage_name} failed: {error}")

    async def _execute_mock(self, task_id, stage_names, cancel_event, task_dir, logs_dir):
        """Execute mock pipeline for local development."""
        for stage_name in stage_names:
            if cancel_event.is_set():
                return

            marker = task_dir / f".stage_{stage_name}_done"
            if marker.exists():
                update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.SKIPPED)
                continue

            log_file = logs_dir / f"{stage_name}.log"
            update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.RUNNING)

            # Simulate stage execution with fake log lines
            fake_logs = [
                f"[{stage_name}] Starting...",
                f"[{stage_name}] Loading models...",
                f"[{stage_name}] Processing...",
                f"[{stage_name}] Progress: 25%",
                f"[{stage_name}] Progress: 50%",
                f"[{stage_name}] Progress: 75%",
                f"[{stage_name}] Finalizing...",
                f"[{stage_name}] Done.",
            ]

            with open(log_file, "w") as f:
                for line in fake_logs:
                    if cancel_event.is_set():
                        return
                    f.write(line + "\n")
                    f.flush()
                    await asyncio.sleep(1)

            marker.touch()
            update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.COMPLETED)

        # Create fake output artifacts for preview testing
        output_dir = task_dir / "output"
        output_dir.mkdir(exist_ok=True)
        (output_dir / "result.ply").write_bytes(b"ply\nmock gaussian splatting data")
        (output_dir / "result.spz").write_bytes(b"mock spz data")

    def _collect_artifacts(self, output_dir: Path, scene_dir: Path) -> list[dict]:
        """Scan output and scene directories for artifacts."""
        artifacts = []

        for search_dir in (output_dir, scene_dir):
            if not search_dir.exists():
                continue
            for f in sorted(search_dir.rglob("*")):
                if not f.is_file() or f.name.startswith("."):
                    continue
                ext = f.suffix.lower()
                if ext in ARTIFACT_TYPES:
                    rel_path = str(f.relative_to(search_dir.parent))
                    artifacts.append({
                        "name": f.name,
                        "path": rel_path,
                        "size": f.stat().st_size,
                        "type": ARTIFACT_TYPES[ext],
                    })

        return artifacts
