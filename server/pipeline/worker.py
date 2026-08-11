"""Async pipeline worker - serial task execution with checkpoint resume."""
from __future__ import annotations

from typing import Optional

import asyncio
import json
import logging
import shutil
import time
import urllib.request
from pathlib import Path

from ..config import Config
from ..db import get_task, list_tasks, set_task_stages, update_stage, update_task
from ..models import StageStatus, TaskStatus
from .stage_runner import run_stage
from .quality import (
    QualityGateError,
    complete_high_quality_video_recovery,
    enforce_alignment_quality,
    postvalidate_worldgen_stage,
    prepare_high_quality_video_recovery,
)
from .recovery import (
    AutoRecoveryError,
    active_scene_dir,
    begin_source_regeneration,
    complete_auto_recovery_batch,
    create_conservative_rescue_trajectories,
    desired_rescue_candidate_count,
    prepare_auto_recovery,
    promote_recovered_scene,
    record_recovery_render_complete,
    record_rescue_skipped,
    record_transient_retry,
    restore_best_attempt_reports,
    schedule_auto_recovery_batch,
    set_active_scene,
    write_auto_video_trajectory_candidates,
)
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

# Stages that need all five GPUs and the CPU memory used by model loading.
# Trajectory generation/rendering still need the resident vLLM service, while
# WorldStereo and all later stages do not.
VLLM_FREE_STAGES = frozenset({"panogen", "video_gen"})


class PipelineWorker:
    def __init__(self, config: Config):
        self.config = config
        self._running = False
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._current_task_id: Optional[str] = None
        self._vllm_stopped = False

    # ------------------------------------------------------------------
    # vLLM lifecycle management (via supervisord)
    #
    # The 80B panogen stage shards weights across all 5 GPUs
    # (device_map="auto", ~34GB/GPU weights + activations), which does not
    # fit alongside the resident vLLM process on GPU 0 (~45GB). We therefore
    # stop vLLM before panogen and restart it afterwards (stages 1/2 need it
    # for trajectory planning / captioning).
    # ------------------------------------------------------------------
    async def _supervisorctl(self, *args: str) -> tuple[bool, str]:
        """Run supervisorctl; returns (success, combined output)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "supervisorctl", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
            return proc.returncode == 0, out.decode(errors="replace").strip()
        except Exception as e:
            logger.warning("supervisorctl %s failed: %s", args, e)
            return False, str(e)

    async def _vllm_healthy(self) -> bool:
        url = f"http://{self.config.VLLM_HOST}:{self.config.VLLM_PORT}/health"
        try:
            resp = await asyncio.to_thread(urllib.request.urlopen, url, timeout=3)
            ok = getattr(resp, "status", 200) == 200
            resp.close()
            return ok
        except Exception:
            return False

    async def _stop_vllm(self) -> None:
        """Stop vLLM to free GPU 0 VRAM before a full-5-GPU stage (panogen)."""
        if not self.config.MANAGE_VLLM or self._vllm_stopped:
            return
        ok, out = await self._supervisorctl("stop", "vllm")
        logger.info("supervisorctl stop vllm: ok=%s out=%s", ok, out)
        self._vllm_stopped = True
        await asyncio.sleep(10)  # allow CUDA context teardown / VRAM release

    async def _start_vllm(self) -> None:
        """Restart vLLM and wait until its health endpoint responds."""
        if not self.config.MANAGE_VLLM or not self._vllm_stopped:
            return
        ok, out = await self._supervisorctl("start", "vllm")
        logger.info("supervisorctl start vllm: ok=%s out=%s", ok, out)
        self._vllm_stopped = False
        if not ok:
            return
        # Model load can take a few minutes.
        for i in range(180):  # up to ~15 min
            if await self._vllm_healthy():
                logger.info("vLLM healthy again after restart (%ss)", i * 5)
                return
            await asyncio.sleep(5)
        logger.warning("vLLM did not become healthy within timeout after restart")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _recover_interrupted_tasks(self) -> None:
        """Return tasks interrupted by a container restart to the resume queue."""
        tasks, _ = list_tasks(self.config.DB_PATH, status=TaskStatus.RUNNING.value, limit=100)
        for task in tasks:
            logger.warning("Recovering interrupted task %s", task["id"])
            update_task(
                self.config.DB_PATH,
                task["id"],
                status=TaskStatus.PENDING.value,
                error=None,
            )

    async def start(self):
        """Main worker loop: poll for pending tasks and execute serially."""
        self._running = True
        logger.info("Pipeline worker started (mock=%s)", self.config.MOCK_PIPELINE)
        self._recover_interrupted_tasks()

        while self._running:
            try:
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

    @staticmethod
    def _is_transient_stage_error(error: str) -> bool:
        lowered = str(error).lower()
        permanent = (
            "no such file or directory",
            "module not found",
            "modulenotfounderror",
            "gated repo",
            "cannot access gated",
            "missing model",
            "invalid configuration",
            "unrecognized arguments",
        )
        if any(pattern in lowered for pattern in permanent):
            return False
        transient = (
            "timeout",
            "timed out",
            "connection reset",
            "connection refused",
            "temporary failure",
            "temporarily unavailable",
            "nccl",
            "network is unreachable",
            "broken pipe",
            "remote disconnected",
        )
        return any(pattern in lowered for pattern in transient)

    async def _run_stage_resilient(self, *, task_dir: Path, **kwargs) -> tuple[bool, str]:
        """Retry transient stage infrastructure errors at 30/120 seconds."""
        stage_name = str(kwargs.get("stage_name", "unknown"))
        backoffs = (30, 120)
        for attempt in range(len(backoffs) + 1):
            success, error = await run_stage(**kwargs)
            if success or not self._is_transient_stage_error(error) or attempt >= len(backoffs):
                return success, error
            try:
                record_transient_retry(
                    task_dir,
                    stage=stage_name,
                    error=error,
                    attempt=attempt + 1,
                )
            except AutoRecoveryError:
                # Non-recovery tasks still get the same retry policy; they do
                # not have an auto_recovery.json journal to update.
                pass
            log_file = kwargs.get("log_file")
            if isinstance(log_file, Path):
                with open(log_file, "a") as log:
                    log.write(
                        f"[Infrastructure Retry] {stage_name} transient failure; "
                        f"retry {attempt + 1}/2 in {backoffs[attempt]} seconds: {error}\n"
                    )
            await asyncio.sleep(backoffs[attempt])
        return False, "Transient retry loop exhausted"

    @staticmethod
    def _missing_recovery_render_assets(
        scene_dir: Path,
        candidate_ids: list[str],
    ) -> list[str]:
        missing: list[str] = []
        render_root = scene_dir / "render_results"
        for candidate_id in candidate_ids:
            parts = Path(candidate_id).parts
            if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
                missing.append(f"{candidate_id}:invalid_id")
                continue
            trajectory_dir = render_root.joinpath(*parts)
            for name in ("render.mp4", "render_mask.mp4", "traj_caption.json"):
                path = trajectory_dir / name
                if not path.is_file() or path.stat().st_size <= 0:
                    missing.append(f"{candidate_id}:{name}")
        return missing

    async def _render_recovery_batch(
        self,
        *,
        task_dir: Path,
        scene_dir: Path,
        candidate_ids: list[str],
        render_spec: tuple[list[str], dict[str, str], Path],
        render_log: Path,
        cancel_event: asyncio.Event,
    ) -> tuple[bool, str]:
        """Render/caption a rescue batch and verify all paid outputs exist."""
        render_cmd, render_env, render_cwd = render_spec
        command = [
            *render_cmd,
            "--trajectory_whitelist", str(task_dir / "video_trajectory_whitelist.json"),
            "--skip_exist",
        ]
        asset_backoffs = (30, 120)
        for attempt in range(len(asset_backoffs) + 1):
            rendered, render_error = await self._run_stage_resilient(
                task_dir=task_dir,
                stage_name="traj_render",
                cmd=command,
                env=render_env,
                log_file=render_log,
                cwd=render_cwd,
                cancel_event=cancel_event,
            )
            if not rendered:
                return False, render_error
            missing = self._missing_recovery_render_assets(scene_dir, candidate_ids)
            if not missing:
                record_recovery_render_complete(task_dir, candidate_ids=candidate_ids)
                return True, ""
            error = "Trajectory rendering completed without required rescue assets: " + ", ".join(missing)
            if attempt >= len(asset_backoffs):
                return False, error
            record_transient_retry(
                task_dir,
                stage="traj_render_assets",
                error=error,
                attempt=attempt + 1,
            )
            with open(render_log, "a") as log:
                log.write(
                    f"[Infrastructure Retry] rescue render asset validation failed; "
                    f"retry {attempt + 1}/2 in {asset_backoffs[attempt]} seconds: {error}\n"
                )
            await asyncio.sleep(asset_backoffs[attempt])
        return False, "Recovery render asset retry loop exhausted"

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

        # A source-regeneration attempt may survive a pod restart before it is
        # promoted.  Resume its isolated scene rather than accidentally
        # returning to the rejected source in the canonical scene directory.
        scene_dir = active_scene_dir(task_dir, scene_dir)

        task_type = task["type"]
        task_config = task.get("config", {})
        task_config.setdefault("hf_home", self.config.HF_HOME)
        task_config.setdefault("num_gpus", self.config.NUM_GPUS)
        task_config.setdefault("panogen_python", self.config.PANOGEN_PYTHON)
        task_config.setdefault("quality_preset", "high")
        task_config.setdefault(
            "recovery_policy",
            "auto" if task_config["quality_preset"] == "high" else "off",
        )

        stage_names = get_stage_names(task_type)
        set_task_stages(self.config.DB_PATH, task_id, stage_names)
        update_task(self.config.DB_PATH, task_id, status=TaskStatus.RUNNING.value)

        logger.info("Executing task %s (%s): %d stages", task_id, task_type, len(stage_names))

        try:
            if self.config.MOCK_PIPELINE:
                await self._execute_mock(task_id, stage_names, cancel_event, task_dir, logs_dir)
            else:
                await self._execute_real(task_id, task_type, task_config, task_dir, scene_dir, input_dir, logs_dir, cancel_event)

            if cancel_event.is_set():
                update_task(self.config.DB_PATH, task_id, status=TaskStatus.CANCELLED.value)
                return

            artifacts = self._collect_artifacts(output_dir, scene_dir, task_dir)
            update_task(
                self.config.DB_PATH, task_id,
                status=TaskStatus.COMPLETED.value,
                progress=1.0,
                artifacts=artifacts,
            )
            logger.info("Task %s completed", task_id)

        except Exception as e:
            logger.exception("Task %s failed", task_id)
            # Surface structured quality evidence even for a deliberately
            # rejected task, so the user can diagnose why it did not proceed.
            artifacts = self._collect_artifacts(output_dir, scene_dir, task_dir)
            update_task(
                self.config.DB_PATH, task_id,
                status=TaskStatus.FAILED.value,
                error=str(e),
                artifacts=artifacts,
            )
        finally:
            # Make sure vLLM is restored even if the task failed mid-way.
            try:
                await self._start_vllm()
            except Exception:
                logger.exception("Failed to restore vLLM")
            self._cancel_events.pop(task_id, None)
            self._current_task_id = None

    async def _execute_real(
        self, task_id, task_type, task_config, task_dir, scene_dir, input_dir, logs_dir, cancel_event
    ):
        """Execute real pipeline stages."""
        regenerated_source_stages: set[str] = set()
        recovery_path = task_dir / "auto_recovery.json"
        recovery_state = {}
        if recovery_path.is_file():
            try:
                recovery_state = json.loads(recovery_path.read_text())
            except json.JSONDecodeError:
                recovery_state = {}
        if task_type == "worldgen" and recovery_state.get("state") == "regenerating_panorama":
            seed = int(recovery_state.get("source_seed", task_config.get("seed", 1024)))
            success, error = await self._run_regenerated_source_attempt(
                task_dir=task_dir,
                source_scene_dir=self.config.task_scene_dir(task_id),
                input_dir=input_dir,
                task_config=task_config,
                seed=seed,
                cancel_event=cancel_event,
                resume=True,
            )
            if not success:
                raise RuntimeError(error)
            # The regenerated source already completed these four stages in
            # its isolated attempt directory.  Mark them as durable root
            # checkpoints so the resumed task proceeds directly to GS data
            # preparation and training instead of returning success early.
            regenerated_source_stages = {"panogen", "traj_generate", "traj_render", "video_gen"}
            for stage_name in regenerated_source_stages:
                (task_dir / f".stage_{stage_name}_done").touch()
            scene_dir = self.config.task_scene_dir(task_id)
        elif task_type == "worldgen_from_pano" and recovery_state.get("state") == "replacing_panorama_source":
            success, error = await self._run_replacement_panorama_attempt(
                task_dir=task_dir,
                source_scene_dir=self.config.task_scene_dir(task_id),
                input_dir=input_dir,
                task_config=task_config,
                cancel_event=cancel_event,
                resume=True,
            )
            if not success:
                raise RuntimeError(error)
            regenerated_source_stages = {"traj_generate", "traj_render", "video_gen"}
            for stage_name in regenerated_source_stages:
                (task_dir / f".stage_{stage_name}_done").touch()
            scene_dir = self.config.task_scene_dir(task_id)
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

        render_spec = next(
            ((command, command_env, command_cwd) for name, command, command_env, command_cwd in commands if name == "traj_render"),
            None,
        )

        for stage_name, cmd, env, cwd in commands:
            if cancel_event.is_set():
                return

            # Content-quality exhaustion is a completed panorama delivery, not
            # a failed pipeline. Keep both downstream stages visibly skipped so
            # the API never advertises a fabricated or empty 3D result.
            if stage_name in {"gen_gs_data", "gs_training"}:
                try:
                    recovery_delivery = json.loads(recovery_path.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    recovery_delivery = {}
                if recovery_delivery.get("delivery_tier") == "panorama_only":
                    update_stage(
                        self.config.DB_PATH,
                        task_id,
                        stage_name,
                        StageStatus.SKIPPED,
                        error=str(
                            recovery_delivery.get("quality_warning")
                            or "3D stage skipped because fewer than four trajectories passed strict quality gates."
                        ),
                    )
                    continue

            # Checkpoint resume: skip completed stages
            marker = task_dir / f".stage_{stage_name}_done"
            if marker.exists():
                logger.info("Stage %s already completed, skipping", stage_name)
                if stage_name in regenerated_source_stages:
                    # The stage did not run in this root log file, but it
                    # succeeded in the promoted isolated source attempt.
                    update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.COMPLETED)
                    continue
                # Re-run output validation on resume.  This regenerates
                # quality manifests introduced by a later release and never
                # lets an old silent success bypass the current policy.
                if task_type in ("worldgen", "worldgen_from_pano"):
                    try:
                        postvalidate_worldgen_stage(
                            stage_name,
                            task_dir=task_dir,
                            scene_dir=scene_dir,
                            log_file=logs_dir / f"{stage_name}.log",
                            quality_preset=task_config["quality_preset"],
                        )
                    except QualityGateError as exc:
                        update_stage(
                            self.config.DB_PATH,
                            task_id,
                            stage_name,
                            StageStatus.FAILED,
                            error=str(exc),
                        )
                        raise RuntimeError(
                            f"Stage {stage_name} failed quality validation on resume: {exc}"
                        ) from exc
                update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.SKIPPED)
                continue

            # Full-model stages need vLLM's GPU and host memory released first.
            if stage_name in VLLM_FREE_STAGES:
                await self._stop_vllm()

            log_file = logs_dir / f"{stage_name}.log"
            update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.RUNNING)

            # A full WorldStereo pass may have enough technically valid
            # planned routes but too few alignment-approved ones.  Refill from
            # untested routes in bounded batches before rejecting the task;
            # this never relaxes the quality thresholds or feeds rejected
            # videos into later 3DGS stages.
            if stage_name == "video_gen" and task_config["quality_preset"] == "high":
                try:
                    if task_config.get("recovery_policy") == "auto":
                        success, error = await self._run_auto_high_quality_video_recovery(
                            task_dir=task_dir,
                            scene_dir=scene_dir,
                            log_file=log_file,
                            cmd=cmd,
                            env=env,
                            cwd=cwd,
                            cancel_event=cancel_event,
                            task_type=task_type,
                            recovery_policy="auto",
                            render_spec=render_spec,
                            input_dir=input_dir,
                            task_config=task_config,
                        )
                    else:
                        success, error = await self._run_high_quality_video_recovery(
                            task_dir=task_dir,
                            scene_dir=scene_dir,
                            log_file=log_file,
                            cmd=cmd,
                            env=env,
                            cwd=cwd,
                            cancel_event=cancel_event,
                        )
                except QualityGateError as exc:
                    update_stage(
                        self.config.DB_PATH,
                        task_id,
                        stage_name,
                        StageStatus.FAILED,
                        error=str(exc),
                    )
                    raise RuntimeError(f"Stage {stage_name} failed quality validation: {exc}") from exc

                if success:
                    marker.touch()
                    update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.COMPLETED)
                    continue
                if cancel_event.is_set():
                    return
                update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.FAILED, error=error)
                raise RuntimeError(f"Stage {stage_name} failed: {error}")

            success, error = await self._run_stage_resilient(
                task_dir=task_dir,
                stage_name=stage_name,
                cmd=cmd,
                env=env,
                log_file=log_file,
                cwd=cwd,
                cancel_event=cancel_event,
            )

            if success:
                try:
                    if task_type in ("worldgen", "worldgen_from_pano"):
                        postvalidate_worldgen_stage(
                            stage_name,
                            task_dir=task_dir,
                            scene_dir=scene_dir,
                            log_file=log_file,
                            quality_preset=task_config["quality_preset"],
                        )
                except QualityGateError as exc:
                    update_stage(
                        self.config.DB_PATH,
                        task_id,
                        stage_name,
                        StageStatus.FAILED,
                        error=str(exc),
                    )
                    raise RuntimeError(f"Stage {stage_name} failed quality validation: {exc}") from exc
                marker.touch()
                update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.COMPLETED)
            else:
                if cancel_event.is_set():
                    # execute_task owns the terminal cancelled status; do not
                    # overwrite it with a synthetic stage failure.
                    return
                update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.FAILED, error=error)
                # Even on failure, restore vLLM so later tasks can use it.
                if stage_name == "panogen":
                    await self._start_vllm()
                raise RuntimeError(f"Stage {stage_name} failed: {error}")

            # Restore vLLM once panogen (the only full-VRAM stage) finishes.
            if stage_name == "panogen":
                await self._start_vllm()

    async def _run_high_quality_video_recovery(
        self,
        *,
        task_dir: Path,
        scene_dir: Path,
        log_file: Path,
        cmd: list[str],
        env: dict[str, str],
        cwd: Path,
        cancel_event: asyncio.Event,
    ) -> tuple[bool, str]:
        """Run/replenish high-quality WorldStereo candidates until the gate passes.

        The manifest is rewritten for each new batch, while ``run_stage``
        appends to one durable log.  This lets alignment validation assess the
        union of prior and supplemental trajectories and makes retry resume
        from the next untested routes after a gate-only failure.
        """
        while not cancel_event.is_set():
            recovery = prepare_high_quality_video_recovery(task_dir)
            candidate_ids = recovery["next_candidate_trajectory_ids"]
            expected_ids = [
                *recovery["attempted_trajectory_ids"],
                *candidate_ids,
            ]
            if not candidate_ids:
                alignment = postvalidate_worldgen_stage(
                    "video_gen",
                    task_dir=task_dir,
                    scene_dir=scene_dir,
                    log_file=log_file,
                    quality_preset="high",
                    expected_trajectory_ids=expected_ids,
                    enforce_alignment=False,
                )
                if not isinstance(alignment, dict):
                    raise QualityGateError("Video alignment report was not produced")
                enforce_alignment_quality(
                    alignment,
                    quality_preset="high",
                    candidate_budget=recovery["candidate_budget"],
                )
                return True, ""

            round_number = len(recovery["batches"]) + 1
            with open(log_file, "a") as log:
                log.write(
                    f"[Quality Recovery] round {round_number}: generating "
                    f"{len(candidate_ids)} untested trajectories "
                    f"({len(recovery['accepted_trajectory_ids'])}/"
                    f"{recovery['minimum_accepted_trajectories']} accepted so far)\n"
                )

            success, error = await self._run_stage_resilient(
                task_dir=task_dir,
                stage_name="video_gen",
                cmd=cmd,
                env=env,
                log_file=log_file,
                cwd=cwd,
                cancel_event=cancel_event,
            )
            if not success:
                return False, error

            alignment = postvalidate_worldgen_stage(
                "video_gen",
                task_dir=task_dir,
                scene_dir=scene_dir,
                log_file=log_file,
                quality_preset="high",
                expected_trajectory_ids=expected_ids,
                enforce_alignment=False,
            )
            if not isinstance(alignment, dict):
                raise QualityGateError("Video alignment report was not produced")
            completed = complete_high_quality_video_recovery(task_dir, alignment)
            if completed["status"] == "accepted":
                enforce_alignment_quality(alignment, quality_preset="high")
                return True, ""

        return False, "Cancelled by user"

    async def _run_auto_high_quality_video_recovery(
        self,
        *,
        task_dir: Path,
        scene_dir: Path,
        log_file: Path,
        cmd: list[str],
        env: dict[str, str],
        cwd: Path,
        cancel_event: asyncio.Event,
        task_type: str,
        recovery_policy: str,
        render_spec: tuple[list[str], dict[str, str], Path] | None,
        input_dir: Path | None = None,
        task_config: dict | None = None,
    ) -> tuple[bool, str]:
        """Run the persistent recovery state machine around WorldStereo.

        Normal batches reuse the standard video invocation.  Rescue batches
        first render only their new safe camera paths, then use the identical
        alignment gate and downstream whitelist as normal trajectories.
        """
        try:
            quality_report = json.loads((task_dir / "quality_report.json").read_text())
            planning = quality_report["planning"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise QualityGateError("Automatic recovery requires a planning quality report") from exc
        if not isinstance(planning, dict):
            raise QualityGateError("Automatic recovery planning report is invalid")

        while not cancel_event.is_set():
            recovery = prepare_auto_recovery(
                task_dir,
                planning=planning,
                task_type=task_type,
                recovery_policy=recovery_policy,
            )
            state = recovery["state"]
            if state in {"accepted", "completed"}:
                return True, ""
            if state == "needs_rescue":
                round_number = recovery.get("rescue_round")
                try:
                    candidates = create_conservative_rescue_trajectories(
                        scene_dir,
                        round_number=int(round_number),
                        planning=planning,
                        target_count=desired_rescue_candidate_count(
                            len(recovery.get("accepted_trajectory_ids", []))
                        ),
                    )
                    if not candidates:
                        record_rescue_skipped(
                            task_dir,
                            round_number=int(round_number),
                            reason="No validated regular camera prefix is available; advancing recovery without failing the task.",
                        )
                        continue
                    recovery = schedule_auto_recovery_batch(
                        task_dir,
                        phase=f"rescue_{round_number}",
                        candidate_ids=[candidate["id"] for candidate in candidates],
                    )
                    write_auto_video_trajectory_candidates(
                        task_dir, recovery["pending_candidate_trajectory_ids"]
                    )
                except (AutoRecoveryError, TypeError, ValueError) as exc:
                    return False, str(exc)

                if render_spec is None:
                    return False, "Automatic recovery cannot render rescue trajectories without a render command"
                render_log = log_file.parent / f"traj_render_recovery_r{round_number}.log"
                await self._start_vllm()
                rendered, render_error = await self._render_recovery_batch(
                    task_dir=task_dir,
                    scene_dir=scene_dir,
                    candidate_ids=recovery["pending_candidate_trajectory_ids"],
                    render_spec=render_spec,
                    render_log=render_log,
                    cancel_event=cancel_event,
                )
                if not rendered:
                    return False, render_error
                await self._stop_vllm()
                continue
            if state == "needs_regeneration":
                if task_type != "worldgen" or input_dir is None or task_config is None:
                    return False, "Automatic recovery requires a regenerated panorama attempt"
                base_seed = int(task_config.get("seed", 1024))
                seed = base_seed + int(recovery.get("source_attempt", 0)) + 1
                return await self._run_regenerated_source_attempt(
                    task_dir=task_dir,
                    source_scene_dir=scene_dir,
                    input_dir=input_dir,
                    task_config=task_config,
                    seed=seed,
                    cancel_event=cancel_event,
                    resume=False,
                )
            if state in {"source_quality_blocked", "quality_blocked", "budget_exhausted"}:
                # v1 journals can reach these states before migration. Prepare
                # once more so v2 converts the content outcome to a delivery.
                continue
            if state == "disabled":
                return False, "Automatic recovery is disabled"

            candidate_ids = recovery.get("pending_candidate_trajectory_ids", [])
            if not candidate_ids:
                return False, f"Automatic recovery entered unexpected state: {state}"
            write_auto_video_trajectory_candidates(task_dir, candidate_ids)

            # A prior render may have stopped after writing MP4s but before
            # captioning or before its durable checkpoint. Resume the rescue
            # render with --skip_exist instead of sending incomplete inputs to
            # WorldStereo and repeating the same missing-caption failure.
            rendered_ids = set(recovery.get("rendered_candidate_trajectory_ids", []))
            pending_rescue_ids = [
                candidate_id
                for candidate_id in candidate_ids
                if candidate_id.startswith("view_recovery_")
            ]
            if pending_rescue_ids and not set(pending_rescue_ids).issubset(rendered_ids):
                if render_spec is None:
                    return False, "Pending rescue trajectories require a render command before WorldStereo"
                round_number = recovery.get("rescue_round") or 1
                resume_render_log = log_file.parent / f"traj_render_recovery_r{round_number}.log"
                await self._start_vllm()
                rendered, render_error = await self._render_recovery_batch(
                    task_dir=task_dir,
                    scene_dir=scene_dir,
                    candidate_ids=pending_rescue_ids,
                    render_spec=render_spec,
                    render_log=resume_render_log,
                    cancel_event=cancel_event,
                )
                if not rendered:
                    return False, render_error
                await self._stop_vllm()
                recovery = prepare_auto_recovery(
                    task_dir,
                    planning=planning,
                    task_type=task_type,
                    recovery_policy=recovery_policy,
                )
            expected_ids = [*recovery.get("attempted_trajectory_ids", []), *candidate_ids]
            phase = recovery.get("phase") or "unknown"
            with open(log_file, "a") as log:
                log.write(
                    f"[Auto Recovery] {phase}: generating {len(candidate_ids)} trajectories "
                    f"({len(recovery.get('accepted_trajectory_ids', []))}/"
                    f"{recovery.get('minimum_accepted_trajectories', 6)} accepted so far)\n"
                )
            started = time.monotonic()
            success, error = await self._run_stage_resilient(
                task_dir=task_dir,
                stage_name="video_gen",
                cmd=cmd,
                env=env,
                log_file=log_file,
                cwd=cwd,
                cancel_event=cancel_event,
            )
            if not success:
                return False, error
            alignment = postvalidate_worldgen_stage(
                "video_gen",
                task_dir=task_dir,
                scene_dir=scene_dir,
                log_file=log_file,
                quality_preset="high",
                expected_trajectory_ids=expected_ids,
                enforce_alignment=False,
            )
            if not isinstance(alignment, dict):
                raise QualityGateError("Video alignment report was not produced")
            complete_auto_recovery_batch(
                task_dir,
                alignment=alignment,
                duration_seconds=time.monotonic() - started,
            )

        return False, "Cancelled by user"

    async def _run_regenerated_source_attempt(
        self,
        *,
        task_dir: Path,
        source_scene_dir: Path,
        input_dir: Path,
        task_config: dict,
        seed: int,
        cancel_event: asyncio.Event,
        resume: bool,
    ) -> tuple[bool, str]:
        """Run one source-isolated panorama attempt and promote only on success."""
        recovery_root = task_dir / "recovery_attempts"
        recovery_root.mkdir(parents=True, exist_ok=True)
        report = json.loads((task_dir / "auto_recovery.json").read_text())
        source_attempt = int(report.get("source_attempt", 0))
        if not resume:
            source_attempt += 1
        attempt_dir = recovery_root / f"attempt-{source_attempt:03d}"
        attempt_scene_dir = attempt_dir / "scene"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_scene_dir.mkdir(parents=True, exist_ok=True)
        scene_relpath = str(attempt_scene_dir.relative_to(task_dir))
        if not resume:
            previous_attempt = source_attempt - 1
            archive_reports = recovery_root / f"attempt-{previous_attempt:03d}" / "reports"
            archive_reports.mkdir(parents=True, exist_ok=True)
            for name in (
                "quality_report.json",
                "alignment_quality.json",
                "trajectory_whitelist.json",
                "video_trajectory_whitelist.json",
                "auto_recovery.json",
            ):
                report_path = task_dir / name
                archive_path = archive_reports / name
                if report_path.is_file() and not archive_path.exists():
                    shutil.copy2(report_path, archive_path)
            begin_source_regeneration(task_dir, seed=seed, scene_relpath=scene_relpath)

        attempt_config = dict(task_config)
        attempt_config["seed"] = seed
        commands = get_worldgen_commands(
            task_dir=task_dir,
            scene_dir=attempt_scene_dir,
            input_dir=input_dir,
            config=attempt_config,
            hyworld_repo=self.config.HYWORLD_REPO,
            vllm_host=self.config.VLLM_HOST,
            vllm_port=self.config.VLLM_PORT,
            vllm_model=self.config.VLLM_MODEL,
            task_type="worldgen",
        )
        by_stage = {name: (command, env, cwd) for name, command, env, cwd in commands}
        for stage_name in ("panogen", "traj_generate", "traj_render"):
            marker = attempt_dir / f".stage_{stage_name}_done"
            if marker.exists():
                continue
            if cancel_event.is_set():
                return False, "Cancelled by user"
            command, env, cwd = by_stage[stage_name]
            if stage_name == "panogen":
                await self._stop_vllm()
            elif stage_name == "traj_generate":
                await self._start_vllm()
            log_file = task_dir / "logs" / f"recovery_attempt_{source_attempt:03d}_{stage_name}.log"
            success, error = await self._run_stage_resilient(
                task_dir=task_dir,
                stage_name=stage_name,
                cmd=command,
                env=env,
                log_file=log_file,
                cwd=cwd,
                cancel_event=cancel_event,
            )
            if not success:
                return False, error
            if stage_name == "traj_generate":
                try:
                    postvalidate_worldgen_stage(
                        stage_name,
                        task_dir=task_dir,
                        scene_dir=attempt_scene_dir,
                        log_file=log_file,
                        quality_preset="high",
                    )
                except QualityGateError as exc:
                    return False, str(exc)
            marker.touch()

        await self._stop_vllm()
        video_cmd, video_env, video_cwd = by_stage["video_gen"]
        render_cmd, render_env, render_cwd = by_stage["traj_render"]
        success, error = await self._run_auto_high_quality_video_recovery(
            task_dir=task_dir,
            scene_dir=attempt_scene_dir,
            log_file=task_dir / "logs" / f"recovery_attempt_{source_attempt:03d}_video_gen.log",
            cmd=video_cmd,
            env=video_env,
            cwd=video_cwd,
            cancel_event=cancel_event,
            task_type="worldgen",
            recovery_policy="auto",
            render_spec=(render_cmd, render_env, render_cwd),
            input_dir=input_dir,
            task_config=attempt_config,
        )
        if not success:
            return False, error

        try:
            completed_report = json.loads((task_dir / "auto_recovery.json").read_text())
            best_attempt = int(completed_report.get("best_source_attempt", source_attempt))
            if best_attempt == source_attempt:
                promote_recovered_scene(
                    task_dir,
                    source_attempt=source_attempt,
                    regenerated_scene_dir=attempt_scene_dir,
                    canonical_scene_dir=source_scene_dir,
                )
                set_active_scene(task_dir, "scene")
            else:
                restore_best_attempt_reports(task_dir, source_attempt=best_attempt)
                if best_attempt == 0:
                    set_active_scene(task_dir, "scene")
        except (AutoRecoveryError, FileNotFoundError, json.JSONDecodeError) as exc:
            return False, str(exc)
        return True, ""

    async def _run_replacement_panorama_attempt(
        self,
        *,
        task_dir: Path,
        source_scene_dir: Path,
        input_dir: Path,
        task_config: dict,
        cancel_event: asyncio.Event,
        resume: bool,
    ) -> tuple[bool, str]:
        """Run a user-supplied panorama retry without ever invoking panogen."""
        try:
            report = json.loads((task_dir / "auto_recovery.json").read_text())
            source_attempt = int(report["source_attempt"])
            scene_relpath = str(report["active_scene_relpath"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return False, f"Replacement panorama recovery report is invalid: {exc}"
        if not resume or report.get("state") != "replacing_panorama_source":
            return False, "Replacement panorama retry was not requested by the recovery state machine"
        attempt_scene_dir = task_dir / scene_relpath
        if not (attempt_scene_dir / "panorama.png").is_file():
            return False, "Replacement panorama is missing; upload a source image before retrying"

        commands = get_worldgen_commands(
            task_dir=task_dir,
            scene_dir=attempt_scene_dir,
            input_dir=input_dir,
            config=task_config,
            hyworld_repo=self.config.HYWORLD_REPO,
            vllm_host=self.config.VLLM_HOST,
            vllm_port=self.config.VLLM_PORT,
            vllm_model=self.config.VLLM_MODEL,
            task_type="worldgen_from_pano",
        )
        by_stage = {name: (command, env, cwd) for name, command, env, cwd in commands}
        for stage_name in ("traj_generate", "traj_render"):
            marker = attempt_scene_dir.parent / f".stage_{stage_name}_done"
            if marker.exists():
                continue
            if cancel_event.is_set():
                return False, "Cancelled by user"
            command, env, cwd = by_stage[stage_name]
            if stage_name == "traj_generate":
                await self._start_vllm()
            log_file = task_dir / "logs" / f"replacement_attempt_{source_attempt:03d}_{stage_name}.log"
            success, error = await self._run_stage_resilient(
                task_dir=task_dir,
                stage_name=stage_name,
                cmd=command,
                env=env,
                log_file=log_file,
                cwd=cwd,
                cancel_event=cancel_event,
            )
            if not success:
                return False, error
            try:
                postvalidate_worldgen_stage(
                    stage_name,
                    task_dir=task_dir,
                    scene_dir=attempt_scene_dir,
                    log_file=log_file,
                    quality_preset="high",
                )
            except QualityGateError as exc:
                return False, str(exc)
            marker.touch()

        await self._stop_vllm()
        video_cmd, video_env, video_cwd = by_stage["video_gen"]
        render_cmd, render_env, render_cwd = by_stage["traj_render"]
        success, error = await self._run_auto_high_quality_video_recovery(
            task_dir=task_dir,
            scene_dir=attempt_scene_dir,
            log_file=task_dir / "logs" / f"replacement_attempt_{source_attempt:03d}_video_gen.log",
            cmd=video_cmd,
            env=video_env,
            cwd=video_cwd,
            cancel_event=cancel_event,
            task_type="worldgen_from_pano",
            recovery_policy="auto",
            render_spec=(render_cmd, render_env, render_cwd),
            input_dir=input_dir,
            task_config=task_config,
        )
        if not success:
            return False, error
        try:
            promote_recovered_scene(
                task_dir,
                source_attempt=source_attempt,
                regenerated_scene_dir=attempt_scene_dir,
                canonical_scene_dir=source_scene_dir,
            )
            set_active_scene(task_dir, "scene")
        except AutoRecoveryError as exc:
            return False, str(exc)
        return True, ""

    async def _execute_mock(self, task_id, stage_names, cancel_event, task_dir, logs_dir):
        """Mock pipeline for local development."""
        for stage_name in stage_names:
            if cancel_event.is_set():
                return

            marker = task_dir / f".stage_{stage_name}_done"
            if marker.exists():
                update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.SKIPPED)
                continue

            log_file = task_dir / "logs" / f"{stage_name}.log"
            update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.RUNNING)

            fake_logs = [
                f"[{stage_name}] Starting...",
                f"[{stage_name}] Loading models...",
                f"[{stage_name}] Progress: 25%",
                f"[{stage_name}] Progress: 50%",
                f"[{stage_name}] Progress: 75%",
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

        output_dir = task_dir / "output"
        output_dir.mkdir(exist_ok=True)
        (output_dir / "result.ply").write_bytes(b"ply\nmock gaussian splatting data")
        (output_dir / "result.spz").write_bytes(b"mock spz data")

    def _collect_artifacts(self, output_dir: Path, scene_dir: Path, task_dir: Path | None = None) -> list[dict]:
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
        # Quality reports live at task root so they remain available even when
        # a gate rejects a later stage.  Include them without recursively
        # rescanning output/scene and duplicating every normal artifact.
        if task_dir:
            for name in (
                "quality_report.json",
                "alignment_quality.json",
                "trajectory_whitelist.json",
                "video_trajectory_whitelist.json",
                "video_recovery.json",
                "auto_recovery.json",
                "source_promotion.json",
            ):
                report = task_dir / name
                if report.is_file():
                    artifacts.append({
                        "name": report.name,
                        "path": report.name,
                        "size": report.stat().st_size,
                        "type": "data",
                    })
        return artifacts
