"""Async pipeline worker - serial task execution with checkpoint resume."""
from __future__ import annotations

from typing import Optional

import asyncio
import json
import logging
import os
import signal
import shutil
import time
import urllib.request
from pathlib import Path

from ..config import Config
from ..db import get_task, list_tasks, set_task_stages, update_stage, update_task
from ..delivery_manifest import write_delivery_manifest
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
    prepare_training_delivery,
    migrate_non_strict_delivery,
    promote_recovered_scene,
    record_recovery_render_complete,
    request_pilot_content_recovery,
    record_rescue_skipped,
    record_transient_retry,
    restore_best_attempt_reports,
    schedule_auto_recovery_batch,
    set_active_scene,
    write_auto_video_trajectory_candidates,
)
from .worldgen import get_stage_names, get_worldgen_commands
from .worldrecon import get_worldrecon_commands
from .preflight import (
    InputPreflightError,
    normalize_task_contract,
    select_ranked_source_for_attempt,
    strict_contract,
    write_json_atomic,
)
from .pilot import PilotValidationError, validate_pilot_output
from .navigation_stress import NavigationStressError
from .delivery import (
    DeliveryValidationError,
    build_training_input_manifest,
    prepare_safe_training_retry,
    validate_completed_delivery,
)

logger = logging.getLogger(__name__)


def _next_recovery_seed(task_config: dict, source_attempt: int) -> int:
    """Return base_seed + attempt without compounding nested attempt seeds."""
    base_seed = task_config.get("_recovery_base_seed", task_config.get("seed", 1024))
    return int(base_seed) + int(source_attempt) + 1


def _should_resume_isolated_source_attempt(recovery: dict) -> bool:
    """Resume every durable, non-terminal phase of an isolated source.

    v57 could be restarted immediately after persisting a WorldStereo batch,
    when the journal is no longer named ``regenerating_panorama``.  The active
    scene path is the authoritative signal that paid work still belongs to an
    isolated source attempt.
    """
    try:
        source_attempt = int(recovery.get("source_attempt", 0) or 0)
    except (TypeError, ValueError):
        return False
    scene_relpath = str(recovery.get("active_scene_relpath") or "")
    state = str(recovery.get("state") or "")
    return bool(
        source_attempt > 0
        and scene_relpath.startswith("recovery_attempts/attempt-")
        and state in {
            "regenerating_panorama",
            "generating_normal",
            "generating_rescue",
            "needs_rescue",
            "ready",
            "accepted",
            "completed",
        }
    )


def _promote_recovery_source_artifacts(
    *, attempt_dir: Path, attempt_input_dir: Path, task_dir: Path, input_dir: Path
) -> None:
    """Promote source provenance for both generated and uploaded retries."""
    source = attempt_input_dir / "input.png"
    if source.is_file():
        canonical_source = input_dir / "input.png"
        canonical_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, canonical_source)
    attempt_report = attempt_dir / "source_generation.json"
    if attempt_report.is_file():
        shutil.copy2(attempt_report, task_dir / "source_generation.json")


def _accepted_trajectory_ids(path: Path) -> list[str]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    accepted = value.get("accepted_trajectory_ids", []) if isinstance(value, dict) else []
    return sorted({str(item) for item in accepted if isinstance(item, str)})


def _archive_current_source_reports(
    *,
    task_dir: Path,
    attempt_dir: Path,
    attempt_scene_dir: Path,
    source_attempt: int,
    completed_report: dict,
) -> None:
    """Make the selected source evidence durable before another report restore.

    v57 kept the current source's alignment report only at the task root.  A
    v58 resume could therefore restore an older best attempt before promoting
    the newly completed source, leaving the journal at seven accepted paths
    while the root whitelist still described three.  Rebuild the current
    report from its immutable attempt log when necessary, verify it against
    the journal snapshot, and archive it before any best-attempt restore.
    """
    snapshot = next(
        (
            item for item in completed_report.get("source_attempts", [])
            if isinstance(item, dict) and int(item.get("source_attempt", -1)) == source_attempt
        ),
        None,
    )
    if not isinstance(snapshot, dict):
        raise AutoRecoveryError(f"Source attempt {source_attempt} is missing from the recovery journal")
    expected_accepted = sorted({
        str(item) for item in snapshot.get("accepted_trajectory_ids", [])
        if isinstance(item, str)
    })
    reports_dir = attempt_dir / "reports"
    archived_alignment = reports_dir / "alignment_quality.json"
    if _accepted_trajectory_ids(archived_alignment) == expected_accepted:
        for name in (
            "quality_report.json",
            "alignment_quality.json",
            "trajectory_whitelist.json",
            "video_trajectory_whitelist.json",
        ):
            source = reports_dir / name
            if source.is_file():
                shutil.copy2(source, task_dir / name)
        return

    root_alignment = task_dir / "alignment_quality.json"
    if _accepted_trajectory_ids(root_alignment) != expected_accepted:
        log_file = task_dir / "logs" / f"recovery_attempt_{source_attempt:03d}_video_gen.log"
        if not log_file.is_file():
            raise AutoRecoveryError(
                f"Source attempt {source_attempt} has no immutable alignment log to rebuild its reports"
            )
        rebuilt = postvalidate_worldgen_stage(
            "video_gen",
            task_dir=task_dir,
            scene_dir=attempt_scene_dir,
            log_file=log_file,
            quality_preset="high",
            expected_trajectory_ids=snapshot.get("attempted_trajectory_ids", []),
            enforce_alignment=False,
        )
        rebuilt_accepted = sorted({
            str(item) for item in (rebuilt or {}).get("accepted_trajectory_ids", [])
            if isinstance(item, str)
        })
        if rebuilt_accepted != expected_accepted:
            raise AutoRecoveryError(
                f"Source attempt {source_attempt} alignment evidence does not match its journal snapshot"
            )

    reports_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "quality_report.json",
        "alignment_quality.json",
        "trajectory_whitelist.json",
        "video_trajectory_whitelist.json",
        "auto_recovery.json",
    ):
        source = task_dir / name
        destination = reports_dir / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)


def _archive_incomplete_final_training(task_dir: Path) -> Path | None:
    """Move unvalidated formal-training output aside before a fresh run."""
    candidates = [
        task_dir / "output",
        task_dir / "artifact_validation.json",
        task_dir / "navigation_validation.json",
        task_dir / "navigation_stress.json",
    ]
    movable = [
        path for path in candidates
        if path.exists() and (not path.is_dir() or any(path.iterdir()))
    ]
    if not movable:
        return None
    recovery_root = task_dir / "recovery_attempts"
    recovery_root.mkdir(parents=True, exist_ok=True)
    attempt = len(list(recovery_root.glob("incomplete-final-training-*"))) + 1
    archive = recovery_root / f"incomplete-final-training-{attempt:03d}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in movable:
        os.replace(path, archive / path.name)
    write_json_atomic(archive / "archive_report.json", {
        "version": 4,
        "reason": "unvalidated_formal_training_must_restart_from_step_zero",
        "archived_names": [path.name for path in movable],
    })
    return archive

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
VLLM_FREE_STAGES = frozenset({
    "source_generate", "panogen", "video_gen", "gen_gs_data",
    "gs_pilot", "gs_training", "worldrecon", "navigation_stress",
})
VLLM_PROCESS_MARKERS = ("/opt/vllm/bin/python", "VLLM::EngineCore")
ISOLATED_SOURCE_COMPLETED_STAGES = frozenset({
    "source_generate", "input_preflight", "panogen", "post_panorama_preflight",
    "traj_generate", "traj_render", "video_gen",
})


def _vllm_process_ids(ps_output: str) -> list[int]:
    """Return only vLLM-owned PIDs from a process-table snapshot."""
    process_ids: list[int] = []
    for raw_line in ps_output.splitlines():
        fields = raw_line.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        if any(marker in fields[1] for marker in VLLM_PROCESS_MARKERS):
            process_ids.append(int(fields[0]))
    return process_ids


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

    async def _running_vllm_process_ids(self) -> list[int]:
        """Inspect the current Pod only; never target node-wide processes."""
        proc = await asyncio.create_subprocess_exec(
            "ps", "-eo", "pid=,args=",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode != 0:
            raise RuntimeError(f"Could not inspect vLLM processes: {out.decode(errors='replace').strip()}")
        return _vllm_process_ids(out.decode(errors="replace"))

    async def _terminate_residual_vllm_processes(self) -> None:
        """Remove orphaned vLLM children left behind by older supervisor configs."""
        process_ids = await self._running_vllm_process_ids()
        if not process_ids:
            return
        logger.warning("Stopping residual vLLM processes before GPU stage: %s", process_ids)
        for process_id in process_ids:
            try:
                os.kill(process_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _ in range(20):
            await asyncio.sleep(0.5)
            process_ids = await self._running_vllm_process_ids()
            if not process_ids:
                return
        for process_id in process_ids:
            try:
                os.kill(process_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await asyncio.sleep(1)
        process_ids = await self._running_vllm_process_ids()
        if process_ids:
            raise RuntimeError(f"vLLM GPU processes did not stop cleanly: {process_ids}")

    async def _stop_vllm(self) -> None:
        """Stop vLLM to free GPU 0 VRAM before a full-5-GPU stage (panogen)."""
        if not self.config.MANAGE_VLLM:
            return
        if not self._vllm_stopped:
            ok, out = await self._supervisorctl("stop", "vllm")
            logger.info("supervisorctl stop vllm: ok=%s out=%s", ok, out)
            self._vllm_stopped = True
            await asyncio.sleep(5)
        await self._terminate_residual_vllm_processes()

    async def _start_vllm(self) -> None:
        """Restart vLLM and wait until its health endpoint responds."""
        if not self.config.MANAGE_VLLM:
            return
        if await self._vllm_healthy():
            self._vllm_stopped = False
            return
        if self._vllm_stopped:
            ok, out = await self._supervisorctl("start", "vllm")
            logger.info("supervisorctl start vllm: ok=%s out=%s", ok, out)
            if not ok:
                raise RuntimeError(f"Could not start vLLM: {out}")
            self._vllm_stopped = False
        else:
            logger.info("Waiting for the supervisor-started vLLM service to become healthy")
        # Model load can take a few minutes.
        for i in range(180):  # up to ~15 min
            if await self._vllm_healthy():
                logger.info("vLLM healthy and ready (%ss)", i * 5)
                return
            await asyncio.sleep(5)
        raise RuntimeError("vLLM did not become healthy within 15 minutes")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _recover_interrupted_tasks(self) -> None:
        """Return tasks interrupted by a container restart to the resume queue."""
        self._migrate_and_validate_historical_worldgen()
        tasks, _ = list_tasks(self.config.DB_PATH, status=TaskStatus.RUNNING.value, limit=100)
        for task in tasks:
            logger.warning("Recovering interrupted task %s", task["id"])
            update_task(
                self.config.DB_PATH,
                task["id"],
                status=TaskStatus.PENDING.value,
                error=None,
            )
        self._recover_prompt_only_failures()

    def _migrate_and_validate_historical_worldgen(self) -> None:
        """Revoke completed Worldgen outputs that cannot pass strict free view.

        This is intentionally read/validate/archive only. It never queues a
        task or launches training, and the migration marker makes restarts
        idempotent.
        """
        completed: list[dict] = []
        offset = 0
        while True:
            page, total = list_tasks(
                self.config.DB_PATH,
                status=TaskStatus.COMPLETED.value,
                limit=100,
                offset=offset,
            )
            completed.extend(page)
            offset += len(page)
            if offset >= total or not page:
                break

        for task in completed:
            task_type = str(task.get("type") or "")
            if task_type not in {"worldgen", "worldgen_from_pano"}:
                continue
            task_dir = self.config.task_dir(task["id"])
            scene_dir = self.config.task_scene_dir(task["id"])
            try:
                recovery = {}
                recovery_path = task_dir / "auto_recovery.json"
                if recovery_path.is_file():
                    recovery = json.loads(recovery_path.read_text())
                if recovery.get("delivery_tier") in {
                    "degraded_3d", "anchored_3d", "panorama_only"
                }:
                    migration = migrate_non_strict_delivery(task_dir, task_type=task_type)
                    self._mark_quality_blocked_task(task, migration)
                    continue

                alignment = {}
                alignment_path = task_dir / "alignment_quality.json"
                if alignment_path.is_file():
                    alignment = json.loads(alignment_path.read_text())
                accepted = alignment.get("accepted_trajectory_ids", [])
                accepted_count = len({value for value in accepted if isinstance(value, str)})
                if accepted_count < 6:
                    migration = migrate_non_strict_delivery(
                        task_dir,
                        task_type=task_type,
                        force=True,
                    )
                    self._mark_quality_blocked_task(task, migration)
                    continue

                build_training_input_manifest(task_dir, scene_dir, "strict_3d")
                validate_completed_delivery(
                    task_dir, task_type=task_type, require_navigation_stress=False
                )
            except (DeliveryValidationError, AutoRecoveryError, json.JSONDecodeError, OSError, ValueError) as exc:
                reason = f"insufficient_free_view_coverage: {exc}. Replace the source and retry."
                try:
                    migration = migrate_non_strict_delivery(
                        task_dir,
                        task_type=task_type,
                        force=True,
                        reason=reason,
                    )
                    self._mark_quality_blocked_task(task, migration)
                except Exception:
                    logger.exception("Could not migrate ineligible historical task %s", task["id"])

    def _mark_quality_blocked_task(self, task: dict, migration: dict | None) -> None:
        if not migration:
            return
        reason = "insufficient_free_view_coverage"
        try:
            recovery = json.loads((self.config.task_dir(task["id"]) / "auto_recovery.json").read_text())
            reason = str(recovery.get("terminal_reason") or reason)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        task_dir = self.config.task_dir(task["id"])
        artifacts = self._collect_artifacts(
            self.config.task_output_dir(task["id"]),
            self.config.task_scene_dir(task["id"]),
            task_dir,
        )
        update_task(
            self.config.DB_PATH,
            task["id"],
            status=TaskStatus.FAILED.value,
            current_stage=None,
            error=reason,
            error_code="insufficient_free_view_coverage",
            error_summary=(
                "Free-view coverage is insufficient for strict 3DGS. "
                "Existing artifacts were preserved; replace the source to retry."
            ),
            artifacts=artifacts,
        )

    def _recover_prompt_only_failures(self) -> None:
        """Idempotently requeue the historical prompt-only missing-input bug."""
        tasks = []
        offset = 0
        while True:
            page, total = list_tasks(
                self.config.DB_PATH,
                status=TaskStatus.FAILED.value,
                limit=100,
                offset=offset,
            )
            tasks.extend(page)
            offset += len(page)
            if offset >= total or not page:
                break
        signature = "input image does not exist:"
        for task in reversed(tasks):
            if task.get("type") != "worldgen":
                continue
            task_dir = self.config.task_dir(task["id"])
            marker = task_dir / ".migration_prompt_source_v1"
            prompt = str((task.get("config") or {}).get("prompt") or "").strip()
            if marker.exists() or not prompt or (task_dir / "input" / "input.png").is_file():
                continue
            if signature not in str(task.get("error") or "").lower():
                continue
            if (task_dir / ".stage_panogen_done").exists():
                continue
            stages = task.get("stages") or []
            if any(stage.get("name") == "panogen" and stage.get("status") == "completed" for stage in stages):
                continue
            log = task_dir / "logs" / "panogen.log"
            archived = task_dir / "logs" / "panogen_missing_input_attempt_000.log"
            if log.is_file() and not archived.exists():
                shutil.copy2(log, archived)
            config = dict(task.get("config") or {})
            config.update({"input_mode": "prompt", "seed": int(config.get("seed", 1024))})
            update_task(
                self.config.DB_PATH,
                task["id"],
                config=config,
                status=TaskStatus.PENDING.value,
                progress=0.0,
                current_stage=None,
                error=None,
                error_code=None,
                error_summary=None,
            )
            marker.touch()

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
    def _replace_option(command: list[str], option: str, value: str) -> list[str]:
        updated = list(command)
        if option in updated:
            updated[updated.index(option) + 1] = value
        else:
            updated.extend([option, value])
        return updated

    async def _run_gs_training_with_safe_fallback(
        self, *, task_dir: Path, scene_dir: Path, task_type: str, **kwargs
    ) -> tuple[bool, str]:
        if (task_dir / "artifact_validation.json").is_file():
            try:
                validate_completed_delivery(
                    task_dir, task_type=task_type, require_navigation_stress=False
                )
                log_file = kwargs.get("log_file")
                if isinstance(log_file, Path):
                    with open(log_file, "a") as log:
                        log.write(
                            "[Artifact Validation] Reusing an existing strict free-view 2,000-step "
                            "PLY/checkpoint/SPZ delivery; training will not be repeated.\n"
                        )
                return True, ""
            except DeliveryValidationError:
                pass
        archived = _archive_incomplete_final_training(task_dir)
        if archived is not None:
            log_file = kwargs.get("log_file")
            if isinstance(log_file, Path):
                with open(log_file, "a") as log:
                    log.write(
                        "[Fresh Final Training] Archived unvalidated output to "
                        f"{archived.relative_to(task_dir)}; restarting the formal 2,000 steps from zero.\n"
                    )
        success, error = await self._run_stage_resilient(task_dir=task_dir, **kwargs)
        validation_error = ""
        try:
            validate_completed_delivery(
                task_dir, task_type=task_type, require_navigation_stress=False
            )
            if not success:
                log_file = kwargs.get("log_file")
                if isinstance(log_file, Path):
                    with open(log_file, "a") as log:
                        log.write(
                            "[Artifact Validation] Training exited non-zero after producing "
                            f"a valid delivery; accepting exported artifacts: {error}\n"
                        )
            return True, ""
        except DeliveryValidationError as exc:
            validation_error = str(exc)
        if success:
            # A deterministic contract failure (metrics, view count, camera
            # coverage, or artifact consistency) cannot be repaired by
            # repeating the identical 2,000-step command with frozen means.
            # Preserve the first output for diagnosis and fail closed instead
            # of spending another full training run.
            return False, validation_error
        content_failure = validation_error or str(error)
        lowered = content_failure.lower()
        if not success and not any(marker in lowered for marker in (
            "out of memory", "cuda oom", "nan", "non-finite", "diverg", "checkpoint", "spz", "gaussian"
        )):
            return False, error
        try:
            prepare_safe_training_retry(task_dir, scene_dir)
        except DeliveryValidationError as exc:
            return False, f"{content_failure}; safe training preparation failed: {exc}"
        retry_root = task_dir / "recovery_attempts"
        retry_root.mkdir(parents=True, exist_ok=True)
        retry_number = len(list(retry_root.glob("strict-training-retry-*"))) + 1
        retry_archive = retry_root / f"strict-training-retry-{retry_number:03d}"
        retry_archive.mkdir(parents=True, exist_ok=False)
        failed_output = task_dir / "output"
        if failed_output.exists():
            os.replace(failed_output, retry_archive / "output")
        for name in ("artifact_validation.json", "navigation_validation.json", "navigation_stress.json"):
            failed_report = task_dir / name
            if failed_report.exists():
                os.replace(failed_report, retry_archive / name)
        safe_command = list(kwargs["cmd"])
        for option, value in (
            ("--means-lr", "0"),
            ("--strategy.refine-start-iter", "1999"),
            ("--strategy.refine-stop-iter", "1999"),
            ("--strategy.refine-every", "99990"),
        ):
            safe_command = self._replace_option(safe_command, option, value)
        kwargs["cmd"] = safe_command
        log_file = kwargs.get("log_file")
        if isinstance(log_file, Path):
            with open(log_file, "a") as log:
                log.write(
                    "[Strict Safe Training] Retrying 2,000 steps with frozen means on the "
                    f"already-approved trajectory set: {content_failure}\n"
                )
        success, error = await self._run_stage_resilient(task_dir=task_dir, **kwargs)
        try:
            validate_completed_delivery(
                task_dir, task_type=task_type, require_navigation_stress=False
            )
        except DeliveryValidationError as exc:
            if not success:
                return False, error
            return False, f"Strict safe training did not produce a valid delivery: {exc}"
        return True, ""

    async def _run_gs_pilot_with_content_recovery(
        self,
        *,
        task_dir: Path,
        scene_dir: Path,
        input_dir: Path,
        task_type: str,
        task_config: dict,
        pilot_kwargs: dict,
        command_map: dict[str, tuple[list[str], dict[str, str], Path]],
        render_spec: tuple[list[str], dict[str, str], Path] | None,
        cancel_event: asyncio.Event,
    ) -> tuple[bool, str]:
        """Run pilot; supplement once, then switch source without reusing its checkpoint."""
        for pilot_round in range(6):
            success, error = await self._run_stage_resilient(task_dir=task_dir, **pilot_kwargs)
            if not success:
                return False, error
            try:
                validate_pilot_output(task_dir)
                return True, ""
            except PilotValidationError as exc:
                try:
                    quality = json.loads((task_dir / "quality_report.json").read_text())
                    planning = quality["planning"]
                    if not isinstance(planning, dict):
                        raise TypeError("planning is not an object")
                except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as planning_exc:
                    return False, f"{exc}; pilot recovery has no planning report: {planning_exc}"

                report = json.loads((task_dir / "auto_recovery.json").read_text())
                source_attempt = int(report.get("source_attempt", 0))
                attempt_root = task_dir / "recovery_attempts" / f"attempt-{source_attempt:03d}"
                failure_number = len(list(attempt_root.glob("pilot-failure-*"))) + 1
                archive = (
                    attempt_root / f"pilot-failure-{failure_number:02d}"
                )
                archive.mkdir(parents=True, exist_ok=False)
                for source in (task_dir / "pilot_output", task_dir / "pilot_validation.json"):
                    if source.exists():
                        os.replace(source, archive / source.name)
                action = request_pilot_content_recovery(
                    task_dir,
                    planning=planning,
                    reason=str(exc),
                )
                if action.get("state") == "quality_blocked":
                    return False, str(action.get("terminal_reason") or exc)

                if action.get("state") == "needs_regeneration":
                    if task_type != "worldgen":
                        return False, "source_replacement_required: upload a new panorama after pilot failure"
                    seed = _next_recovery_seed(task_config, int(action.get("source_attempt", 0)))
                    recovered, recovery_error = await self._run_regenerated_source_attempt(
                        task_dir=task_dir,
                        source_scene_dir=scene_dir,
                        input_dir=input_dir,
                        task_config=task_config,
                        seed=seed,
                        cancel_event=cancel_event,
                        resume=False,
                    )
                else:
                    video_cmd, video_env, video_cwd = command_map["video_gen"]
                    recovered, recovery_error = await self._run_auto_high_quality_video_recovery(
                        task_dir=task_dir,
                        scene_dir=scene_dir,
                        log_file=task_dir / "logs" / f"pilot_recovery_{pilot_round + 1:02d}_video_gen.log",
                        cmd=video_cmd,
                        env=video_env,
                        cwd=video_cwd,
                        cancel_event=cancel_event,
                        task_type=task_type,
                        recovery_policy="auto",
                        render_spec=render_spec,
                        input_dir=input_dir,
                        task_config=task_config,
                    )
                if not recovered:
                    return False, recovery_error

                old_gs_data = scene_dir / "gs_data"
                if old_gs_data.exists():
                    os.replace(old_gs_data, archive / "gs_data")
                gen_cmd, gen_env, gen_cwd = command_map["gen_gs_data"]
                delivery = prepare_training_delivery(task_dir)
                build_training_input_manifest(
                    task_dir,
                    scene_dir,
                    str(delivery.get("delivery_tier") or "strict_3d"),
                )
                generated, generation_error = await self._run_stage_resilient(
                    task_dir=task_dir,
                    stage_name="gen_gs_data",
                    cmd=gen_cmd,
                    env=gen_env,
                    log_file=task_dir / "logs" / f"pilot_recovery_{pilot_round + 1:02d}_gen_gs_data.log",
                    cwd=gen_cwd,
                    cancel_event=cancel_event,
                )
                if not generated:
                    return False, generation_error
                try:
                    postvalidate_worldgen_stage(
                        "gen_gs_data",
                        task_dir=task_dir,
                        scene_dir=scene_dir,
                        log_file=task_dir / "logs" / f"pilot_recovery_{pilot_round + 1:02d}_gen_gs_data.log",
                        quality_preset="high",
                    )
                except QualityGateError as validation_error:
                    return False, str(validation_error)
        return False, "source_replacement_required: pilot recovery exhausted all three validated sources"

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
        if task_type == "worldgen":
            task_config.setdefault(
                "input_mode",
                "image" if (input_dir / "input.png").is_file() else "prompt",
            )
            task_config.setdefault("seed", 1024)
        task_config = normalize_task_contract(task_config, task_type)
        task_config.setdefault("vllm_host", self.config.VLLM_HOST)
        task_config.setdefault("vllm_port", self.config.VLLM_PORT)
        task_config.setdefault("vllm_model", self.config.VLLM_MODEL)
        write_json_atomic(task_dir / "task_contract.json", strict_contract(task_type))
        # v57 prompt tasks may already have paid for panorama/WorldStereo work.
        # Adopt that persisted source into v4 without generating four new
        # images mid-task; the following input_preflight stage still evaluates
        # the exact existing source before any further GPU stage proceeds.
        if task_type == "worldgen" and (task_dir / ".stage_panogen_done").is_file():
            (task_dir / ".stage_source_generate_done").touch()

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
            if not self.config.MOCK_PIPELINE:
                validate_completed_delivery(task_dir, task_type=task_type)
                artifacts = self._collect_artifacts(output_dir, scene_dir, task_dir)
            update_task(
                self.config.DB_PATH, task_id,
                status=TaskStatus.COMPLETED.value,
                progress=1.0,
                artifacts=artifacts,
            )
            completed_task = get_task(self.config.DB_PATH, task_id)
            if completed_task is None:
                raise RuntimeError("Completed task disappeared before delivery manifest creation")
            write_delivery_manifest(task_dir, completed_task, artifacts)
            artifacts = self._collect_artifacts(output_dir, scene_dir, task_dir)
            update_task(self.config.DB_PATH, task_id, artifacts=artifacts)
            logger.info("Task %s completed", task_id)

        except Exception as e:
            logger.exception("Task %s failed", task_id)
            # Surface structured quality evidence even for a deliberately
            # rejected task, so the user can diagnose why it did not proceed.
            artifacts = self._collect_artifacts(output_dir, scene_dir, task_dir)
            source_report = {}
            try:
                source_report = json.loads((task_dir / "source_generation.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            quality_blocked = (
                isinstance(e, (
                    AutoRecoveryError, DeliveryValidationError, InputPreflightError,
                    PilotValidationError, NavigationStressError,
                ))
                or "insufficient_free_view_coverage" in str(e)
                or "source_replacement_required" in str(e)
                or "input_preflight_failed" in str(e)
                or "pilot_content_failed" in str(e)
                or "navigation_stress_failed" in str(e)
                or "gaussian_repair_failed" in str(e)
            )
            message = str(e)
            if "source_replacement_required" in message:
                error_code = "source_replacement_required"
            elif "input_preflight_failed" in message:
                error_code = "input_preflight_failed"
            elif "pilot_content_failed" in message:
                error_code = "pilot_content_failed"
            elif "navigation_stress_failed" in message:
                error_code = "navigation_stress_failed"
            elif "gaussian_repair_failed" in message:
                error_code = "gaussian_repair_failed"
            elif quality_blocked:
                error_code = "insufficient_free_view_coverage"
            else:
                error_code = source_report.get("error_code") or "pipeline_failed"
            error_summary = source_report.get("error_summary")
            if quality_blocked:
                if error_code == "navigation_stress_failed":
                    error_summary = (
                        "Navigation validation did not pass. WASD remains disabled; "
                        "use the guided model repair when available, otherwise replace the source."
                    )
                else:
                    error_summary = (
                        "A strict free-view quality gate did not pass. No WASD-capable model was delivered; "
                        "follow the recommended action in the quality report."
                    )
            elif not error_summary:
                error_summary = f"The {task.get('current_stage') or 'pipeline'} stage could not be completed."
            update_task(
                self.config.DB_PATH, task_id,
                status=TaskStatus.FAILED.value,
                error=str(e),
                error_code=error_code,
                error_summary=error_summary,
                artifacts=artifacts,
            )
            try:
                failed_task = get_task(self.config.DB_PATH, task_id)
                if failed_task is not None:
                    write_delivery_manifest(task_dir, failed_task, artifacts)
                    artifacts = self._collect_artifacts(output_dir, scene_dir, task_dir)
                    update_task(self.config.DB_PATH, task_id, artifacts=artifacts)
            except Exception:
                logger.exception("Could not write failed-task delivery manifest for %s", task_id)
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
        if (
            task_type == "worldgen"
            and recovery_state.get("state") == "completed"
            and int(recovery_state.get("source_attempt", 0) or 0) > 0
            and int(recovery_state.get("best_source_attempt", -1) or -1)
            == int(recovery_state.get("source_attempt", 0) or 0)
        ):
            # A prior release may already have promoted the selected scene but
            # left the task-root alignment manifests pointing at the older
            # source.  Reconcile that evidence before any checkpointed GS
            # stage tries to build its strict training manifest.
            source_attempt = int(recovery_state["source_attempt"])
            attempt_dir = task_dir / "recovery_attempts" / f"attempt-{source_attempt:03d}"
            _archive_current_source_reports(
                task_dir=task_dir,
                attempt_dir=attempt_dir,
                attempt_scene_dir=attempt_dir / "scene",
                source_attempt=source_attempt,
                completed_report=recovery_state,
            )
            regenerated_source_stages = set(ISOLATED_SOURCE_COMPLETED_STAGES)
        if task_type == "worldgen" and recovery_state.get("state") == "needs_regeneration":
            seed = _next_recovery_seed(
                task_config,
                int(recovery_state.get("source_attempt", 0)),
            )
            success, error = await self._run_regenerated_source_attempt(
                task_dir=task_dir,
                source_scene_dir=self.config.task_scene_dir(task_id),
                input_dir=input_dir,
                task_config=task_config,
                seed=seed,
                cancel_event=cancel_event,
                resume=False,
            )
            if not success:
                raise RuntimeError(error)
            regenerated_source_stages = set(ISOLATED_SOURCE_COMPLETED_STAGES)
            for stage_name in regenerated_source_stages:
                (task_dir / f".stage_{stage_name}_done").touch()
            scene_dir = self.config.task_scene_dir(task_id)
        elif task_type == "worldgen" and _should_resume_isolated_source_attempt(recovery_state):
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
            # The regenerated source already completed source selection,
            # preflight, expansion, and WorldStereo in its isolated attempt.
            # Mark those stages as durable root
            # checkpoints so the resumed task proceeds directly to GS data
            # preparation and training instead of returning success early.
            regenerated_source_stages = set(ISOLATED_SOURCE_COMPLETED_STAGES)
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
        elif str(recovery_state.get("phase") or "").startswith("pilot_content") and recovery_state.get(
            "pending_candidate_trajectory_ids"
        ):
            resume_archive = task_dir / "recovery_attempts" / "pilot-recovery-resume"
            resume_archive.mkdir(parents=True, exist_ok=True)
            for name in (".stage_video_gen_done", ".stage_gen_gs_data_done", ".stage_gs_pilot_done"):
                marker = task_dir / name
                if marker.exists() and not (resume_archive / name).exists():
                    os.replace(marker, resume_archive / name)
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
        command_map = {
            name: (command, command_env, command_cwd)
            for name, command, command_env, command_cwd in commands
        }

        for stage_name, cmd, env, cwd in commands:
            if cancel_event.is_set():
                return

            if stage_name == "gen_gs_data" and task_type in ("worldgen", "worldgen_from_pano"):
                recovery_delivery = prepare_training_delivery(task_dir)
                tier = str(recovery_delivery.get("delivery_tier") or "strict_3d")
                build_training_input_manifest(task_dir, scene_dir, tier)

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
                            # A resumed video checkpoint still regenerates
                            # alignment evidence, but strict trajectory count
                            # now selects the 3D delivery tier instead of
                            # blocking GS data preparation.
                            enforce_alignment=stage_name != "video_gen",
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
            if stage_name == "input_preflight":
                await self._start_vllm()
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
                    raise RuntimeError(
                        f"insufficient_free_view_coverage: Stage {stage_name} failed quality validation: {exc}"
                    ) from exc

                if success:
                    marker.touch()
                    update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.COMPLETED)
                    continue
                if cancel_event.is_set():
                    return
                update_stage(self.config.DB_PATH, task_id, stage_name, StageStatus.FAILED, error=error)
                raise RuntimeError(f"Stage {stage_name} failed: {error}")

            stage_kwargs = {
                "stage_name": stage_name,
                "cmd": cmd,
                "env": env,
                "log_file": log_file,
                "cwd": cwd,
                "cancel_event": cancel_event,
            }
            if stage_name == "gs_pilot" and task_type in ("worldgen", "worldgen_from_pano"):
                success, error = await self._run_gs_pilot_with_content_recovery(
                    task_dir=task_dir,
                    scene_dir=scene_dir,
                    input_dir=input_dir,
                    task_type=task_type,
                    task_config=task_config,
                    pilot_kwargs=stage_kwargs,
                    command_map=command_map,
                    render_spec=render_spec,
                    cancel_event=cancel_event,
                )
            elif stage_name == "gs_training" and task_type in ("worldgen", "worldgen_from_pano"):
                success, error = await self._run_gs_training_with_safe_fallback(
                    task_dir=task_dir,
                    scene_dir=scene_dir,
                    task_type=task_type,
                    **stage_kwargs,
                )
            else:
                success, error = await self._run_stage_resilient(task_dir=task_dir, **stage_kwargs)

            if success:
                try:
                    if stage_name == "gs_pilot":
                        validate_pilot_output(task_dir)
                    if task_type in ("worldgen", "worldgen_from_pano"):
                        postvalidate_worldgen_stage(
                            stage_name,
                            task_dir=task_dir,
                            scene_dir=scene_dir,
                            log_file=log_file,
                            quality_preset=task_config["quality_preset"],
                        )
                        if stage_name == "gen_gs_data":
                            delivery = prepare_training_delivery(task_dir)
                            build_training_input_manifest(
                                task_dir, scene_dir, str(delivery.get("delivery_tier") or "strict_3d")
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
            if stage_name in {"source_generate", "panogen"}:
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
                seed = _next_recovery_seed(
                    task_config,
                    int(recovery.get("source_attempt", 0)),
                )
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
                return False, str(
                    recovery.get("terminal_reason")
                    or "insufficient_free_view_coverage: replace the source and retry"
                )
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
        attempt_config["_recovery_base_seed"] = int(
            task_config.get("_recovery_base_seed", task_config.get("seed", 1024))
        )
        attempt_config["seed"] = seed
        attempt_config["_source_attempt"] = source_attempt
        staged_attempt_input = attempt_dir / "input"
        attempt_input_dir = input_dir
        preflight_marker = attempt_dir / ".stage_input_preflight_done"
        adopted_v57_source = bool(
            resume
            and (staged_attempt_input / "input.png").is_file()
            and int(report.get("version", 0) or 0) < 4
        )
        adopted_isolated_source = bool(
            resume
            and (staged_attempt_input / "input.png").is_file()
            and (adopted_v57_source or preflight_marker.exists())
        )
        if adopted_isolated_source:
            # A v57 prompt recovery may already have paid for HunyuanImage
            # generation when the rollout happens. Adopt that exact atomic
            # source file, preflight it below, and never regenerate or replace
            # it merely because the journal contract was migrated to v4.
            attempt_input_dir = staged_attempt_input
            shutil.copy2(attempt_input_dir / "input.png", attempt_input_dir / "selected_source.png")
            attempt_config["input_mode"] = "image"
        elif attempt_config.get("input_mode") == "prompt":
            attempt_input_dir = staged_attempt_input
            attempt_input_dir.mkdir(parents=True, exist_ok=True)
            try:
                selected = select_ranked_source_for_attempt(
                    task_dir,
                    source_attempt,
                    attempt_input_dir / "selected_source.png",
                )
            except InputPreflightError as exc:
                return False, str(exc)
            shutil.copy2(attempt_input_dir / "selected_source.png", attempt_input_dir / "input.png")
            attempt_config["input_mode"] = "image"
            attempt_config["seed"] = int(selected.get("seed", seed))
        elif (staged_attempt_input / "input.png").is_file():
            attempt_input_dir = staged_attempt_input
            attempt_input_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(attempt_input_dir / "input.png", attempt_input_dir / "selected_source.png")
        commands = get_worldgen_commands(
            task_dir=task_dir,
            scene_dir=attempt_scene_dir,
            input_dir=attempt_input_dir,
            config=attempt_config,
            hyworld_repo=self.config.HYWORLD_REPO,
            vllm_host=self.config.VLLM_HOST,
            vllm_port=self.config.VLLM_PORT,
            vllm_model=self.config.VLLM_MODEL,
            task_type="worldgen",
        )
        by_stage = {name: (command, env, cwd) for name, command, env, cwd in commands}
        if adopted_isolated_source:
            if not preflight_marker.exists():
                await self._start_vllm()
                command, env, cwd = by_stage["input_preflight"]
                success, error = await self._run_stage_resilient(
                    task_dir=task_dir,
                    stage_name="input_preflight",
                    cmd=command,
                    env=env,
                    log_file=task_dir / "logs" / f"recovery_attempt_{source_attempt:03d}_input_preflight.log",
                    cwd=cwd,
                    cancel_event=cancel_event,
                )
                if not success:
                    return False, error
                preflight_marker.touch()
        for stage_name in ("panogen", "post_panorama_preflight", "traj_generate", "traj_render"):
            marker = attempt_dir / f".stage_{stage_name}_done"
            if marker.exists():
                continue
            if cancel_event.is_set():
                return False, "Cancelled by user"
            command, env, cwd = by_stage[stage_name]
            if stage_name == "panogen":
                await self._stop_vllm()
            elif stage_name in {"post_panorama_preflight", "traj_generate"}:
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
            input_dir=attempt_input_dir,
            task_config=attempt_config,
        )
        if not success:
            return False, error

        try:
            completed_report = json.loads((task_dir / "auto_recovery.json").read_text())
            _archive_current_source_reports(
                task_dir=task_dir,
                attempt_dir=attempt_dir,
                attempt_scene_dir=attempt_scene_dir,
                source_attempt=source_attempt,
                completed_report=completed_report,
            )
            best_attempt = int(completed_report.get("best_source_attempt", source_attempt))
            if best_attempt == source_attempt:
                promote_recovered_scene(
                    task_dir,
                    source_attempt=source_attempt,
                    regenerated_scene_dir=attempt_scene_dir,
                    canonical_scene_dir=source_scene_dir,
                )
                _promote_recovery_source_artifacts(
                    attempt_dir=attempt_dir,
                    attempt_input_dir=attempt_input_dir,
                    task_dir=task_dir,
                    input_dir=input_dir,
                )
                set_active_scene(task_dir, "scene")
            else:
                restore_best_attempt_reports(task_dir, source_attempt=best_attempt)
                if best_attempt > 0:
                    best_dir = task_dir / "recovery_attempts" / f"attempt-{best_attempt:03d}"
                    _promote_recovery_source_artifacts(
                        attempt_dir=best_dir,
                        attempt_input_dir=best_dir / "input",
                        task_dir=task_dir,
                        input_dir=input_dir,
                    )
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
        for stage_name in ("post_panorama_preflight", "traj_generate", "traj_render"):
            marker = attempt_scene_dir.parent / f".stage_{stage_name}_done"
            if marker.exists():
                continue
            if cancel_event.is_set():
                return False, "Cancelled by user"
            command, env, cwd = by_stage[stage_name]
            if stage_name in {"post_panorama_preflight", "traj_generate"}:
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
        search_dirs = [output_dir, scene_dir]
        if task_dir:
            search_dirs.append(task_dir / "input")
        for search_dir in search_dirs:
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
                "source_generation.json",
                "source_pool.json",
                "input_preflight.json",
                "post_panorama_preflight.json",
                "task_contract.json",
                "training_input_manifest.json",
                "pilot_validation.json",
                "artifact_validation.json",
                "navigation_validation.json",
                "navigation_stress.json",
                "gaussian_repair.json",
                "gaussian_repair_request.json",
                "viewer_metadata_validation.json",
                "source_retry.json",
                "safe_training_retry.json",
                "strict_free_view_migration.json",
                "delivery_manifest.json",
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
