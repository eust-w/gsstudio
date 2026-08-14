"""Durable, bounded automatic recovery for high-quality WorldStereo tasks.

The recovery policy intentionally operates above HY-World: it never changes an
alignment threshold and only adds camera paths that have already been proven
collision-safe by the regular planner.  The JSON journal is written atomically
so a Kubernetes restart resumes the exact pending inference batch.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable


AUTO_RECOVERY_REPORT_NAME = "auto_recovery.json"
STRICT_FREE_VIEW_MIGRATION_NAME = "strict_free_view_migration.json"
AUTO_RECOVERY_VERSION = 4
MINIMUM_ACCEPTED_TRAJECTORIES = 6
TARGET_ACCEPTED_TRAJECTORIES = 10
INITIAL_NORMAL_CANDIDATE_COUNT = 8
NORMAL_REFILL_CANDIDATE_COUNT = 4
RESCUE_TARGET_ROTATIONS = {1: 25.0, 2: 12.0}
MAX_RESCUE_CANDIDATE_COUNT = 6
MAX_VIDEO_TRAJECTORIES = 24
MAX_VIDEO_SECONDS = 4 * 60 * 60
# Attempts are zero-based: source 0 plus two switches gives three sources.
MAX_SOURCE_REGENERATIONS = 2
MAX_TRAJECTORIES_PER_SOURCE = 16
# This is an early-recovery trigger, not an acceptance threshold. Real strict
# deliveries clear the immutable 0.80 quality gate; a complete initial batch
# whose best worst-frame guidance is still below 0.60 is source-limited, so
# repeating smaller camera prefixes only spends GPU time on the same bad prior.
GUIDANCE_COLLAPSE_CUTOFF = 0.60


class AutoRecoveryError(RuntimeError):
    """Raised when a safe automatic recovery action cannot be constructed."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_object_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temporary:
            json.dump(value, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _ordered_unique(ids: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in ids:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _normal_candidates(planning: dict[str, Any]) -> list[str]:
    trajectories = planning.get("trajectories", [])
    if not isinstance(trajectories, list):
        return []
    normalized = [
        item for item in trajectories
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    # Regular trajectories are most reliable for indoor WorldStereo alignment.
    # Keep remaining WorldNav classes deterministic as a fallback, not a first
    # response to a low-guidance failure.
    priority = {"regular": 0, "surround": 1, "reconstruction": 2, "exploration": 3}
    def sort_key(item: dict[str, Any]) -> tuple[int, int, float, str]:
        value = item.get("estimated_overlap_score")
        try:
            overlap = float(value)
        except (TypeError, ValueError):
            return (priority.get(str(item.get("kind")), 4), 1, 0.0, item["id"])
        if not math.isfinite(overlap):
            return (priority.get(str(item.get("kind")), 4), 1, 0.0, item["id"])
        return (priority.get(str(item.get("kind")), 4), 0, -overlap, item["id"])

    normalized.sort(key=sort_key)
    return _ordered_unique(item["id"] for item in normalized)


def _legacy_alignment(task_dir: Path) -> tuple[list[str], list[str]]:
    alignment = _read_object(task_dir / "alignment_quality.json")
    trajectories = alignment.get("trajectories", {})
    attempted = list(trajectories) if isinstance(trajectories, dict) else []
    accepted = alignment.get("accepted_trajectory_ids", [])
    return _ordered_unique(attempted), _ordered_unique(accepted if isinstance(accepted, list) else [])


def _new_report(task_dir: Path, planning: dict[str, Any], task_type: str) -> dict[str, Any]:
    legacy_attempted, legacy_accepted = _legacy_alignment(task_dir)
    legacy = bool(legacy_attempted)
    report: dict[str, Any] = {
        "version": AUTO_RECOVERY_VERSION,
        "contract_version": AUTO_RECOVERY_VERSION,
        "success_policy": "strict_free_view",
        "input_lane": "production_multiview" if task_type == "worldrecon" else "generative_worldgen",
        "task_type": task_type,
        "state": "legacy_alignment" if legacy else "ready",
        "phase": "legacy" if legacy else None,
        "rescue_round": None,
        "source_attempt": 0,
        "active_scene_relpath": "scene",
        "minimum_accepted_trajectories": MINIMUM_ACCEPTED_TRAJECTORIES,
        "target_accepted_trajectories": TARGET_ACCEPTED_TRAJECTORIES,
        "limits": {
            "maximum_video_trajectories": MAX_VIDEO_TRAJECTORIES,
            "maximum_video_seconds": MAX_VIDEO_SECONDS,
            "maximum_source_regenerations": MAX_SOURCE_REGENERATIONS,
            "maximum_source_attempts": MAX_SOURCE_REGENERATIONS + 1,
            "maximum_trajectories_per_source": MAX_TRAJECTORIES_PER_SOURCE,
        },
        "normal_candidate_trajectory_ids": _normal_candidates(planning),
        "attempted_trajectory_ids": legacy_attempted,
        "accepted_trajectory_ids": legacy_accepted,
        "pending_candidate_trajectory_ids": [],
        "rendered_candidate_trajectory_ids": [],
        "generated_video_trajectory_count": 0,
        "video_seconds": 0.0,
        "batches": ([{
            "phase": "legacy",
            "candidate_trajectory_ids": legacy_attempted,
            "accepted_trajectory_ids": legacy_accepted,
            "historical": True,
        }] if legacy else []),
        "attempt_history": [],
        "source_attempts": [],
        "best_source_attempt": 0,
        "delivery_tier": None,
        "training_mode": None,
        "strict_accepted_count": len(legacy_accepted),
        "guaranteed_3d": False,
        "free_view_eligible": False,
        "quality_warning": None,
        "recovery_exhausted": False,
        "transient_retry_count": 0,
        "source_guidance_collapse": False,
        "latest_guidance_diagnostics": None,
        "terminal_reason": None,
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
    }
    return report


def _migrate_report(
    report: dict[str, Any],
    *,
    planning: dict[str, Any] | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Migrate an existing journal without discarding paid GPU work."""
    migrated = dict(report)
    migrated["version"] = AUTO_RECOVERY_VERSION
    migrated["contract_version"] = AUTO_RECOVERY_VERSION
    migrated["success_policy"] = "strict_free_view"
    migrated["task_type"] = str(task_type or migrated.get("task_type") or "worldgen")
    migrated.setdefault("state", "ready")
    migrated.setdefault("phase", None)
    migrated.setdefault("rescue_round", None)
    migrated.setdefault("source_attempt", 0)
    migrated.setdefault("active_scene_relpath", "scene")
    migrated.setdefault("minimum_accepted_trajectories", MINIMUM_ACCEPTED_TRAJECTORIES)
    migrated["target_accepted_trajectories"] = TARGET_ACCEPTED_TRAJECTORIES
    migrated.setdefault(
        "input_lane",
        "production_multiview" if migrated["task_type"] == "worldrecon" else "generative_worldgen",
    )
    migrated["limits"] = {
        "maximum_video_trajectories": MAX_VIDEO_TRAJECTORIES,
        "maximum_video_seconds": MAX_VIDEO_SECONDS,
        "maximum_source_regenerations": MAX_SOURCE_REGENERATIONS,
        "maximum_source_attempts": MAX_SOURCE_REGENERATIONS + 1,
        "maximum_trajectories_per_source": MAX_TRAJECTORIES_PER_SOURCE,
    }
    if planning is not None and not migrated.get("normal_candidate_trajectory_ids"):
        migrated["normal_candidate_trajectory_ids"] = _normal_candidates(planning)
    migrated.setdefault("normal_candidate_trajectory_ids", [])
    for key in (
        "attempted_trajectory_ids",
        "accepted_trajectory_ids",
        "pending_candidate_trajectory_ids",
        "rendered_candidate_trajectory_ids",
    ):
        migrated[key] = _ordered_unique(migrated.get(key, []))
    migrated.setdefault("generated_video_trajectory_count", len(migrated["attempted_trajectory_ids"]))
    migrated.setdefault("video_seconds", 0.0)
    migrated.setdefault("batches", [])
    migrated.setdefault("attempt_history", [])
    migrated.setdefault("source_attempts", [])
    migrated.setdefault("best_source_attempt", int(migrated.get("source_attempt", 0)))
    migrated.setdefault("delivery_tier", None)
    migrated.setdefault("training_mode", None)
    migrated.setdefault("strict_accepted_count", len(migrated["accepted_trajectory_ids"]))
    migrated["guaranteed_3d"] = False
    migrated.setdefault("free_view_eligible", False)
    migrated.setdefault("quality_warning", None)
    migrated.setdefault("recovery_exhausted", False)
    migrated.setdefault("transient_retry_count", 0)
    migrated.setdefault("source_guidance_collapse", False)
    migrated.setdefault("latest_guidance_diagnostics", None)
    migrated.setdefault("terminal_reason", None)
    migrated.setdefault("created_at", _utcnow())
    migrated.setdefault("updated_at", _utcnow())
    if migrated.get("delivery_tier") in {"degraded_3d", "anchored_3d", "panorama_only"}:
        accepted_count = len(migrated["accepted_trajectory_ids"])
        migrated["state"] = "quality_blocked"
        migrated["delivery_tier"] = None
        migrated["training_mode"] = None
        migrated["strict_accepted_count"] = accepted_count
        migrated["free_view_eligible"] = False
        migrated["quality_warning"] = None
        migrated["terminal_reason"] = (
            "insufficient_free_view_coverage: strict 3DGS requires at least "
            f"{MINIMUM_ACCEPTED_TRAJECTORIES} accepted trajectories; only {accepted_count} passed."
        )
    return migrated


def prepare_training_delivery(task_dir: Path) -> dict[str, Any]:
    """Require and persist strict free-view eligibility before GS training."""
    stored = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    if not stored:
        report = _new_report(task_dir, {}, "worldgen")
        _, accepted = _legacy_alignment(task_dir)
        report["accepted_trajectory_ids"] = accepted
        report["attempted_trajectory_ids"] = accepted
    else:
        report = _migrate_report(stored)
    if not report.get("delivery_tier"):
        if len(_ordered_unique(report.get("accepted_trajectory_ids", []))) >= MINIMUM_ACCEPTED_TRAJECTORIES:
            _finalize_delivery(report, exhausted=False, reason="Strict free-view coverage selected for training")
        else:
            _block_delivery(report, exhausted=bool(report.get("recovery_exhausted")))
    if report.get("delivery_tier") != "strict_3d":
        _persist(task_dir, report)
        raise AutoRecoveryError(str(report.get("terminal_reason") or "Strict free-view coverage is not satisfied"))
    return _persist(task_dir, report)


def migrate_non_strict_delivery(
    task_dir: Path,
    *,
    task_type: str = "worldgen",
    force: bool = False,
    reason: str | None = None,
) -> dict[str, Any] | None:
    """Revoke one legacy non-strict delivery without deleting its evidence.

    Evidence is archived before the public recovery journal is changed, and
    the marker is written last as the migration commit. Re-running this
    function after a restart is therefore harmless and never schedules GPU
    work.
    """
    marker_path = task_dir / STRICT_FREE_VIEW_MIGRATION_NAME
    marker = _read_object(marker_path)
    if marker:
        return marker

    report_path = task_dir / AUTO_RECOVERY_REPORT_NAME
    stored = _read_object(report_path)
    previous_tier = stored.get("delivery_tier")
    if not force and previous_tier not in {"degraded_3d", "anchored_3d", "panorama_only"}:
        return None

    archive_dir = task_dir / "recovery_attempts" / "strict_free_view_v3_migration"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        AUTO_RECOVERY_REPORT_NAME,
        "quality_report.json",
        "training_input_manifest.json",
        "artifact_validation.json",
        "navigation_validation.json",
    ):
        source = task_dir / name
        destination = archive_dir / name
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
    logs_dir = task_dir / "logs"
    if logs_dir.is_dir():
        archived_logs = archive_dir / "logs"
        shutil.copytree(logs_dir, archived_logs, dirs_exist_ok=True)

    migrated = (
        _migrate_report(stored, task_type=task_type)
        if stored
        else _new_report(task_dir, {}, task_type)
    )
    _block_delivery(migrated, exhausted=True)
    migrated["recovery_exhausted"] = True
    migrated["terminal_reason"] = reason or (
        "insufficient_free_view_coverage: this historical non-strict delivery "
        "does not qualify for WASD navigation. Replace the source and retry."
    )
    migration = {
        "version": 1,
        "state": "completed",
        "migrated_at": _utcnow(),
        "previous_delivery_tier": previous_tier,
        "new_status": "failed",
        "error_code": "insufficient_free_view_coverage",
        "archive_relpath": str(archive_dir.relative_to(task_dir)),
        "artifacts_preserved": True,
        "gpu_work_scheduled": False,
    }
    _persist(task_dir, migrated)
    _write_object_atomic(marker_path, migration)
    return migration


def _load_or_create(task_dir: Path, planning: dict[str, Any], task_type: str) -> dict[str, Any]:
    report = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    if report:
        return _migrate_report(report, planning=planning, task_type=task_type)
    return _new_report(task_dir, planning, task_type)


def _persist(task_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    report["updated_at"] = _utcnow()
    _write_object_atomic(task_dir / AUTO_RECOVERY_REPORT_NAME, report)
    # Keep the task's single quality artifact useful even while the worker is
    # mid-recovery.  This mirrors the existing planning/alignment reports and
    # lets artifact download expose a stable, complete diagnosis.
    quality_path = task_dir / "quality_report.json"
    quality = _read_object(quality_path)
    quality["auto_recovery"] = report
    _write_object_atomic(quality_path, quality)
    return report


def _merge_alignment(report: dict[str, Any], alignment: dict[str, Any]) -> None:
    accepted = alignment.get("accepted_trajectory_ids", [])
    if isinstance(accepted, list):
        report["accepted_trajectory_ids"] = _ordered_unique([
            *report.get("accepted_trajectory_ids", []),
            *accepted,
        ])


def _guidance_collapse_diagnostics(
    alignment: dict[str, Any],
    candidate_ids: Iterable[str],
) -> dict[str, Any]:
    """Classify an initial batch whose guidance is complete but uniformly weak."""
    trajectories = alignment.get("trajectories", {})
    if not isinstance(trajectories, dict):
        trajectories = {}
    candidates = _ordered_unique(candidate_ids)
    observed: list[float] = []
    missing: list[str] = []
    for trajectory_id in candidates:
        trajectory = trajectories.get(trajectory_id)
        if not isinstance(trajectory, dict):
            missing.append(trajectory_id)
            continue
        value = trajectory.get("minimum_guidance_mask")
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            missing.append(trajectory_id)
            continue
        if not math.isfinite(normalized):
            missing.append(trajectory_id)
            continue
        observed.append(normalized)
    best = max(observed) if observed else None
    detected = bool(candidates) and not missing and best is not None and best < GUIDANCE_COLLAPSE_CUTOFF
    return {
        "detected": detected,
        "diagnostic_cutoff": GUIDANCE_COLLAPSE_CUTOFF,
        "candidate_count": len(candidates),
        "observed_count": len(observed),
        "missing_trajectory_ids": missing,
        "best_minimum_guidance_mask": best,
    }


def _budget_exhausted(report: dict[str, Any]) -> bool:
    limits = report["limits"]
    elapsed = 0.0
    try:
        created = datetime.fromisoformat(str(report.get("created_at", "")).replace("Z", "+00:00"))
        elapsed = max((_utc_datetime() - created).total_seconds(), 0.0)
    except (TypeError, ValueError):
        pass
    report["elapsed_seconds"] = elapsed
    return (
        int(report.get("generated_video_trajectory_count", 0)) >= int(limits["maximum_video_trajectories"])
        or float(report.get("video_seconds", 0.0)) >= float(limits["maximum_video_seconds"])
        or elapsed >= float(limits["maximum_video_seconds"])
    )


def _utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def desired_rescue_candidate_count(accepted_count: int) -> int:
    """Adaptive rescue size: one spare beyond the gap, bounded to 2..6."""
    missing = max(MINIMUM_ACCEPTED_TRAJECTORIES - max(int(accepted_count), 0), 0)
    return min(MAX_RESCUE_CANDIDATE_COUNT, max(2, missing + 1))


def _delivery_tier(accepted_count: int) -> str | None:
    if accepted_count >= MINIMUM_ACCEPTED_TRAJECTORIES:
        return "strict_3d"
    return None


def _best_source_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    attempts = [item for item in report.get("source_attempts", []) if isinstance(item, dict)]
    excluded_attempts = {
        int(value) for value in report.get("pilot_failed_source_attempts", [])
        if isinstance(value, (int, str)) and str(value).isdigit()
    }
    eligible_attempts = [
        item for item in attempts
        if int(item.get("source_attempt", -1)) not in excluded_attempts
    ]
    attempts = eligible_attempts
    current_attempt = int(report.get("source_attempt", 0))
    if current_attempt not in excluded_attempts and not any(
        int(item.get("source_attempt", -1)) == current_attempt for item in attempts
    ):
        accepted = _ordered_unique(report.get("accepted_trajectory_ids", []))
        attempts.append({
            "source_attempt": current_attempt,
            "scene_relpath": report.get("active_scene_relpath", "scene"),
            "accepted_trajectory_ids": accepted,
            "accepted_count": len(accepted),
            "quality_margin": 0.0,
            "manual_source_retry": bool(report.get("manual_source_retry")),
            "batches": report.get("batches", []),
        })
    if not attempts:
        accepted = _ordered_unique(report.get("accepted_trajectory_ids", []))
        attempts.append({
            "source_attempt": current_attempt,
            "scene_relpath": report.get("active_scene_relpath", "scene"),
            "accepted_trajectory_ids": accepted,
            "accepted_count": len(accepted),
            "quality_margin": 0.0,
            "manual_source_retry": bool(report.get("manual_source_retry")),
            "batches": report.get("batches", []),
            "pilot_failed": current_attempt in excluded_attempts,
        })
    return max(
        attempts,
        key=lambda item: (
            int(item.get("accepted_count", len(_ordered_unique(item.get("accepted_trajectory_ids", []))))),
            float(item.get("quality_margin", 0.0)),
            int(bool(item.get("manual_source_retry"))),
            (
                int(item.get("source_attempt", 0))
                if item.get("manual_source_retry")
                else -int(item.get("source_attempt", 0))
            ),
        ),
    )


def _block_delivery(report: dict[str, Any], *, exhausted: bool) -> None:
    accepted_count = len(_ordered_unique(report.get("accepted_trajectory_ids", [])))
    report["state"] = "quality_blocked"
    report["phase"] = None
    report["rescue_round"] = None
    report["delivery_tier"] = None
    report["training_mode"] = None
    report["strict_accepted_count"] = accepted_count
    report["guaranteed_3d"] = False
    report["free_view_eligible"] = False
    report["quality_warning"] = None
    report["recovery_exhausted"] = bool(exhausted)
    code = "source_replacement_required" if exhausted else "insufficient_free_view_coverage"
    report["terminal_reason"] = (
        f"{code}: strict 3DGS requires at least {MINIMUM_ACCEPTED_TRAJECTORIES} accepted trajectories; "
        f"only {accepted_count} passed. Replace the source and retry."
    )


def _finalize_delivery(report: dict[str, Any], *, exhausted: bool, reason: str) -> None:
    best = _best_source_snapshot(report)
    best_ids = _ordered_unique(best.get("accepted_trajectory_ids", []))
    report["best_source_attempt"] = int(best.get("source_attempt", 0))
    report["best_scene_relpath"] = best.get("scene_relpath", "scene")
    report["accepted_trajectory_ids"] = best_ids
    accepted_count = len(best_ids)
    tier = _delivery_tier(accepted_count)
    if tier is None:
        _block_delivery(report, exhausted=exhausted)
        return
    report["state"] = "completed"
    report["phase"] = None
    report["rescue_round"] = None
    report["delivery_tier"] = tier
    report["training_mode"] = "accepted_trajectories"
    report["strict_accepted_count"] = accepted_count
    report["guaranteed_3d"] = False
    report["free_view_eligible"] = False
    report["quality_warning"] = None
    report["recovery_exhausted"] = bool(exhausted)
    report["terminal_reason"] = reason


def _set_terminal(report: dict[str, Any], state: str, reason: str) -> None:
    report["state"] = state
    report["phase"] = None
    report["rescue_round"] = None
    report["terminal_reason"] = reason


def schedule_auto_recovery_batch(
    task_dir: Path,
    *,
    phase: str,
    candidate_ids: Iterable[str],
) -> dict[str, Any]:
    """Persist the exact next batch before an expensive subprocess starts."""
    stored = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    if not stored:
        raise AutoRecoveryError("Automatic recovery report has not been initialized")
    report = _migrate_report(stored)
    pending = _ordered_unique(report.get("pending_candidate_trajectory_ids", []))
    if pending:
        return report
    if _budget_exhausted(report):
        _finalize_delivery(
            report,
            exhausted=True,
            reason="Automatic recovery video budget is exhausted; the best valid result was delivered.",
        )
        return _persist(task_dir, report)

    capacity = int(report["limits"]["maximum_video_trajectories"]) - int(
        report.get("generated_video_trajectory_count", 0)
    )
    batch = _ordered_unique(candidate_ids)[:max(capacity, 0)]
    if not batch:
        return report
    report["state"] = "generating_normal" if phase.startswith("normal") else "generating_rescue"
    report["phase"] = phase
    report["rescue_round"] = int(phase.rsplit("_", 1)[-1]) if phase.startswith("rescue_") else None
    report["pending_candidate_trajectory_ids"] = batch
    report["terminal_reason"] = None
    return _persist(task_dir, report)


def record_rescue_skipped(
    task_dir: Path,
    *,
    round_number: int,
    reason: str,
) -> dict[str, Any]:
    """Advance a rescue phase that has no safe camera sources."""
    stored = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    if not stored:
        raise AutoRecoveryError("Automatic recovery report has not been initialized")
    report = _migrate_report(stored)
    phase = f"rescue_{int(round_number)}"
    batches = report.get("batches", [])
    if not isinstance(batches, list):
        batches = []
    if not any(isinstance(item, dict) and item.get("phase") == phase for item in batches):
        batches.append({
            "phase": phase,
            "candidate_trajectory_ids": [],
            "accepted_trajectory_ids": [],
            "rescue_skipped": True,
            "reason": str(reason),
            "completed_at": _utcnow(),
        })
    report["batches"] = batches
    report["state"] = "ready"
    report["phase"] = None
    report["rescue_round"] = None
    report["pending_candidate_trajectory_ids"] = []
    return _persist(task_dir, report)


def record_transient_retry(
    task_dir: Path,
    *,
    stage: str,
    error: str,
    attempt: int,
) -> dict[str, Any]:
    """Journal a retryable infrastructure interruption without losing state."""
    stored = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    if not stored:
        raise AutoRecoveryError("Automatic recovery report has not been initialized")
    report = _migrate_report(stored)
    retries = report.get("transient_retries", [])
    if not isinstance(retries, list):
        retries = []
    retries.append({
        "stage": str(stage),
        "attempt": max(int(attempt), 1),
        "error": str(error)[-2000:],
        "recorded_at": _utcnow(),
    })
    report["transient_retries"] = retries
    report["transient_retry_count"] = len(retries)
    return _persist(task_dir, report)


def record_recovery_render_complete(
    task_dir: Path,
    *,
    candidate_ids: list[str],
) -> dict[str, Any]:
    """Durably mark a rescue render/caption batch before WorldStereo starts.

    A container or process failure can happen after the camera videos were
    written but before the pending WorldStereo batch completes. Keeping this
    checkpoint separate lets retry either reuse a fully rendered batch or
    safely rerun ``--skip_exist`` to fill missing captions.
    """
    stored = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    if not stored:
        raise AutoRecoveryError("Automatic recovery report has not been initialized")
    report = _migrate_report(stored)
    report["rendered_candidate_trajectory_ids"] = _ordered_unique([
        *report.get("rendered_candidate_trajectory_ids", []),
        *candidate_ids,
    ])
    report["render_completed_at"] = _utcnow()
    return _persist(task_dir, report)


def prepare_auto_recovery(
    task_dir: Path,
    *,
    planning: dict[str, Any],
    task_type: str,
    recovery_policy: str,
) -> dict[str, Any]:
    """Return the next durable recovery action without spending GPU work."""
    if recovery_policy != "auto":
        return {
            "state": "disabled",
            "phase": None,
            "pending_candidate_trajectory_ids": [],
            "accepted_trajectory_ids": [],
        }

    report = _load_or_create(task_dir, planning, task_type)
    if not report.get("normal_candidate_trajectory_ids"):
        report["normal_candidate_trajectory_ids"] = _normal_candidates(planning)

    pending = _ordered_unique(report.get("pending_candidate_trajectory_ids", []))
    if pending:
        report["pending_candidate_trajectory_ids"] = pending
        return _persist(task_dir, report)

    # Scheduling is deliberately a separate durable transition.  Persist the
    # newly initialized journal first so a crash between initialization and
    # the candidate manifest can never fabricate a second initial batch.
    _persist(task_dir, report)

    accepted_count = len(_ordered_unique(report.get("accepted_trajectory_ids", [])))
    if accepted_count >= TARGET_ACCEPTED_TRAJECTORIES:
        _finalize_delivery(report, exhausted=False, reason="Ten-trajectory recovery target satisfied")
        return _persist(task_dir, report)
    if _budget_exhausted(report):
        if accepted_count >= MINIMUM_ACCEPTED_TRAJECTORIES:
            _finalize_delivery(
                report,
                exhausted=True,
                reason="Global recovery budget ended after the strict six-trajectory minimum passed.",
            )
        else:
            _finalize_delivery(
                report,
                exhausted=True,
                reason="Global recovery budget ended before strict free-view coverage was satisfied.",
            )
        return _persist(task_dir, report)

    batches = report.get("batches", [])
    phases = [batch.get("phase") for batch in batches if isinstance(batch, dict)]
    normal_candidates = _ordered_unique(report.get("normal_candidate_trajectory_ids", []))
    attempted = set(_ordered_unique(report.get("attempted_trajectory_ids", [])))
    source_attempted_count = len(attempted)

    if "normal_initial" not in phases and "legacy" not in phases:
        return schedule_auto_recovery_batch(
            task_dir,
            phase="normal_initial",
            candidate_ids=[candidate for candidate in normal_candidates if candidate not in attempted][:INITIAL_NORMAL_CANDIDATE_COUNT],
        )

    if "normal_initial" in phases and accepted_count == 0 and report.get("source_guidance_collapse"):
        diagnostics = report.get("latest_guidance_diagnostics") or {}
        best = diagnostics.get("best_minimum_guidance_mask")
        detail = f"; best worst-frame guidance was {best:.4f}" if isinstance(best, (int, float)) else ""
        if task_type == "worldgen" and int(report.get("source_attempt", 0)) < int(
            report["limits"]["maximum_source_regenerations"]
        ):
            report["state"] = "needs_regeneration"
            report["phase"] = "source_guidance_collapse"
            report["rescue_round"] = None
            report["terminal_reason"] = (
                "Initial WorldStereo guidance collapsed across every candidate"
                f"{detail}; regenerating the source before rescue trajectories."
            )
            return _persist(task_dir, report)
        _finalize_delivery(
            report,
            exhausted=True,
            reason=(
                "Initial WorldStereo guidance collapsed across every candidate"
                f"{detail}; replace the source image or panorama to satisfy strict free-view coverage."
            ),
        )
        return _persist(task_dir, report)

    if source_attempted_count >= MAX_TRAJECTORIES_PER_SOURCE and accepted_count < MINIMUM_ACCEPTED_TRAJECTORIES:
        if task_type == "worldgen" and int(report.get("source_attempt", 0)) < int(
            report["limits"]["maximum_source_regenerations"]
        ):
            report["state"] = "needs_regeneration"
            report["phase"] = "source_under_minimum_after_16"
            report["rescue_round"] = None
            report["terminal_reason"] = (
                f"Source {report.get('source_attempt', 0)} produced only {accepted_count} strict trajectories "
                f"after {source_attempted_count} attempts; switching to the next preflight-ranked source."
            )
            return _persist(task_dir, report)
        _finalize_delivery(
            report,
            exhausted=True,
            reason="This source exhausted 16 trajectories before the strict minimum.",
        )
        return _persist(task_dir, report)

    # A legacy v33 task already spent its normal-path budget.  Do not repeat
    # those rejected paths merely because its new journal did not exist yet.
    if (
        "normal_initial" in phases
        and 4 <= accepted_count < TARGET_ACCEPTED_TRAJECTORIES
        and "normal_refill" not in phases
        and any(candidate not in attempted for candidate in normal_candidates)
    ):
        return schedule_auto_recovery_batch(
            task_dir,
            phase="normal_refill",
            candidate_ids=[candidate for candidate in normal_candidates if candidate not in attempted][:NORMAL_REFILL_CANDIDATE_COUNT],
        )

    if "rescue_1" not in phases:
        report["state"] = "needs_rescue"
        report["phase"] = "rescue_1"
        report["rescue_round"] = 1
        return _persist(task_dir, report)
    if "rescue_2" not in phases:
        report["state"] = "needs_rescue"
        report["phase"] = "rescue_2"
        report["rescue_round"] = 2
        return _persist(task_dir, report)

    if accepted_count >= MINIMUM_ACCEPTED_TRAJECTORIES:
        _finalize_delivery(
            report,
            exhausted=False,
            reason="Recovery target exhausted after the strict six-trajectory minimum passed.",
        )
        return _persist(task_dir, report)

    if task_type == "worldgen" and int(report.get("source_attempt", 0)) < int(report["limits"]["maximum_source_regenerations"]):
        report["state"] = "needs_regeneration"
        report["phase"] = "source_regeneration"
        report["rescue_round"] = None
        return _persist(task_dir, report)

    _finalize_delivery(
        report,
        exhausted=True,
        reason="All validated source candidates are exhausted before strict free-view coverage.",
    )
    return _persist(task_dir, report)


def complete_auto_recovery_batch(
    task_dir: Path,
    *,
    alignment: dict[str, Any],
    duration_seconds: float,
) -> dict[str, Any]:
    """Commit a completed video batch once, including all accepted routes."""
    stored = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    if not stored:
        raise AutoRecoveryError("Automatic recovery report has not been initialized")
    report = _migrate_report(stored)
    pending = _ordered_unique(report.get("pending_candidate_trajectory_ids", []))
    if not pending:
        raise AutoRecoveryError("Automatic recovery has no pending batch to complete")
    accepted_before = set(_ordered_unique(report.get("accepted_trajectory_ids", [])))
    _merge_alignment(report, alignment)
    accepted_after = set(_ordered_unique(report.get("accepted_trajectory_ids", [])))
    attempted = _ordered_unique([*report.get("attempted_trajectory_ids", []), *pending])
    batches = report.get("batches", [])
    if not isinstance(batches, list):
        batches = []
    phase = report.get("phase")
    guidance_diagnostics = _guidance_collapse_diagnostics(alignment, pending)
    if phase == "normal_initial":
        report["latest_guidance_diagnostics"] = guidance_diagnostics
        report["source_guidance_collapse"] = bool(
            not accepted_after and guidance_diagnostics["detected"]
        )
    batches.append({
        "phase": phase,
        "candidate_trajectory_ids": pending,
        "accepted_trajectory_ids": [candidate for candidate in pending if candidate in accepted_after],
        "accepted_count_before": len(accepted_before),
        "accepted_count_after": len(accepted_after),
        "duration_seconds": max(float(duration_seconds), 0.0),
        "guidance_diagnostics": guidance_diagnostics,
        "completed_at": _utcnow(),
    })
    report["batches"] = batches
    report["attempted_trajectory_ids"] = attempted
    report["pending_candidate_trajectory_ids"] = []
    report["generated_video_trajectory_count"] = int(report.get("generated_video_trajectory_count", 0)) + len(pending)
    report["video_seconds"] = float(report.get("video_seconds", 0.0)) + max(float(duration_seconds), 0.0)
    report["state"] = "ready"
    report["phase"] = None
    report["rescue_round"] = None
    _update_source_attempt_snapshot(report, alignment)
    return _persist(task_dir, report)


def request_pilot_content_recovery(
    task_dir: Path,
    *,
    planning: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Schedule one trajectory supplement, then switch source on recurrence."""
    stored = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    if not stored:
        raise AutoRecoveryError("Pilot recovery requires an automatic recovery report")
    report = _migrate_report(stored, planning=planning)
    failures = report.get("pilot_content_failures", [])
    if not isinstance(failures, list):
        failures = []
    source_attempt = int(report.get("source_attempt", 0))
    source_failures = sum(
        1 for item in failures
        if isinstance(item, dict) and int(item.get("source_attempt", -1)) == source_attempt
    )
    failures.append({
        "source_attempt": source_attempt,
        "reason": reason,
        "recorded_at": _utcnow(),
    })
    report["pilot_content_failures"] = failures
    if source_failures >= 1:
        report["pilot_failed_source_attempts"] = sorted({
            *[
                int(value) for value in report.get("pilot_failed_source_attempts", [])
                if isinstance(value, (int, str)) and str(value).isdigit()
            ],
            source_attempt,
        })
    report["delivery_tier"] = None
    report["training_mode"] = None
    report["free_view_eligible"] = False
    report["recovery_exhausted"] = False
    report["terminal_reason"] = reason

    if _budget_exhausted(report):
        _finalize_delivery(report, exhausted=True, reason=reason)
        if report.get("state") == "completed":
            # Passing six trajectories is sufficient for training, but a
            # failed pilot proves the content is still not deliverable.
            _block_delivery(report, exhausted=True)
        return _persist(task_dir, report)

    if source_failures == 0:
        attempted = set(_ordered_unique(report.get("attempted_trajectory_ids", [])))
        remaining = [
            item for item in _normal_candidates(planning)
            if item not in attempted
        ][:NORMAL_REFILL_CANDIDATE_COUNT]
        per_source_budget = MAX_TRAJECTORIES_PER_SOURCE - len(attempted)
        remaining = remaining[:max(per_source_budget, 0)]
        if remaining:
            _persist(task_dir, report)
            return schedule_auto_recovery_batch(
                task_dir,
                phase="pilot_content_refill",
                candidate_ids=remaining,
            )

    if report.get("task_type") == "worldgen" and source_attempt < int(
        report["limits"]["maximum_source_regenerations"]
    ):
        report["state"] = "needs_regeneration"
        report["phase"] = "pilot_source_replacement"
        report["rescue_round"] = None
        report["terminal_reason"] = (
            "pilot_content_failed: trajectory supplementation did not repair the source; "
            "switching to the next preflight-ranked source."
        )
        return _persist(task_dir, report)

    _block_delivery(report, exhausted=True)
    report["terminal_reason"] = (
        "source_replacement_required: the 500-step pilot still failed after trajectory supplementation."
    )
    return _persist(task_dir, report)


def _alignment_quality_margin(alignment: dict[str, Any]) -> float:
    trajectories = alignment.get("trajectories", {})
    if not isinstance(trajectories, dict):
        return 0.0
    margins: list[float] = []
    for value in trajectories.values():
        if not isinstance(value, dict) or not value.get("accepted"):
            continue
        try:
            guidance = float(value.get("minimum_guidance_mask", value.get("guidance_mask", 0.8))) - 0.8
            relative = 0.5 - float(value.get("max_relative_deviation", value.get("depth_relative_error", 0.5)))
            keyframe = 0.1 - float(value.get("maximum_keyframe_depth_error_ratio", value.get("depth_error_ratio", 0.1)))
        except (TypeError, ValueError):
            continue
        margins.append(guidance + relative + keyframe)
    return sum(margins) / len(margins) if margins else 0.0


def _update_source_attempt_snapshot(report: dict[str, Any], alignment: dict[str, Any]) -> None:
    source_attempt = int(report.get("source_attempt", 0))
    existing = next(
        (
            item for item in report.get("source_attempts", [])
            if isinstance(item, dict) and int(item.get("source_attempt", -1)) == source_attempt
        ),
        {},
    )
    calculated_margin = _alignment_quality_margin(alignment)
    snapshot = {
        "source_attempt": source_attempt,
        "scene_relpath": report.get("active_scene_relpath", "scene"),
        "accepted_trajectory_ids": _ordered_unique(report.get("accepted_trajectory_ids", [])),
        "attempted_trajectory_ids": _ordered_unique(report.get("attempted_trajectory_ids", [])),
        "accepted_count": len(_ordered_unique(report.get("accepted_trajectory_ids", []))),
        "quality_margin": calculated_margin if calculated_margin else float(existing.get("quality_margin", 0.0)),
        "manual_source_retry": bool(report.get("manual_source_retry")),
        "batches": report.get("batches", []),
        "updated_at": _utcnow(),
    }
    attempts = [
        item for item in report.get("source_attempts", [])
        if isinstance(item, dict) and int(item.get("source_attempt", -1)) != source_attempt
    ]
    attempts.append(snapshot)
    attempts.sort(key=lambda item: int(item.get("source_attempt", 0)))
    report["source_attempts"] = attempts
    best = max(
        attempts,
        key=lambda item: (
            int(item.get("accepted_count", 0)),
            float(item.get("quality_margin", 0.0)),
            int(bool(item.get("manual_source_retry"))),
            (
                int(item.get("source_attempt", 0))
                if item.get("manual_source_retry")
                else -int(item.get("source_attempt", 0))
            ),
        ),
    )
    report["best_source_attempt"] = int(best.get("source_attempt", 0))


def begin_source_regeneration(task_dir: Path, *, seed: int, scene_relpath: str) -> dict[str, Any]:
    """Switch the recovery journal to one deterministic regenerated source."""
    stored = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    if not stored:
        raise AutoRecoveryError("Automatic recovery report has not been initialized")
    report = _migrate_report(stored)
    if report.get("state") != "needs_regeneration":
        raise AutoRecoveryError("Source regeneration was not requested by the recovery state machine")
    _update_source_attempt_snapshot(report, {})
    report["attempt_history"].append({
        "source_attempt": report.get("source_attempt", 0),
        "accepted_trajectory_ids": report.get("accepted_trajectory_ids", []),
        "batches": report.get("batches", []),
    })
    report["source_attempt"] = int(report.get("source_attempt", 0)) + 1
    report["source_seed"] = int(seed)
    report["active_scene_relpath"] = scene_relpath
    report["state"] = "regenerating_panorama"
    report["phase"] = "source_regeneration"
    report["rescue_round"] = None
    report["normal_candidate_trajectory_ids"] = []
    report["attempted_trajectory_ids"] = []
    report["accepted_trajectory_ids"] = []
    report["pending_candidate_trajectory_ids"] = []
    report["rendered_candidate_trajectory_ids"] = []
    report.pop("render_completed_at", None)
    report["batches"] = []
    report["source_guidance_collapse"] = False
    report["latest_guidance_diagnostics"] = None
    report["terminal_reason"] = None
    return _persist(task_dir, report)


def request_source_retry(
    task_dir: Path,
    *,
    task_type: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Durably queue a user-requested replacement source attempt.

    Automatic recovery is deliberately limited to one regenerated panorama.
    A user can make an explicit additional source attempt after inspecting the
    diagnosis.  It retains the prior journal in ``attempt_history`` and resets
    only the candidate accounting; no quality threshold is changed.
    """
    if task_type not in {"worldgen", "worldgen_from_pano"}:
        raise AutoRecoveryError("Only world generation tasks support source retry")
    stored = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    if not stored:
        raise AutoRecoveryError("Source retry requires an automatic recovery report")
    report = _migrate_report(stored, task_type=task_type)
    source_attempt = int(report.get("source_attempt", 0)) + 1
    attempt_history = report.get("attempt_history", [])
    if not isinstance(attempt_history, list):
        attempt_history = []
    attempt_history.append({
        "source_attempt": source_attempt - 1,
        "accepted_trajectory_ids": _ordered_unique(report.get("accepted_trajectory_ids", [])),
        "attempted_trajectory_ids": _ordered_unique(report.get("attempted_trajectory_ids", [])),
        "batches": report.get("batches", []),
        "manual_retry_requested_at": _utcnow(),
    })
    report["attempt_history"] = attempt_history
    report["source_attempt"] = source_attempt
    report["active_scene_relpath"] = f"recovery_attempts/attempt-{source_attempt:03d}/scene"
    report["state"] = "regenerating_panorama" if task_type == "worldgen" else "replacing_panorama_source"
    report["phase"] = "source_regeneration" if task_type == "worldgen" else "source_replacement"
    report["rescue_round"] = None
    report["normal_candidate_trajectory_ids"] = []
    report["attempted_trajectory_ids"] = []
    report["accepted_trajectory_ids"] = []
    report["pending_candidate_trajectory_ids"] = []
    report["rendered_candidate_trajectory_ids"] = []
    report.pop("render_completed_at", None)
    # Manual retry is explicit, but it still honors the task-wide 40-path /
    # 120-minute allowance accumulated by automatic recovery.
    report["batches"] = []
    report["source_guidance_collapse"] = False
    report["latest_guidance_diagnostics"] = None
    report["terminal_reason"] = None
    report["delivery_tier"] = None
    report["training_mode"] = None
    report["free_view_eligible"] = False
    report["quality_warning"] = None
    report["recovery_exhausted"] = False
    report.pop("best_scene_relpath", None)
    report["manual_source_retry"] = True
    if task_type == "worldgen":
        if seed is None:
            raise AutoRecoveryError("Generated source retry requires a deterministic seed")
        report["source_seed"] = int(seed)
    else:
        report.pop("source_seed", None)
    return _persist(task_dir, report)


def archive_downstream_checkpoints(task_dir: Path, *, source_attempt: int) -> None:
    """Keep failed downstream artifacts while forcing data/training to rerun.

    A manual source retry can follow an earlier successfully promoted source
    that later failed during GS preparation or training.  Its completed
    promotion journal belongs to that earlier attempt; retain it alongside
    the old output so the next attempt can create its own atomic journal.
    """
    archive_dir = task_dir / "recovery_attempts" / f"attempt-{source_attempt:03d}" / "previous_downstream"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        ".stage_gen_gs_data_done", ".stage_gs_pilot_done",
        ".stage_gs_training_done", ".stage_navigation_stress_done",
    ):
        source = task_dir / name
        if source.exists():
            destination = archive_dir / name
            if destination.exists():
                raise AutoRecoveryError(f"Downstream checkpoint archive already exists: {destination.name}")
            os.replace(source, destination)
    output_dir = task_dir / "output"
    if output_dir.exists():
        output_archive = archive_dir / "output"
        if output_archive.exists():
            raise AutoRecoveryError("Downstream output archive already exists")
        os.replace(output_dir, output_archive)
    pilot_dir = task_dir / "pilot_output"
    if pilot_dir.exists():
        pilot_archive = archive_dir / "pilot_output"
        if pilot_archive.exists():
            raise AutoRecoveryError("Pilot output archive already exists")
        os.replace(pilot_dir, pilot_archive)
    for name in (
        "post_panorama_preflight.json", "pilot_validation.json",
        "artifact_validation.json", "navigation_validation.json",
        "navigation_stress.json", "viewer_metadata_validation.json",
        "gaussian_repair.json", "gaussian_repair_request.json",
    ):
        source = task_dir / name
        if source.exists():
            destination = archive_dir / name
            if destination.exists():
                raise AutoRecoveryError(f"Downstream report archive already exists: {name}")
            os.replace(source, destination)
    promotion = task_dir / "source_promotion.json"
    if promotion.exists():
        promotion_archive = archive_dir / "source_promotion.json"
        if promotion_archive.exists():
            raise AutoRecoveryError("Source promotion archive already exists")
        os.replace(promotion, promotion_archive)


def restore_best_attempt_reports(task_dir: Path, *, source_attempt: int) -> None:
    """Restore the selected attempt's quality manifests, never its journal."""
    stored = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    if not stored:
        raise AutoRecoveryError("Automatic recovery report has not been initialized")
    report = _migrate_report(stored)
    reports_dir = task_dir / "recovery_attempts" / f"attempt-{int(source_attempt):03d}" / "reports"
    for name in (
        "quality_report.json",
        "alignment_quality.json",
        "trajectory_whitelist.json",
        "video_trajectory_whitelist.json",
    ):
        source = reports_dir / name
        if source.is_file():
            shutil.copy2(source, task_dir / name)
    # The accepted IDs in the v2 journal are authoritative even if an old
    # report archive predates the final rescue batch.
    _write_object_atomic(
        task_dir / "trajectory_whitelist.json",
        {"accepted_trajectory_ids": _ordered_unique(report.get("accepted_trajectory_ids", []))},
    )
    _persist(task_dir, report)


def set_active_scene(task_dir: Path, scene_relpath: str) -> dict[str, Any]:
    """Record the canonical scene after a successful atomic promotion."""
    stored = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    if not stored:
        raise AutoRecoveryError("Automatic recovery report has not been initialized")
    report = _migrate_report(stored)
    report["active_scene_relpath"] = scene_relpath
    return _persist(task_dir, report)


def promote_recovered_scene(
    task_dir: Path,
    *,
    source_attempt: int,
    regenerated_scene_dir: Path,
    canonical_scene_dir: Path,
) -> dict[str, Any]:
    """Atomically promote an accepted regenerated scene, with crash recovery.

    A directory replacement itself is atomic on the task volume, but two
    replacements are necessary: first archive the rejected canonical scene,
    then place the accepted candidate in its place.  ``source_promotion.json``
    makes the gap durable and this function is intentionally idempotent, so a
    pod restart resumes from the exact completed rename instead of re-running
    WorldStereo.
    """
    if source_attempt < 1:
        raise AutoRecoveryError("Recovered source attempt must be at least one")
    try:
        regenerated_relpath = regenerated_scene_dir.relative_to(task_dir)
        canonical_relpath = canonical_scene_dir.relative_to(task_dir)
    except ValueError as exc:
        raise AutoRecoveryError("Source promotion paths must be inside the task directory") from exc
    promotion_path = task_dir / "source_promotion.json"
    promotion = _read_object(promotion_path)
    archived_relpath = promotion.get("archive") if promotion else None
    if isinstance(archived_relpath, str) and archived_relpath:
        archive_scene_dir = task_dir / archived_relpath
    else:
        # A failed regenerated attempt remains in its own attempt directory.
        # Archive the current canonical source in the first free historical
        # slot rather than overwriting that diagnostic scene on a manual retry.
        archive_scene_dir = next(
            (
                task_dir / "recovery_attempts" / f"attempt-{attempt:03d}" / "scene"
                for attempt in range(source_attempt - 1, -1, -1)
                if not (task_dir / "recovery_attempts" / f"attempt-{attempt:03d}" / "scene").exists()
            ),
            None,
        )
        if archive_scene_dir is None:
            raise AutoRecoveryError("No immutable archive slot remains for the canonical scene")
    expected = {
        "source_attempt": source_attempt,
        "from": str(regenerated_relpath),
        "to": str(canonical_relpath),
        "archive": str(archive_scene_dir.relative_to(task_dir)),
    }
    if promotion and any(promotion.get(key) != value for key, value in expected.items()):
        raise AutoRecoveryError("A different source promotion is already pending")
    if promotion.get("state") == "completed":
        if canonical_scene_dir.is_dir() and not regenerated_scene_dir.exists():
            return promotion
        raise AutoRecoveryError("Completed source promotion has inconsistent scene directories")

    promotion = {**expected, "state": promotion.get("state", "prepared")}
    _write_object_atomic(promotion_path, promotion)

    if canonical_scene_dir.exists():
        if archive_scene_dir.exists():
            raise AutoRecoveryError("Cannot archive canonical scene because its immutable archive already exists")
        archive_scene_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(canonical_scene_dir, archive_scene_dir)
        promotion["state"] = "original_archived"
        _write_object_atomic(promotion_path, promotion)

    if regenerated_scene_dir.exists():
        if canonical_scene_dir.exists():
            raise AutoRecoveryError("Cannot promote regenerated scene over a canonical scene")
        canonical_scene_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(regenerated_scene_dir, canonical_scene_dir)
    elif not canonical_scene_dir.is_dir() or not archive_scene_dir.is_dir():
        raise AutoRecoveryError("Source promotion cannot find both the archived and regenerated scene")

    promotion["state"] = "completed"
    _write_object_atomic(promotion_path, promotion)
    return promotion


def active_scene_dir(task_dir: Path, fallback_scene_dir: Path) -> Path:
    report = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    relative = report.get("active_scene_relpath")
    if not isinstance(relative, str) or not relative:
        return fallback_scene_dir
    candidate = task_dir / relative
    return candidate if candidate.exists() else fallback_scene_dir


def _product_quality_fields(task_dir: Path, *, free_view_eligible: bool) -> dict[str, Any]:
    """Translate technical gate reports into one stable product-facing state."""
    from .gaussian_repair import assess_repair_eligibility

    input_preflight = _read_object(task_dir / "input_preflight.json")
    post_panorama = _read_object(task_dir / "post_panorama_preflight.json")
    navigation = _read_object(task_dir / "navigation_stress.json")
    repair = _read_object(task_dir / "gaussian_repair.json")
    repair_request = _read_object(task_dir / "gaussian_repair_request.json")
    eligibility = assess_repair_eligibility(task_dir)
    if free_view_eligible:
        state = "ready"
        message = "严格自由视角验收已全部通过，可以安全使用 3D 查看和 WASD。"
        next_action = "view_3d"
    elif input_preflight.get("state") == "failed" or post_panorama.get("state") == "failed":
        state = "source_replacement_required"
        message = "输入或生成后的全景不适合稳定自由视角重建，请更换为无人物、少反射且透视一致的场景源。"
        next_action = "replace_source"
    elif repair.get("state") in {"running", "repaired_pending_validation"} or repair_request.get("state") == "queued":
        state = "repairing"
        message = "正在归档原模型并修复稀疏巨大 Gaussian，完成后会自动重新执行全部导航压力测试。"
        next_action = "wait"
    elif navigation.get("state") == "failed" and (
        eligibility.get("eligible") or eligibility.get("requires_reanalysis")
    ):
        state = "repair_available"
        message = "模型主体质量合格，仅检测到少量巨大 Gaussian；可执行一次安全修复并重新验收。"
        next_action = "repair_model"
    elif navigation.get("state") == "failed" or repair.get("state") in {"failed", "failed_validation"}:
        state = "source_replacement_required"
        message = "该失败无法通过安全稀疏修复解决，请更换源后重新生成。原始产物和报告仍保留。"
        next_action = "replace_source"
    else:
        state = "processing"
        message = "严格自由视角闭环正在执行，所有质量门通过前不会开放 3D/WASD。"
        next_action = "wait"
    return {
        "product_state": state,
        "user_message": message,
        "next_action": next_action,
        "repair_eligibility": eligibility,
        "repair_report": repair or None,
        "repair_request": repair_request or None,
        "post_panorama_preflight": post_panorama or None,
    }


def auto_recovery_summary(task_dir: Path) -> dict[str, Any]:
    """Return the compact, safe-to-poll public recovery state."""
    stored = _read_object(task_dir / AUTO_RECOVERY_REPORT_NAME)
    source = _read_object(task_dir / "source_generation.json")
    if stored:
        # While recovery is running, surface the source currently being
        # generated. Once delivery is finalized, report the source backing the
        # selected best scene instead of the last (possibly rejected) attempt.
        source_attempt = int(stored.get("source_attempt", 0) or 0)
        if stored.get("state") == "completed":
            source_attempt = int(stored.get("best_source_attempt", source_attempt) or 0)
        if source_attempt > 0:
            attempt_source = _read_object(
                task_dir / "recovery_attempts" / f"attempt-{source_attempt:03d}" / "source_generation.json"
            )
            if attempt_source:
                source = attempt_source
    source_summary = {
        key: source.get(key)
        for key in (
            "mode", "state", "source_attempt", "seed", "artifact_path",
            "width", "height", "duration_seconds", "error_code",
            "error_summary", "updated_at", "completed_at", "original_prompt",
            "effective_prompt", "requested_source_width", "requested_source_height",
        )
        if key in source
    }
    training_manifest = _read_object(task_dir / "training_input_manifest.json")
    artifact_validation = _read_object(task_dir / "artifact_validation.json")
    navigation_validation = _read_object(task_dir / "navigation_validation.json")
    source_pool = _read_object(task_dir / "source_pool.json")
    input_preflight = _read_object(task_dir / "input_preflight.json")
    pilot_validation = _read_object(task_dir / "pilot_validation.json")
    navigation_stress = _read_object(task_dir / "navigation_stress.json")
    contract = _read_object(task_dir / "task_contract.json")
    if not stored:
        is_production_multiview = contract.get("input_lane") == "production_multiview"
        free_view_eligible = bool(
            artifact_validation.get("free_view_eligible") is True
            and navigation_stress.get("free_view_eligible") is True
            and (
                is_production_multiview
                or navigation_validation.get("free_view_eligible") is True
            )
        )
        if not free_view_eligible:
            return {
                "available": bool(input_preflight or source_pool or navigation_stress),
                "contract_version": contract.get("contract_version"),
                "input_lane": contract.get("input_lane"),
                "source_generation": source_summary or None,
                "source_pool": source_pool or None,
                "input_preflight": input_preflight or None,
                "pilot_validation": pilot_validation or None,
                "navigation_validation": navigation_validation or None,
                "navigation_stress": navigation_stress or None,
                "free_view_eligible": False,
                **_product_quality_fields(task_dir, free_view_eligible=False),
            }
        alignment = _read_object(task_dir / "alignment_quality.json")
        accepted = alignment.get("accepted_trajectory_ids", [])
        accepted_count = len(_ordered_unique(accepted if isinstance(accepted, list) else []))
        return {
            "available": True,
            "contract_version": contract.get("contract_version", AUTO_RECOVERY_VERSION),
            "input_lane": contract.get("input_lane", "generative_worldgen"),
            "state": "completed",
            "accepted_trajectories": accepted_count,
            "required_trajectories": MINIMUM_ACCEPTED_TRAJECTORIES,
            "attempted_trajectories": accepted_count,
            "pending_trajectories": 0,
            "delivery_tier": "strict_3d",
            "training_mode": "accepted_trajectories",
            "strict_accepted_count": accepted_count,
            "guaranteed_3d": False,
            "free_view_eligible": True,
            "navigation_validation": navigation_validation or None,
            "navigation_stress": navigation_stress,
            "pilot_validation": pilot_validation or None,
            "source_pool": source_pool or None,
            "input_preflight": input_preflight or None,
            "source_generation": source_summary or None,
            "anchor_view_count": training_manifest.get("anchor_view_count", 0),
            "training_input_manifest": training_manifest,
            "artifact_validation": artifact_validation,
            "recovery_exhausted": False,
            **_product_quality_fields(task_dir, free_view_eligible=True),
        }
    report = _migrate_report(stored)
    persisted_contract_version = int(
        contract.get("contract_version")
        or stored.get("contract_version")
        or stored.get("version")
        or 0
    )
    legacy_strict_delivery = bool(
        persisted_contract_version < AUTO_RECOVERY_VERSION
        and stored.get("state") == "completed"
        and stored.get("delivery_tier") == "strict_3d"
    )
    free_view_eligible = bool(
        report.get("delivery_tier") == "strict_3d"
        and artifact_validation.get("free_view_eligible") is True
        and navigation_validation.get("free_view_eligible") is True
        and (
            navigation_stress.get("free_view_eligible") is True
            or legacy_strict_delivery
        )
    )
    return {
        "available": True,
        "contract_version": persisted_contract_version or report.get("contract_version", AUTO_RECOVERY_VERSION),
        "input_lane": report.get("input_lane"),
        "success_policy": report.get("success_policy", "strict_free_view"),
        "state": report.get("state"),
        "phase": report.get("phase"),
        "rescue_round": report.get("rescue_round"),
        "source_attempt": report.get("source_attempt"),
        "accepted_trajectories": len(_ordered_unique(report.get("accepted_trajectory_ids", []))),
        "required_trajectories": report.get("minimum_accepted_trajectories"),
        "target_trajectories": report.get("target_accepted_trajectories", TARGET_ACCEPTED_TRAJECTORIES),
        "attempted_trajectories": len(_ordered_unique(report.get("attempted_trajectory_ids", []))),
        "pending_trajectories": len(_ordered_unique(report.get("pending_candidate_trajectory_ids", []))),
        "generated_video_trajectories": report.get("generated_video_trajectory_count", 0),
        "video_seconds": report.get("video_seconds", 0.0),
        "limits": report.get("limits", {}),
        "terminal_reason": report.get("terminal_reason"),
        "delivery_tier": report.get("delivery_tier"),
        "training_mode": report.get("training_mode"),
        "strict_accepted_count": report.get("strict_accepted_count", 0),
        "guaranteed_3d": False,
        "free_view_eligible": free_view_eligible,
        "legacy_strict_validation": legacy_strict_delivery,
        "navigation_validation": navigation_validation or None,
        "navigation_stress": navigation_stress or None,
        "pilot_validation": pilot_validation or None,
        "source_pool": source_pool or None,
        "input_preflight": input_preflight or None,
        "best_source_attempt": report.get("best_source_attempt"),
        "quality_warning": report.get("quality_warning"),
        "recovery_exhausted": report.get("recovery_exhausted", False),
        "transient_retry_count": report.get("transient_retry_count", 0),
        "source_guidance_collapse": report.get("source_guidance_collapse", False),
        "guidance_diagnostics": report.get("latest_guidance_diagnostics"),
        "updated_at": report.get("updated_at"),
        "source_generation": source_summary or None,
        "anchor_view_count": training_manifest.get("anchor_view_count", 0),
        "training_input_manifest": training_manifest or None,
        "artifact_validation": artifact_validation or None,
        "fallback_reason": report.get("quality_warning"),
        "global_budget": {
            "worldstereo_trajectories_used": report.get("generated_video_trajectory_count", 0),
            "worldstereo_seconds_used": report.get("video_seconds", 0.0),
            "elapsed_seconds": report.get("elapsed_seconds", 0.0),
            **report.get("limits", {}),
        },
        **_product_quality_fields(task_dir, free_view_eligible=free_view_eligible),
    }


def write_auto_video_trajectory_candidates(task_dir: Path, candidate_ids: Iterable[str]) -> Path:
    """Write the upstream WorldStereo whitelist for the currently pending batch."""
    manifest = task_dir / "video_trajectory_whitelist.json"
    _write_object_atomic(
        manifest,
        {"candidate_trajectory_ids": _ordered_unique(candidate_ids)},
    )
    return manifest


def _rotation_angle_degrees(first: list[list[float]], current: list[list[float]]) -> float:
    try:
        relative = [
            [sum(float(first[k][row]) * float(current[k][column]) for k in range(3)) for column in range(3)]
            for row in range(3)
        ]
    except (IndexError, TypeError, ValueError) as exc:
        raise AutoRecoveryError("Regular trajectory contains an invalid camera rotation") from exc
    cosine = (relative[0][0] + relative[1][1] + relative[2][2] - 1.0) / 2.0
    cosine = min(1.0, max(-1.0, cosine))
    return math.degrees(math.acos(cosine))


def _resample_safe_prefix(values: list[Any], last_index: int) -> list[Any]:
    prefix = values[:last_index + 1]
    if not prefix:
        raise AutoRecoveryError("Regular trajectory has no safe prefix")
    if len(prefix) == 1:
        return [prefix[0] for _ in range(21)]
    return [prefix[round(index * (len(prefix) - 1) / 20)] for index in range(21)]


def _pose_is_finite_and_orthonormal(pose: object) -> bool:
    if not isinstance(pose, list) or len(pose) < 4:
        return False
    try:
        matrix = [[float(pose[row][column]) for column in range(4)] for row in range(4)]
    except (IndexError, TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for row in matrix for value in row):
        return False
    rotation = [row[:3] for row in matrix[:3]]
    for row in rotation:
        if abs(sum(value * value for value in row) - 1.0) > 1e-3:
            return False
    for first in range(3):
        for second in range(first + 1, 3):
            if abs(sum(rotation[first][axis] * rotation[second][axis] for axis in range(3))) > 1e-3:
                return False
    return True


def _planned_regular_ids(planning: dict[str, Any] | None) -> list[str]:
    if not isinstance(planning, dict):
        return []
    trajectories = planning.get("trajectories", [])
    if not isinstance(trajectories, list):
        return []
    return _ordered_unique(
        item.get("id")
        for item in trajectories
        if isinstance(item, dict)
        and item.get("kind") == "regular"
        and isinstance(item.get("id"), str)
        and not str(item.get("id")).startswith("view_recovery_")
        and item.get("valid", True) is not False
        and item.get("collision_valid", True) is not False
        and item.get("status") not in {"failed", "rejected", "invalid"}
    )


def _source_regular_cameras(
    scene_dir: Path,
    planning: dict[str, Any] | None,
) -> list[tuple[str, Path]]:
    render_dir = scene_dir / "render_results"
    source_ids = _planned_regular_ids(planning)
    # Backward-compatible fallback for reports produced before trajectory kind
    # was persisted. It is intentionally unrestricted by traj number.
    if not source_ids and planning is None:
        source_ids = _ordered_unique(
            "/".join(path.relative_to(render_dir).parts[:2])
            for path in render_dir.glob("view*/**/camera.json")
            if not path.relative_to(render_dir).parts[0].startswith("view_recovery_")
        )
    sources: list[tuple[str, Path]] = []
    for source_id in source_ids:
        parts = Path(source_id).parts
        if len(parts) != 2 or parts[0].startswith("view_recovery_"):
            continue
        camera_path = render_dir / parts[0] / parts[1] / "camera.json"
        start_frame = render_dir / parts[0] / "start_frame.png"
        camera = _read_object(camera_path)
        extrinsic = camera.get("extrinsic")
        if (
            not camera_path.is_file()
            or not start_frame.is_file()
            or not isinstance(extrinsic, list)
            or len(extrinsic) < 2
            or not all(_pose_is_finite_and_orthonormal(pose) for pose in extrinsic)
        ):
            continue
        sources.append((source_id, camera_path))
    return sources


def _existing_rescue_candidates(
    scene_dir: Path,
    *,
    round_number: int,
    allowed_source_ids: set[str],
) -> list[dict[str, Any]]:
    render_dir = scene_dir / "render_results"
    candidates: list[dict[str, Any]] = []
    for camera_path in sorted(render_dir.glob(f"view_recovery_r{round_number}_*/traj0/camera.json")):
        camera = _read_object(camera_path)
        source_id = camera.get("recovery_source_trajectory")
        extrinsic = camera.get("extrinsic")
        start_frame = camera_path.parent.parent / "start_frame.png"
        if (
            not isinstance(source_id, str)
            or source_id not in allowed_source_ids
            or not isinstance(extrinsic, list)
            or len(extrinsic) != 21
            or not all(_pose_is_finite_and_orthonormal(pose) for pose in extrinsic)
            or not start_frame.is_file()
        ):
            continue
        relative = camera_path.relative_to(render_dir)
        candidates.append({
            "id": "/".join(relative.parts[:2]),
            "kind": "regular",
            "source_trajectory_id": source_id,
        })
    return candidates


def create_conservative_rescue_trajectories(
    scene_dir: Path,
    *,
    round_number: int,
    planning: dict[str, Any] | None = None,
    target_count: int | None = None,
) -> list[dict[str, Any]]:
    """Create every available reusable collision-safe camera prefix.

    Prefixing a regular route preserves its verified camera path.  Repeating
    selected prefix poses to 21 frames is safer than extrapolating translation
    or inventing a NavMesh route after the original planner has failed. Fewer
    than the requested number is a normal content outcome, never an exception.
    """
    target_rotation = RESCUE_TARGET_ROTATIONS.get(round_number)
    if target_rotation is None:
        raise AutoRecoveryError(f"Unknown conservative rescue round: {round_number}")
    requested = MAX_RESCUE_CANDIDATE_COUNT if target_count is None else max(int(target_count), 0)
    render_dir = scene_dir / "render_results"
    sources = _source_regular_cameras(scene_dir, planning)
    allowed_source_ids = {source_id for source_id, _ in sources}
    candidates = _existing_rescue_candidates(
        scene_dir,
        round_number=round_number,
        allowed_source_ids=allowed_source_ids,
    )
    used_sources = {item["source_trajectory_id"] for item in candidates}
    # Existing partial paths belong to a previously persisted atomic batch.
    # Always return them, even when the newly calculated adaptive target is
    # smaller, so a retry cannot orphan paid work or renumber output paths.
    required_total = max(requested, len(candidates))
    occupied_indices = {
        int(path.parent.parent.name.rsplit("_", 1)[-1])
        for path in render_dir.glob(f"view_recovery_r{round_number}_*/traj0/camera.json")
        if path.parent.parent.name.rsplit("_", 1)[-1].isdigit()
    }
    for source_id, source_path in sources:
        if len(candidates) >= required_total:
            break
        if source_id in used_sources:
            continue
        source = _read_object(source_path)
        extrinsic = source.get("extrinsic")
        intrinsic = source.get("intrinsic")
        if not isinstance(extrinsic, list) or len(extrinsic) < 2 or not isinstance(intrinsic, list):
            continue
        try:
            final_rotation = _rotation_angle_degrees(extrinsic[0], extrinsic[-1])
        except AutoRecoveryError:
            continue
        if final_rotation < 8.0:
            continue
        effective_rotation = min(target_rotation, final_rotation)
        safe_last_index = 0
        for index, pose in enumerate(extrinsic):
            if _rotation_angle_degrees(extrinsic[0], pose) <= effective_rotation + 1e-6:
                safe_last_index = index
            else:
                break
        if safe_last_index == 0:
            continue
        index = 0
        while index in occupied_indices:
            index += 1
        occupied_indices.add(index)
        view_id = f"view_recovery_r{round_number}_{index}"
        trajectory_id = f"{view_id}/traj0"
        output_dir = render_dir / view_id / "traj0"
        output_path = output_dir / "camera.json"
        source_view = source_id.split("/", 1)[0]
        source_frame = render_dir / source_view / "start_frame.png"
        if not source_frame.is_file():
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        output_intrinsic = _resample_safe_prefix(intrinsic, min(safe_last_index, len(intrinsic) - 1))
        camera = {
            **source,
            "extrinsic": _resample_safe_prefix(extrinsic, safe_last_index),
            "intrinsic": output_intrinsic,
            "type": "conservative-recovery",
            "recovery_source_trajectory": source_id,
            "recovery_round": round_number,
            "recovery_target_rotation_degrees": target_rotation,
            "recovery_effective_rotation_degrees": effective_rotation,
        }
        _write_object_atomic(output_path, camera)
        shutil.copy2(source_frame, output_dir.parent / "start_frame.png")
        candidates.append({
            "id": trajectory_id,
            "kind": "regular",
            "source_trajectory_id": source_id,
        })
        used_sources.add(source_id)
    candidates.sort(key=lambda item: item["id"])
    return candidates
