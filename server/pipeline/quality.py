"""Quality policy, validation, and reports for real HY-World tasks.

The upstream trajectory planner can log a NavMesh failure while exiting with a
zero status.  This module makes that state explicit and keeps the downstream
3DGS stage from consuming trajectories whose alignment diagnostics are poor.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


QUALITY_PRESETS = {
    "high": {
        "worldstereo_model_type": "worldstereo-memory",
        "minimum_trajectories": 8,
        "minimum_accepted_trajectories": 6,
    },
    "fast": {
        "worldstereo_model_type": "worldstereo-memory-dmd",
        "minimum_trajectories": 0,
        "minimum_accepted_trajectories": 0,
    },
}
# High-quality auto recovery starts with the smallest useful WorldStereo
# batch.  Regular views have the most reliable guidance for indoor scenes;
# non-regular WorldNav paths remain available later but no longer consume a
# full-model pass before low acceptance can be detected.
HIGH_VIDEO_CANDIDATE_LIMIT = 8
HIGH_VIDEO_CANDIDATE_BUDGET = 32
HIGH_VIDEO_RECOVERY_BATCH_SIZE = 8
VIDEO_RECOVERY_REPORT_NAME = "video_recovery.json"
VIDEO_CANDIDATE_KIND_ORDER = ("regular", "surround", "exploration", "reconstruction")

PLANNING_ERROR_PATTERNS = (
    re.compile(r"navmesh\s+error", re.IGNORECASE),
    re.compile(r"failed\s+to\s+load\s+segmentation\s+labels", re.IGNORECASE),
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SUMMARY_RE = re.compile(
    r"(?P<trajectory>(?:view|target|reconstruct|wonder)[^\s]*/traj\d+)\s+"
    r"\((?P<frame_count>\d+)\s+frames\):.*?"
    r"max_abs_dev=\[(?P<abs_min>-?[\d.eE+]+),(?P<abs_max>-?[\d.eE+]+)\],\s*"
    r"max_rel_dev=\[(?P<rel_min>-?[\d.eE+]+)%,(?P<rel_max>-?[\d.eE+]+)%\]"
)
FRAME_RE = re.compile(
    r"\[Frame\s+(?P<trajectory>(?:view|target|reconstruct|wonder)[^\s]*/traj\d+)/"
    r"(?P<frame>[^\]]+)\]\s+\[(?P<status>Success|Warning|Failed)\]"
    r"(?:\s+Guidance mask ratio:\s*(?P<guidance>[\d.]+)"
    r"(?:,\s*depth percentile error ratio:\s*(?P<depth_error>[\d.]+))?)?",
    re.IGNORECASE,
)


class QualityGateError(RuntimeError):
    """A stage completed at process level but failed the output-quality gate."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


def normalize_worldgen_config(
    config: dict[str, Any] | None,
    *,
    form_prompt: str | None,
) -> dict[str, Any]:
    """Normalize persisted world-generation settings at the API boundary."""
    normalized = dict(config or {})
    quality_preset = normalized.get("quality_preset", "high")
    if quality_preset == "fast":
        raise ValueError("Fast is not available for World Generation; use the strict high-quality preset")
    if quality_preset not in QUALITY_PRESETS:
        choices = ", ".join(sorted(QUALITY_PRESETS))
        raise ValueError(f"quality_preset must be one of: {choices}")
    normalized["quality_preset"] = quality_preset
    normalized["worldstereo_model_type"] = QUALITY_PRESETS[quality_preset]["worldstereo_model_type"]

    recovery_policy = normalized.get("recovery_policy", "auto")
    if recovery_policy not in {"auto", "off"}:
        raise ValueError("recovery_policy must be one of: auto, off")
    normalized["recovery_policy"] = recovery_policy

    if form_prompt is not None and form_prompt.strip():
        normalized["prompt"] = form_prompt.strip()
    elif "prompt" in normalized:
        if not isinstance(normalized["prompt"], str):
            raise ValueError("prompt must be a string")
        normalized["prompt"] = normalized["prompt"].strip()

    if "seed" in normalized and normalized["seed"] is not None:
        try:
            normalized["seed"] = int(normalized["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer") from exc

    return normalized


def _trajectory_kind(view_id: str) -> str:
    lowered = view_id.lower()
    if lowered.startswith("target"):
        return "surround"
    if lowered.startswith("wonder"):
        return "exploration"
    if lowered.startswith("reconstruct"):
        return "reconstruction"
    return "regular"


def _collision_drops(log_text: str) -> int:
    # WorldNav emits both explicit drop/reject messages and the terse
    # "Too many collisions … ignore" form. Count both so quality_report.json
    # accurately explains why a candidate path was not retained.
    return len(
        re.findall(
            r"(?:collision|collid).*?(?:drop|discard|reject|ignore)",
            log_text,
            re.IGNORECASE,
        )
    )


def _read_log(log_file: Path) -> str:
    return log_file.read_text(errors="replace") if log_file.exists() else ""


def _camera_overlap_estimate(camera_path: Path) -> dict[str, float] | None:
    """Estimate source-view overlap from the first and last planned poses."""
    try:
        payload = json.loads(camera_path.read_text())
        extrinsic = payload.get("extrinsic", [])
        if not isinstance(extrinsic, list) or len(extrinsic) < 2:
            return None
        first, last = extrinsic[0], extrinsic[-1]
        if not all(isinstance(matrix, list) and len(matrix) >= 3 for matrix in (first, last)):
            return None
        first_translation = [float(first[index][3]) for index in range(3)]
        last_translation = [float(last[index][3]) for index in range(3)]
        translation = math.sqrt(sum(
            (last_translation[index] - first_translation[index]) ** 2
            for index in range(3)
        ))
        # Use the full R0^T R1 trace so arbitrary planned start poses work.
        relative_trace = sum(
            sum(float(first[k][row]) * float(last[k][row]) for k in range(3))
            for row in range(3)
        )
        cosine = max(-1.0, min(1.0, (relative_trace - 1.0) / 2.0))
        rotation_degrees = math.degrees(math.acos(cosine))
        overlap = 1.0 / (1.0 + translation + rotation_degrees / 45.0)
        return {
            "camera_translation": translation,
            "camera_rotation_degrees": rotation_degrees,
            "estimated_overlap_score": overlap,
        }
    except (IndexError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _candidate_sort_key(item: dict[str, Any]) -> tuple[int, float, str]:
    value = item.get("estimated_overlap_score")
    try:
        overlap = float(value)
    except (TypeError, ValueError):
        return (1, 0.0, str(item.get("id", "")))
    if not math.isfinite(overlap):
        return (1, 0.0, str(item.get("id", "")))
    return (0, -overlap, str(item.get("id", "")))


def validate_trajectory_plan(
    scene_dir: Path,
    log_file: Path,
    *,
    quality_preset: str,
) -> dict[str, Any]:
    """Validate persisted NavMesh outputs after `traj_generate` exits.

    This intentionally happens after the upstream command exits because the
    upstream implementation catches NavMesh exceptions and returns zero.
    """
    log_text = _read_log(log_file)
    navmesh_dir = scene_dir / "navmesh"
    reconstruct_pairs = navmesh_dir / "reconstruct_pairs.json"
    segmented_targets_path = scene_dir / "camera_trajectory" / "target_camera.json"
    segmented_targets: list[Any] = []
    if segmented_targets_path.is_file():
        try:
            loaded_targets = json.loads(segmented_targets_path.read_text())
            if isinstance(loaded_targets, list):
                segmented_targets = loaded_targets
        except (json.JSONDecodeError, OSError):
            # A malformed non-empty segmentation checkpoint means upstream was
            # expected to construct reconstruction pairs but could not do so.
            segmented_targets = ["invalid-segmentation-checkpoint"]
    reconstruct_pairs_required = bool(segmented_targets)
    planning_warnings = []
    if not reconstruct_pairs.is_file() and not reconstruct_pairs_required:
        planning_warnings.append(
            "reconstruct_pairs.json was not produced because no segmented reconstruction targets were available"
        )
    cameras = sorted((scene_dir / "render_results").glob("*/traj*/camera.json"))
    trajectories = []
    for camera in cameras:
        relative = camera.relative_to(scene_dir / "render_results")
        view_id, traj_id = relative.parts[:2]
        trajectory = {
            "id": f"{view_id}/{traj_id}",
            "kind": _trajectory_kind(view_id),
            "camera": str(camera.relative_to(scene_dir)),
        }
        overlap = _camera_overlap_estimate(camera)
        if overlap:
            trajectory.update(overlap)
        trajectories.append(trajectory)

    by_kind: dict[str, int] = defaultdict(int)
    for trajectory in trajectories:
        by_kind[trajectory["kind"]] += 1
    planning_errors = [
        line.strip()
        for line in log_text.splitlines()
        if any(pattern.search(line) for pattern in PLANNING_ERROR_PATTERNS)
    ]
    report = {
        "quality_preset": quality_preset,
        "planning": {
            "navmesh_present": navmesh_dir.is_dir(),
            "reconstruct_pairs_present": reconstruct_pairs.is_file(),
            "reconstruct_pairs_required": reconstruct_pairs_required,
            "valid_trajectory_count": len(trajectories),
            "non_regular_trajectory_count": sum(
                count for kind, count in by_kind.items() if kind != "regular"
            ),
            "by_kind": dict(sorted(by_kind.items())),
            "trajectories": trajectories,
            "collision_dropped_count": _collision_drops(log_text),
            "log_errors": planning_errors,
            "warnings": planning_warnings,
        },
    }

    if planning_errors:
        raise QualityGateError(
            "Navmesh Error or trajectory-planning error was found in the stage log", report
        )
    if reconstruct_pairs_required and not reconstruct_pairs.is_file():
        raise QualityGateError("Missing required NavMesh output: reconstruct_pairs.json", report)
    if quality_preset == "high":
        if len(trajectories) < QUALITY_PRESETS["high"]["minimum_trajectories"]:
            raise QualityGateError("High quality requires at least 8 valid trajectories", report)
        if report["planning"]["non_regular_trajectory_count"] == 0:
            raise QualityGateError(
                "High quality requires a surround, exploration, or reconstruction trajectory", report
            )
    return report


def select_high_quality_video_candidates(planning: dict[str, Any]) -> list[dict[str, Any]]:
    """Select the initial eight full-WorldStereo candidates, regular first."""
    trajectories = sorted(planning.get("trajectories", []), key=_candidate_sort_key)
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        by_kind[trajectory["kind"]].append(trajectory)

    selected = list(by_kind["regular"])
    for kind in ("surround", "reconstruction", "exploration"):
        selected.extend(by_kind[kind])
    return selected[:HIGH_VIDEO_CANDIDATE_LIMIT]


def rank_high_quality_video_candidates(planning: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every valid path in deterministic high-quality priority order.

    The initial batch and replenishment both prefer regular routes, followed by surround,
    exploration, and reconstruction routes.  This makes indoor recovery
    favor the paths with the most stable camera guidance without pretending
    that a rejected route is acceptable.
    """
    trajectories = sorted(planning.get("trajectories", []), key=_candidate_sort_key)
    initial = select_high_quality_video_candidates(planning)
    selected_ids = {candidate["id"] for candidate in initial}
    ranked = list(initial)
    for kind in VIDEO_CANDIDATE_KIND_ORDER:
        ranked.extend(
            trajectory
            for trajectory in trajectories
            if trajectory["kind"] == kind and trajectory["id"] not in selected_ids
        )
    ranked_ids = {candidate["id"] for candidate in ranked}
    ranked.extend(
        trajectory
        for trajectory in trajectories
        if trajectory["id"] not in ranked_ids
    )
    return ranked


def write_video_trajectory_candidates(task_dir: Path, candidates: list[dict[str, Any]]) -> Path:
    """Persist the high-quality WorldStereo input set before video inference."""
    manifest_path = task_dir / "video_trajectory_whitelist.json"
    manifest_path.write_text(
        json.dumps(
            {"candidate_trajectory_ids": [candidate["id"] for candidate in candidates]},
            indent=2,
        )
        + "\n"
    )
    return manifest_path


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        parsed = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _ordered_intersection(ids: Iterable[str], allowed: set[str]) -> list[str]:
    seen: set[str] = set()
    selected: list[str] = []
    for trajectory_id in ids:
        if not isinstance(trajectory_id, str):
            continue
        if trajectory_id not in allowed or trajectory_id in seen:
            continue
        seen.add(trajectory_id)
        selected.append(trajectory_id)
    return selected


def _write_video_recovery_report(task_dir: Path, recovery: dict[str, Any]) -> Path:
    report_path = task_dir / VIDEO_RECOVERY_REPORT_NAME
    report_path.write_text(json.dumps(recovery, indent=2, sort_keys=True) + "\n")
    merge_quality_report(task_dir, "video_recovery", recovery)
    return report_path


def prepare_high_quality_video_recovery(task_dir: Path) -> dict[str, Any]:
    """Choose the next safe WorldStereo batch for a high-quality task.

    This function is intentionally restart-safe.  A pre-recovery task already
    has ``alignment_quality.json`` and an initial candidate manifest, so the
    generated trajectory IDs are inferred as attempted and the first retry
    starts with untested paths rather than regenerating the rejected batch.
    A pending batch is retained until the worker records its subprocess result,
    so an interrupted job reuses that exact set on retry.
    """
    quality_report = _read_json_object(task_dir / "quality_report.json")
    planning = quality_report.get("planning")
    if not isinstance(planning, dict) or not planning.get("trajectories"):
        raise QualityGateError("Cannot replenish video trajectories without a valid planning report")

    ranked = rank_high_quality_video_candidates(planning)
    candidate_budget = ranked[:HIGH_VIDEO_CANDIDATE_BUDGET]
    candidate_ids = [candidate["id"] for candidate in candidate_budget]
    candidate_by_id = {candidate["id"]: candidate for candidate in candidate_budget}
    allowed_ids = set(candidate_ids)
    initial_ids = candidate_ids[:HIGH_VIDEO_CANDIDATE_LIMIT]

    alignment = _read_json_object(task_dir / "alignment_quality.json")
    recovery = _read_json_object(task_dir / VIDEO_RECOVERY_REPORT_NAME)
    alignment_trajectories = alignment.get("trajectories", {})
    if not isinstance(alignment_trajectories, dict):
        alignment_trajectories = {}
    attempted_ids = _ordered_intersection(
        [
            *recovery.get("attempted_trajectory_ids", []),
            *alignment_trajectories.keys(),
        ],
        allowed_ids,
    )
    attempted_set = set(attempted_ids)
    accepted_ids = _ordered_intersection(
        alignment.get("accepted_trajectory_ids", []),
        allowed_ids,
    )
    pending_ids = _ordered_intersection(
        recovery.get("pending_candidate_trajectory_ids", []),
        allowed_ids,
    )
    minimum = QUALITY_PRESETS["high"]["minimum_accepted_trajectories"]

    if len(accepted_ids) >= minimum:
        next_ids: list[str] = []
        status = "accepted"
    elif pending_ids:
        next_ids = pending_ids
        status = "initial" if not attempted_ids else "recovering"
    else:
        remaining_initial = [trajectory_id for trajectory_id in initial_ids if trajectory_id not in attempted_set]
        if remaining_initial:
            next_ids = remaining_initial
            status = "initial" if not attempted_ids else "recovering"
        else:
            remaining_ids = [trajectory_id for trajectory_id in candidate_ids if trajectory_id not in attempted_set]
            next_ids = remaining_ids[:HIGH_VIDEO_RECOVERY_BATCH_SIZE]
            status = "recovering" if next_ids else "exhausted"

    remaining_ids = [
        trajectory_id
        for trajectory_id in candidate_ids
        if trajectory_id not in attempted_set and trajectory_id not in next_ids
    ]
    batches = recovery.get("batches", [])
    if not isinstance(batches, list):
        batches = []
    prepared = {
        "quality_preset": "high",
        "minimum_accepted_trajectories": minimum,
        "candidate_budget": len(candidate_ids),
        "candidate_budget_trajectory_ids": candidate_ids,
        "initial_candidate_trajectory_ids": initial_ids,
        "attempted_trajectory_ids": attempted_ids,
        "accepted_trajectory_ids": accepted_ids,
        "pending_candidate_trajectory_ids": next_ids,
        "remaining_trajectory_ids": remaining_ids,
        "next_candidate_trajectory_ids": next_ids,
        "status": status,
        "batches": batches,
    }
    _write_video_recovery_report(task_dir, prepared)
    if next_ids:
        write_video_trajectory_candidates(task_dir, [candidate_by_id[trajectory_id] for trajectory_id in next_ids])
    return prepared


def complete_high_quality_video_recovery(
    task_dir: Path,
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """Record a completed WorldStereo batch before deciding whether to refill."""
    recovery = _read_json_object(task_dir / VIDEO_RECOVERY_REPORT_NAME)
    candidate_ids = recovery.get("candidate_budget_trajectory_ids", [])
    if not isinstance(candidate_ids, list) or not candidate_ids:
        raise QualityGateError("Cannot complete trajectory recovery without a prepared candidate budget")
    allowed_ids = set(candidate_ids)
    pending_ids = _ordered_intersection(
        recovery.get("pending_candidate_trajectory_ids", []), allowed_ids
    )
    attempted_ids = _ordered_intersection(
        [*recovery.get("attempted_trajectory_ids", []), *pending_ids], allowed_ids
    )
    accepted_ids = _ordered_intersection(
        alignment.get("accepted_trajectory_ids", []), allowed_ids
    )
    minimum = QUALITY_PRESETS["high"]["minimum_accepted_trajectories"]
    remaining_ids = [trajectory_id for trajectory_id in candidate_ids if trajectory_id not in set(attempted_ids)]
    batches = recovery.get("batches", [])
    if not isinstance(batches, list):
        batches = []
    if pending_ids:
        batches.append(
            {
                "round": len(batches) + 1,
                "candidate_trajectory_ids": pending_ids,
                "accepted_trajectory_ids": [
                    trajectory_id for trajectory_id in pending_ids if trajectory_id in set(accepted_ids)
                ],
            }
        )
    completed = {
        **recovery,
        "attempted_trajectory_ids": attempted_ids,
        "accepted_trajectory_ids": accepted_ids,
        "pending_candidate_trajectory_ids": [],
        "remaining_trajectory_ids": remaining_ids,
        "next_candidate_trajectory_ids": [],
        "status": "accepted" if len(accepted_ids) >= minimum else "recovering" if remaining_ids else "exhausted",
        "batches": batches,
    }
    _write_video_recovery_report(task_dir, completed)
    return completed


def _as_ratio(value: str) -> float:
    return float(value) / 100.0


def _frame_key(frame: str) -> tuple[int, str]:
    numeric = re.search(r"\d+", frame)
    return (int(numeric.group()) if numeric else 0, frame)


def build_alignment_quality(
    log_text: str,
    *,
    quality_preset: str,
    expected_trajectory_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Parse WorldStereo alignment diagnostics into a durable JSON report."""
    cleaned = ANSI_ESCAPE.sub("", log_text)
    trajectories: dict[str, dict[str, Any]] = {}

    for match in SUMMARY_RE.finditer(cleaned):
        trajectory = trajectories.setdefault(match["trajectory"], {"frames": []})
        trajectory.update({
            "frame_count": int(match["frame_count"]),
            "max_absolute_deviation": float(match["abs_max"]),
            "max_relative_deviation": _as_ratio(match["rel_max"]),
        })

    for match in FRAME_RE.finditer(cleaned):
        trajectory = trajectories.setdefault(match["trajectory"], {"frames": []})
        trajectory["frames"].append({
            "id": match["frame"],
            "status": match["status"].lower(),
            "guidance_mask": float(match["guidance"]) if match["guidance"] else None,
            "depth_error_ratio": float(match["depth_error"]) if match["depth_error"] else None,
        })

    # A zero-exit upstream command can still omit an expected trajectory.  It
    # must be recorded as a rejection rather than silently disappearing from
    # the recovery accounting and causing the same path to be retried forever.
    for trajectory_id in expected_trajectory_ids or ():
        trajectories.setdefault(trajectory_id, {"frames": []})

    accepted_ids: list[str] = []
    for trajectory_id in sorted(trajectories):
        trajectory = trajectories[trajectory_id]
        frames = sorted(trajectory["frames"], key=lambda item: _frame_key(item["id"]))
        trajectory["frames"] = frames
        guidance_values = [frame["guidance_mask"] for frame in frames if frame["guidance_mask"] is not None]
        trajectory["minimum_guidance_mask"] = min(guidance_values) if guidance_values else None
        key_frames = frames[:1] + frames[-1:] if len(frames) > 1 else frames
        key_depth_errors = [frame["depth_error_ratio"] for frame in key_frames if frame["depth_error_ratio"] is not None]
        trajectory["maximum_keyframe_depth_error_ratio"] = max(key_depth_errors) if key_depth_errors else None
        trajectory.setdefault("max_absolute_deviation", None)
        trajectory.setdefault("max_relative_deviation", None)

        reasons: list[str] = []
        if quality_preset == "high":
            if not frames:
                reasons.append("missing_alignment_frames")
            if trajectory["minimum_guidance_mask"] is None or trajectory["minimum_guidance_mask"] < 0.80:
                reasons.append("guidance_mask_below_0.80")
            if trajectory["max_relative_deviation"] is not None and trajectory["max_relative_deviation"] > 0.50:
                reasons.append("depth_relative_deviation_above_0.50")
            if (
                trajectory["maximum_keyframe_depth_error_ratio"] is not None
                and trajectory["maximum_keyframe_depth_error_ratio"] > 0.10
            ):
                reasons.append("keyframe_depth_error_ratio_above_0.10")
        trajectory["rejection_reasons"] = reasons
        trajectory["accepted"] = not reasons
        if trajectory["accepted"]:
            accepted_ids.append(trajectory_id)

    return {
        "quality_preset": quality_preset,
        "thresholds": {
            "minimum_guidance_mask": 0.80,
            "maximum_relative_deviation": 0.50,
            "maximum_keyframe_depth_error_ratio": 0.10,
        },
        "trajectories": trajectories,
        "accepted_trajectory_ids": accepted_ids,
        "rejected_trajectory_ids": [
            trajectory_id for trajectory_id, trajectory in trajectories.items() if not trajectory["accepted"]
        ],
    }


def enforce_alignment_quality(
    report: dict[str, Any],
    *,
    quality_preset: str,
    candidate_budget: int | None = None,
) -> None:
    """Fail high-quality work before 3DGS when too few trajectories survive."""
    if quality_preset != "high":
        return
    minimum = QUALITY_PRESETS["high"]["minimum_accepted_trajectories"]
    accepted = report.get("accepted_trajectory_ids", [])
    if len(accepted) < minimum:
        exhausted_detail = (
            f" after exhausting {candidate_budget} planned candidates" if candidate_budget is not None else ""
        )
        raise QualityGateError(
            f"High quality requires at least {minimum} trajectories accepted by alignment quality gates; "
            f"only {len(accepted)} passed{exhausted_detail}"
        )


def write_alignment_quality(task_dir: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    """Persist detailed diagnostics and the compact whitelist for HY-World."""
    alignment_path = task_dir / "alignment_quality.json"
    manifest_path = task_dir / "trajectory_whitelist.json"
    alignment_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(
        json.dumps({"accepted_trajectory_ids": report["accepted_trajectory_ids"]}, indent=2) + "\n"
    )
    return alignment_path, manifest_path


def merge_quality_report(task_dir: Path, section: str, payload: dict[str, Any]) -> Path:
    """Merge stage-specific quality evidence into a single task report."""
    report_path = task_dir / "quality_report.json"
    current: dict[str, Any] = {}
    if report_path.exists():
        try:
            current = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            current = {}
    current[section] = payload
    report_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    return report_path


def postvalidate_worldgen_stage(
    stage_name: str,
    *,
    task_dir: Path,
    scene_dir: Path,
    log_file: Path,
    quality_preset: str,
    expected_trajectory_ids: Iterable[str] | None = None,
    enforce_alignment: bool = True,
) -> dict[str, Any] | None:
    """Run quality gates immediately after a successful upstream subprocess."""
    if stage_name == "traj_generate":
        try:
            planning = validate_trajectory_plan(
                scene_dir, log_file, quality_preset=quality_preset
            )
        except QualityGateError as exc:
            if exc.report is not None:
                merge_quality_report(task_dir, "planning", exc.report["planning"])
            raise
        planning_report = planning["planning"]
        if quality_preset == "high":
            candidates = select_high_quality_video_candidates(planning_report)
            planning_report["video_candidate_trajectory_ids"] = [
                candidate["id"] for candidate in candidates
            ]
            planning_report["video_candidate_by_kind"] = dict(
                sorted(
                    (
                        kind,
                        sum(candidate["kind"] == kind for candidate in candidates),
                    )
                    for kind in VIDEO_CANDIDATE_KIND_ORDER
                    if any(candidate["kind"] == kind for candidate in candidates)
                )
            )
            write_video_trajectory_candidates(task_dir, candidates)
        merge_quality_report(task_dir, "planning", planning_report)
        return planning

    if stage_name == "video_gen":
        alignment = build_alignment_quality(
            _read_log(log_file),
            quality_preset=quality_preset,
            expected_trajectory_ids=expected_trajectory_ids,
        )
        write_alignment_quality(task_dir, alignment)
        merge_quality_report(task_dir, "alignment", alignment)
        if enforce_alignment:
            enforce_alignment_quality(alignment, quality_preset=quality_preset)
        return alignment

    if stage_name == "gen_gs_data":
        gs_data = scene_dir / "gs_data"
        required_files = [gs_data / "cameras.json", gs_data / "points.ply"]
        missing = [str(path.relative_to(scene_dir)) for path in required_files if not path.is_file() or path.stat().st_size <= 0]
        image_count = len(list((gs_data / "images").glob("*.png")))
        depth_count = len(list((gs_data / "depths").glob("*.png")))
        report = {
            "required_files": [str(path.relative_to(scene_dir)) for path in required_files],
            "missing_files": missing,
            "image_count": image_count,
            "depth_count": depth_count,
        }
        merge_quality_report(task_dir, "gs_data", report)
        if missing or image_count == 0 or depth_count == 0:
            raise QualityGateError(
                "GS data preparation completed without usable cameras, points, RGB, and depth outputs",
                report={"gs_data": report},
            )
        try:
            cameras = json.loads((gs_data / "cameras.json").read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise QualityGateError(
                f"GS data camera manifest is invalid: {exc}",
                report={"gs_data": report},
            ) from exc
        if not isinstance(cameras, dict) or not cameras:
            raise QualityGateError(
                "GS data camera manifest is empty",
                report={"gs_data": report},
            )
        report["camera_count"] = len(cameras)
        merge_quality_report(task_dir, "gs_data", report)
        return report

    return None
