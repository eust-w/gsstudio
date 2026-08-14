"""Bounded, auditable repair for sparse giant-Gaussian navigation failures.

The repair is deliberately narrow: it can remove at most one small set of
Gaussians that the production gsplat renderer proved to be giant in otherwise
valid navigation views.  Original PLY, SPZ, checkpoints, and metadata are
archived before an atomic replacement, and the full navigation test runs again
before delivery can become eligible.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from .navigation_stress import NavigationStressError, run_navigation_stress
from .preflight import CONTRACT_VERSION, write_json_atomic


REPAIR_REPORT_NAME = "gaussian_repair.json"
REPAIR_REQUEST_NAME = "gaussian_repair_request.json"
MAXIMUM_REPAIR_ATTEMPTS = 1


class GaussianRepairError(RuntimeError):
    """A model cannot be repaired without weakening artifact guarantees."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assess_repair_eligibility(task_dir: Path) -> dict[str, Any]:
    """Return a public-safe decision without mutating task artifacts."""
    stress = _read_object(task_dir / "navigation_stress.json")
    analysis = stress.get("repair_analysis")
    prior = _read_object(task_dir / REPAIR_REPORT_NAME)
    attempts = int(prior.get("attempt", 0) or 0)
    reasons = stress.get("failure_reasons", [])
    giant_only = bool(reasons) and all(
        isinstance(reason, str)
        and reason.startswith("view_")
        and reason.endswith("_giant_gaussian")
        for reason in reasons
    )
    if stress.get("free_view_eligible") is True:
        return {
            "eligible": False,
            "requires_reanalysis": False,
            "reason": "navigation_already_passed",
            "attempts_used": attempts,
            "maximum_attempts": MAXIMUM_REPAIR_ATTEMPTS,
        }
    if attempts >= MAXIMUM_REPAIR_ATTEMPTS:
        return {
            "eligible": False,
            "requires_reanalysis": False,
            "reason": "repair_attempt_exhausted",
            "attempts_used": attempts,
            "maximum_attempts": MAXIMUM_REPAIR_ATTEMPTS,
        }
    if not isinstance(analysis, dict):
        # Reports produced before offender IDs were persisted can be safely
        # queued for one deterministic re-analysis, but only for giant-only
        # failures. No artifact is changed until the new report is eligible.
        return {
            "eligible": False,
            "requires_reanalysis": giant_only,
            "reason": "renderer_reanalysis_required" if giant_only else "failure_not_sparse_giant_only",
            "attempts_used": attempts,
            "maximum_attempts": MAXIMUM_REPAIR_ATTEMPTS,
        }
    eligible = bool(analysis.get("eligible") is True)
    return {
        "eligible": eligible,
        "requires_reanalysis": False,
        "reason": "sparse_giant_gaussians" if eligible else "repair_safety_limits_not_met",
        "offending_gaussian_count": int(analysis.get("offending_gaussian_count", 0) or 0),
        "offending_gaussian_fraction": float(analysis.get("offending_gaussian_fraction", 0.0) or 0.0),
        "maximum_repair_gaussians": int(analysis.get("maximum_repair_gaussians", 0) or 0),
        "maximum_repair_fraction": float(analysis.get("maximum_repair_fraction", 0.0) or 0.0),
        "attempts_used": attempts,
        "maximum_attempts": MAXIMUM_REPAIR_ATTEMPTS,
    }


def request_gaussian_repair(task_dir: Path, *, requested_by: str = "user") -> dict[str, Any]:
    assessment = assess_repair_eligibility(task_dir)
    if not (assessment.get("eligible") or assessment.get("requires_reanalysis")):
        raise GaussianRepairError(str(assessment.get("reason") or "model_repair_not_available"))
    request = {
        "version": CONTRACT_VERSION,
        "state": "queued",
        "requested_by": requested_by,
        "requested_at": _utcnow(),
        "eligibility_at_request": assessment,
    }
    write_json_atomic(task_dir / REPAIR_REQUEST_NAME, request)
    return request


def _final_gaussian_artifacts(task_dir: Path) -> tuple[Path, Path | None, list[Path], Path | None]:
    output = task_dir / "output"
    spz_files = sorted(output.rglob("point_cloud_*.spz"))
    spz_path = spz_files[-1] if spz_files else None
    if spz_path is not None:
        ply_path = spz_path.with_suffix(".ply")
        if not ply_path.is_file():
            raise GaussianRepairError("repair_artifact_mismatch: matching Gaussian PLY is missing")
    else:
        ply_candidates = sorted(
            path for path in output.rglob("*.ply")
            if "gaussian" in path.name.lower() or "point_cloud" in path.name.lower()
        )
        if not ply_candidates:
            raise GaussianRepairError("repair_artifact_mismatch: Gaussian PLY is missing")
        ply_path = ply_candidates[-1]

    step_match = re.search(r"point_cloud_(\d+)$", ply_path.stem)
    checkpoints: list[Path] = []
    if step_match:
        checkpoints = sorted((output / "ckpts").glob(f"ckpt_{step_match.group(1)}_rank*.pt"))
    metadata = ply_path.parent / "position_meta_info.json"
    return ply_path, spz_path, checkpoints, metadata if metadata.is_file() else None


def _write_pruned_ply(source: Path, destination: Path, remove_ids: list[int]) -> tuple[Any, int]:
    import numpy as np
    from plyfile import PlyData

    ply = PlyData.read(str(source))
    vertex = ply["vertex"].data
    if not remove_ids or min(remove_ids) < 0 or max(remove_ids) >= len(vertex):
        raise GaussianRepairError("repair_report_indices_do_not_match_ply")
    keep = np.ones(len(vertex), dtype=bool)
    keep[np.asarray(remove_ids, dtype=np.int64)] = False
    ply["vertex"].data = vertex[keep]
    destination.parent.mkdir(parents=True, exist_ok=True)
    ply.write(str(destination))
    return ply["vertex"].data, int((~keep).sum())


def _verify_spz_ply_order(spz_path: Path, ply_vertex: Any) -> None:
    import numpy as np
    import spz

    cloud = spz.load_spz(str(spz_path), spz.UnpackOptions())
    spz_positions = np.asarray(cloud.positions, dtype=np.float64).reshape(-1, 3)
    ply_positions = np.column_stack([
        ply_vertex[name] for name in ("x", "y", "z")
    ]).astype(np.float64)
    if len(spz_positions) != len(ply_positions):
        raise GaussianRepairError("repair_artifact_mismatch: SPZ and PLY counts disagree")
    maximum_error = float(np.max(np.abs(spz_positions - ply_positions))) if len(ply_positions) else 0.0
    if not maximum_error < 0.01:
        raise GaussianRepairError("repair_artifact_mismatch: SPZ and PLY ordering is not provable")


def _write_spz_from_ply(source_ply: Path, destination: Path) -> None:
    import spz

    unpack_options = spz.UnpackOptions()
    unpack_options.to_coord = spz.CoordinateSystem.RUB
    cloud = spz.load_splat_from_ply(str(source_ply), unpack_options)
    cloud.antialiased = True
    pack_options = spz.PackOptions()
    if hasattr(pack_options, "version"):
        pack_options.version = 3
    pack_options.from_coord = spz.CoordinateSystem.RDF
    destination.parent.mkdir(parents=True, exist_ok=True)
    spz.save_spz(cloud, pack_options, str(destination))


def _checkpoint_target_rows(ply_vertex: Any, remove_ids: list[int]) -> tuple[Any, set[bytes]]:
    import numpy as np

    required = ("x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity")
    names = set(ply_vertex.dtype.names or ())
    if not set(required).issubset(names):
        raise GaussianRepairError("repair_artifact_mismatch: PLY lacks checkpoint matching fields")
    rows = np.column_stack([ply_vertex[name] for name in required]).astype(np.float32, copy=False)
    targets = rows[np.asarray(remove_ids, dtype=np.int64)]
    keys = {row.tobytes() for row in targets}
    if len(keys) != len(remove_ids):
        raise GaussianRepairError("repair_artifact_mismatch: offender rows are not uniquely identifiable")
    return rows, keys


def _write_pruned_checkpoints(
    checkpoints: list[Path],
    destination_root: Path,
    target_keys: set[bytes],
) -> tuple[list[tuple[Path, Path]], int]:
    import numpy as np
    import torch

    outputs: list[tuple[Path, Path]] = []
    matched_keys: set[bytes] = set()
    removed_total = 0
    for checkpoint in checkpoints:
        value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        splats = value.get("splats") if isinstance(value, dict) else None
        if not isinstance(splats, dict) or not all(name in splats for name in ("means", "scales", "opacities")):
            raise GaussianRepairError(f"repair_checkpoint_invalid: {checkpoint.name}")
        count = int(splats["means"].shape[0])
        rows = torch.cat([
            splats["means"].detach().float(),
            splats["scales"].detach().float(),
            splats["opacities"].detach().float().reshape(-1, 1),
        ], dim=1).cpu().numpy().astype(np.float32, copy=False)
        remove = np.zeros(count, dtype=bool)
        for index, row in enumerate(rows):
            key = row.tobytes()
            if key in target_keys:
                remove[index] = True
                matched_keys.add(key)
        keep_tensor = torch.from_numpy(~remove)
        for name, tensor in list(splats.items()):
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1 and int(tensor.shape[0]) == count:
                splats[name] = tensor[keep_tensor]
        removed_total += int(remove.sum())
        destination = destination_root / checkpoint.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, destination)
        outputs.append((destination, checkpoint))
    if matched_keys != target_keys:
        raise GaussianRepairError(
            f"repair_checkpoint_mismatch: matched {len(matched_keys)} of {len(target_keys)} offenders"
        )
    return outputs, removed_total


def _write_updated_metadata(source: Path, destination: Path, ply_vertex: Any) -> None:
    import numpy as np

    metadata = _read_object(source)
    positions = np.column_stack([ply_vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    for axis, index in zip(("x", "y", "z"), range(3)):
        metadata[f"{axis}_min"] = float(lower[index])
        metadata[f"{axis}_max"] = float(upper[index])
    air_wall = metadata.get("air_wall")
    if isinstance(air_wall, dict) and isinstance(air_wall.get("bbox"), list) and len(air_wall["bbox"]) >= 6:
        air_wall["bbox"][:6] = [
            float(lower[0]), float(upper[0]),
            float(lower[1]), float(upper[1]),
            float(lower[2]), float(upper[2]),
        ]
    write_json_atomic(destination, metadata)


def repair_sparse_giant_gaussians(task_dir: Path) -> dict[str, Any]:
    """Repair one synchronized artifact set after a proven sparse failure."""
    assessment = assess_repair_eligibility(task_dir)
    if assessment.get("eligible") is not True:
        raise GaussianRepairError(str(assessment.get("reason") or "model_repair_not_available"))
    stress = _read_object(task_dir / "navigation_stress.json")
    analysis = stress["repair_analysis"]
    remove_ids = sorted({int(value) for value in analysis.get("offending_gaussian_ids", [])})
    if len(remove_ids) != int(analysis.get("offending_gaussian_count", -1)):
        raise GaussianRepairError("repair_report_is_incomplete")

    ply_path, spz_path, checkpoints, metadata_path = _final_gaussian_artifacts(task_dir)
    task_type = "worldgen" if checkpoints and spz_path else "worldrecon"
    if task_type == "worldgen" and (spz_path is None or not checkpoints):
        raise GaussianRepairError("repair_artifact_mismatch: synchronized Worldgen artifacts are incomplete")

    attempt = int(_read_object(task_dir / REPAIR_REPORT_NAME).get("attempt", 0) or 0) + 1
    if attempt > MAXIMUM_REPAIR_ATTEMPTS:
        raise GaussianRepairError("repair_attempt_exhausted")
    archive = task_dir / "repair_attempts" / f"attempt-{attempt:03d}" / "original"
    build = Path(tempfile.mkdtemp(prefix=".gaussian-repair-", dir=task_dir))
    report = {
        "version": CONTRACT_VERSION,
        "state": "running",
        "attempt": attempt,
        "automatic": True,
        "started_at": _utcnow(),
        "source_gaussian_count": int(stress.get("gaussian_count", 0) or 0),
        "requested_removal_count": len(remove_ids),
        "requested_removal_fraction": len(remove_ids) / max(int(stress.get("gaussian_count", 0) or 0), 1),
        "archive_path": str(archive.parent.relative_to(task_dir)),
        "original_artifacts_preserved": False,
        "full_navigation_revalidation_required": True,
    }
    write_json_atomic(task_dir / REPAIR_REPORT_NAME, report)

    replacements: list[tuple[Path, Path]] = []
    originals = [ply_path, *checkpoints]
    if spz_path:
        originals.append(spz_path)
    if metadata_path:
        originals.append(metadata_path)
    try:
        from plyfile import PlyData
        original_ply = PlyData.read(str(ply_path))["vertex"].data
        if spz_path:
            _verify_spz_ply_order(spz_path, original_ply)
        _, target_keys = _checkpoint_target_rows(original_ply, remove_ids) if checkpoints else (None, set())

        pruned_ply = build / ply_path.name
        pruned_vertex, removed_ply = _write_pruned_ply(ply_path, pruned_ply, remove_ids)
        replacements.append((pruned_ply, ply_path))
        if spz_path:
            pruned_spz = build / spz_path.name
            _write_spz_from_ply(pruned_ply, pruned_spz)
            replacements.append((pruned_spz, spz_path))
        removed_checkpoints = 0
        if checkpoints:
            checkpoint_replacements, removed_checkpoints = _write_pruned_checkpoints(
                checkpoints, build / "ckpts", target_keys,
            )
            replacements.extend(checkpoint_replacements)
        if metadata_path:
            updated_metadata = build / metadata_path.name
            _write_updated_metadata(metadata_path, updated_metadata, pruned_vertex)
            replacements.append((updated_metadata, metadata_path))

        archive.mkdir(parents=True, exist_ok=False)
        for source in originals:
            destination = archive / source.relative_to(task_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        report["original_artifacts_preserved"] = True

        committed: list[Path] = []
        try:
            for source, destination in replacements:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
                committed.append(destination)
        except Exception:
            for destination in committed:
                archived = archive / destination.relative_to(task_dir)
                temporary = destination.with_name(f".{destination.name}.restore")
                shutil.copy2(archived, temporary)
                os.replace(temporary, destination)
            raise

        report.update({
            "state": "repaired_pending_validation",
            "removed_gaussian_count": removed_ply,
            "removed_checkpoint_rows": removed_checkpoints,
            "result_gaussian_count": len(pruned_vertex),
            "artifact_hashes": {
                str(destination.relative_to(task_dir)): _sha256(destination)
                for _, destination in replacements
            },
            "repaired_at": _utcnow(),
        })
        write_json_atomic(task_dir / REPAIR_REPORT_NAME, report)
        return report
    except Exception as exc:
        report.update({
            "state": "failed",
            "failure_reason": str(exc),
            "failed_at": _utcnow(),
        })
        write_json_atomic(task_dir / REPAIR_REPORT_NAME, report)
        if isinstance(exc, GaussianRepairError):
            raise
        raise GaussianRepairError(f"gaussian_repair_failed: {exc}") from exc
    finally:
        shutil.rmtree(build, ignore_errors=True)


def run_navigation_with_repair(task_dir: Path, *, task_type: str) -> dict[str, Any]:
    """Run stress, perform at most one bounded repair, then run it in full again."""
    try:
        return run_navigation_stress(task_dir, task_type=task_type)
    except NavigationStressError as initial_error:
        assessment = assess_repair_eligibility(task_dir)
        if assessment.get("eligible") is not True:
            raise
        repair = repair_sparse_giant_gaussians(task_dir)
        try:
            result = run_navigation_stress(task_dir, task_type=task_type)
        except Exception as exc:
            repair.update({
                "state": "failed_validation",
                "failure_reason": str(exc),
                "validated_at": _utcnow(),
            })
            write_json_atomic(task_dir / REPAIR_REPORT_NAME, repair)
            raise NavigationStressError(
                f"navigation_stress_failed_after_repair: {exc}"
            ) from initial_error
        repair.update({
            "state": "passed",
            "navigation_stress_passed": True,
            "validated_at": _utcnow(),
        })
        write_json_atomic(task_dir / REPAIR_REPORT_NAME, repair)
        request_path = task_dir / REPAIR_REQUEST_NAME
        if request_path.is_file():
            request = _read_object(request_path)
            request.update({"state": "completed", "completed_at": _utcnow()})
            write_json_atomic(request_path, request)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--task-type", choices=("worldgen", "worldgen_from_pano", "worldrecon"), required=True)
    args = parser.parse_args()
    run_navigation_with_repair(Path(args.task_dir), task_type=args.task_type)


if __name__ == "__main__":
    main()
