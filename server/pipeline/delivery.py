"""Training-input accounting and the completed-3DGS delivery gate."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
import tempfile
from typing import Any


MINIMUM_GAUSSIANS = 50_000


class DeliveryValidationError(RuntimeError):
    """A world-generation task does not yet have a usable 3DGS delivery."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _training_mode(tier: str) -> str:
    return {
        "strict_3d": "accepted_trajectories",
        "degraded_3d": "accepted_plus_anchors",
        "anchored_3d": "panorama_anchored",
    }.get(tier, "panorama_anchored")


def build_training_input_manifest(task_dir: Path, scene_dir: Path, delivery_tier: str) -> dict[str, Any]:
    alignment = _read_object(task_dir / "alignment_quality.json")
    accepted = sorted({str(item) for item in alignment.get("accepted_trajectory_ids", [])})
    trajectories = alignment.get("trajectories", {})
    excluded = sorted(
        str(trajectory_id) for trajectory_id, value in trajectories.items()
        if not isinstance(value, dict) or not value.get("accepted")
    ) if isinstance(trajectories, dict) else []
    anchor_files: list[str] = []
    gs_images = scene_dir / "gs_data" / "images"
    gs_depths = scene_dir / "gs_data" / "depths"
    if gs_images.is_dir() and gs_depths.is_dir():
        for image in sorted(gs_images.glob("*.png")):
            if image.name.startswith(("panorama_", "polar_")) and (gs_depths / image.name).is_file():
                anchor_files.append(str(image.relative_to(scene_dir)))
    else:
        for bank in ("pano_bank", "polar_bank"):
            images = scene_dir / "render_results" / bank / "images"
            depths = scene_dir / "render_results" / bank / "depths"
            if not images.is_dir() or not depths.is_dir():
                continue
            for image in sorted(images.glob("*.png")):
                if (depths / image.name).is_file():
                    anchor_files.append(str(image.relative_to(scene_dir)))
    point_clouds = [
        path for path in (
            scene_dir / "render_results" / "global_pcd.ply",
            scene_dir / "render_results" / "sky_pcd.ply",
            scene_dir / "render_results" / "generation_bank_worldstereo-memory" / "global_pcd.ply",
            scene_dir / "render_results" / "generation_bank_worldstereo-memory" / "aligned_pcd.ply",
            scene_dir / "render_results" / "generation_bank_worldstereo-memory" / "sky_pcd.ply",
        ) if path.is_file()
    ]
    manifest = {
        "version": 1,
        "delivery_tier": delivery_tier,
        "training_mode": _training_mode(delivery_tier),
        "accepted_trajectory_ids": accepted,
        "excluded_trajectory_ids": excluded,
        "anchor_view_count": len(anchor_files),
        "anchor_views": anchor_files,
        "point_clouds": [str(path.relative_to(scene_dir)) for path in point_clouds],
        "rejected_trajectories_consumed": False,
    }
    _write_atomic(task_dir / "training_input_manifest.json", manifest)
    return manifest


def _find_viewer_metadata(task_dir: Path) -> Path | None:
    for path in task_dir.rglob("position_meta_info.json"):
        value = _read_object(path)
        if all(key in value for key in ("center_point", "facing_direction", "up_direction")):
            return path
    return None


def validate_completed_delivery(task_dir: Path, *, task_type: str) -> dict[str, Any]:
    if task_type not in {"worldgen", "worldgen_from_pano"}:
        return {"required": False, "valid": True}
    spz_files = sorted((task_dir / "output").rglob("*.spz"))
    ply_files = sorted((task_dir / "output").rglob("*.ply"))
    checkpoints = sorted((task_dir / "output" / "ckpts").glob("ckpt_*.pt"))
    if not spz_files:
        raise DeliveryValidationError("Completed world generation requires an SPZ artifact")
    if not ply_files:
        raise DeliveryValidationError("Completed world generation requires a trained Gaussian PLY")
    if not checkpoints:
        raise DeliveryValidationError("Completed world generation requires a training checkpoint")
    spz_step = re.search(r"point_cloud_(\d+)$", spz_files[-1].stem)
    checkpoint_steps = {
        match.group(1)
        for checkpoint in checkpoints
        if (match := re.search(r"ckpt_(\d+)_rank\d+$", checkpoint.stem))
    }
    if spz_step is None or spz_step.group(1) not in checkpoint_steps:
        raise DeliveryValidationError(
            "Completed world generation requires SPZ and checkpoint from the same training step"
        )
    metadata = _find_viewer_metadata(task_dir)
    if metadata is None:
        raise DeliveryValidationError("Completed world generation requires parseable viewer metadata")

    try:
        import numpy as np
        import spz
        options = spz.UnpackOptions()
        cloud = spz.load_spz(str(spz_files[-1]), options)
        point_count = int(cloud.num_points)
        if point_count < MINIMUM_GAUSSIANS:
            raise DeliveryValidationError(
                f"SPZ contains {point_count} Gaussians; at least {MINIMUM_GAUSSIANS} are required"
            )
        arrays = (cloud.positions, cloud.colors, cloud.scales, cloud.rotations)
        if not all(np.isfinite(values).all() for values in arrays):
            raise DeliveryValidationError("SPZ contains NaN or infinite Gaussian attributes")
        # SPZ stores opacity as a quantized probability, while its Python
        # decoder exposes opacity logits. Exact encoded endpoints therefore
        # legitimately decode to -Inf/+Inf (opacity 0/1). Validate the actual
        # opacity domain and reject NaN instead of rejecting valid endpoints.
        alpha_logits = np.asarray(cloud.alphas, dtype=np.float64)
        if np.isnan(alpha_logits).any():
            raise DeliveryValidationError("SPZ contains NaN opacity values")
        opacities = np.empty_like(alpha_logits)
        positive = alpha_logits >= 0
        opacities[positive] = 1.0 / (1.0 + np.exp(-alpha_logits[positive]))
        negative_exp = np.exp(alpha_logits[~positive])
        opacities[~positive] = negative_exp / (1.0 + negative_exp)
        if not np.isfinite(opacities).all() or ((opacities < 0) | (opacities > 1)).any():
            raise DeliveryValidationError("SPZ contains invalid opacity values")
        opacity_logit_endpoint_count = int(np.isinf(alpha_logits).sum())
        positions = np.asarray(cloud.positions, dtype=np.float64).reshape(-1, 3)
        extent = positions.max(axis=0) - positions.min(axis=0)
        if not np.isfinite(extent).all() or float(np.linalg.norm(extent)) <= 1e-6:
            raise DeliveryValidationError("SPZ has a degenerate spatial bounding box")
    except DeliveryValidationError:
        raise
    except Exception as exc:
        raise DeliveryValidationError(f"SPZ could not be decoded: {exc}") from exc

    result = {
        "required": True,
        "valid": True,
        "spz_path": str(spz_files[-1].relative_to(task_dir)),
        "ply_path": str(ply_files[-1].relative_to(task_dir)),
        "checkpoint_path": str(checkpoints[-1].relative_to(task_dir)),
        "metadata_path": str(metadata.relative_to(task_dir)),
        "gaussian_count": point_count,
        "finite_attributes": True,
        "finite_opacities": True,
        "opacity_logit_endpoint_count": opacity_logit_endpoint_count,
        "non_degenerate_bounds": True,
    }
    _write_atomic(task_dir / "artifact_validation.json", result)
    return result


def prepare_safe_training_retry(task_dir: Path, scene_dir: Path, *, maximum_points: int = 1_000_000) -> dict[str, Any]:
    """Deterministically reduce initialization memory before anchored-safe training."""
    points_path = scene_dir / "gs_data" / "points.ply"
    if not points_path.is_file():
        raise DeliveryValidationError("Anchored-safe retry requires gs_data/points.ply")
    try:
        import numpy as np
        import trimesh
        point_cloud = trimesh.load(str(points_path), process=False)
        points = np.asarray(point_cloud.vertices)
        colors = np.asarray(point_cloud.colors)
        original_count = int(len(points))
        if original_count > maximum_points:
            generator = np.random.default_rng(1024)
            selected = np.sort(generator.choice(original_count, maximum_points, replace=False))
            reduced = trimesh.PointCloud(vertices=points[selected], colors=colors[selected])
            temporary = points_path.with_suffix(".safe.tmp.ply")
            reduced.export(str(temporary))
            archive = points_path.with_name("points.full.ply")
            if not archive.exists():
                os.replace(points_path, archive)
            else:
                points_path.unlink()
            os.replace(temporary, points_path)
            final_count = maximum_points
        else:
            final_count = original_count
    except DeliveryValidationError:
        raise
    except Exception as exc:
        raise DeliveryValidationError(f"Could not prepare deterministic safe-training point cloud: {exc}") from exc
    output_dir = task_dir / "output"
    if output_dir.exists() and any(output_dir.iterdir()):
        attempt = 1
        while (task_dir / f"output_failed_attempt_{attempt}").exists():
            attempt += 1
        os.replace(output_dir, task_dir / f"output_failed_attempt_{attempt}")
    report = {
        "state": "prepared",
        "seed": 1024,
        "original_point_count": original_count,
        "training_point_count": final_count,
        "means_frozen": True,
        "densification_disabled": True,
        "maximum_steps": 2000,
    }
    _write_atomic(task_dir / "safe_training_retry.json", report)
    return report
