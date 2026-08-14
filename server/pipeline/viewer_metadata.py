"""Derive navigable viewer metadata and collision hulls for WorldMirror output."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .preflight import CONTRACT_VERSION, write_json_atomic


class ViewerMetadataError(RuntimeError):
    """World reconstruction did not produce stable navigation bounds."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _camera_matrices(payload: Any) -> list[Any]:
    def unwrap(values: Any) -> list[Any]:
        if not isinstance(values, list):
            return []
        return [
            item.get("matrix", item.get("extrinsic", item.get("pose")))
            if isinstance(item, dict) else item
            for item in values
        ]

    if isinstance(payload, dict):
        for key in ("extrinsics", "camera_extrinsics", "poses"):
            if isinstance(payload.get(key), list):
                return unwrap(payload[key])
        cameras = payload.get("cameras")
        if isinstance(cameras, list):
            return [item.get("extrinsic", item.get("pose")) for item in cameras if isinstance(item, dict)]
        if isinstance(cameras, dict):
            return [item.get("extrinsic", item.get("pose")) for item in cameras.values() if isinstance(item, dict)]
    if isinstance(payload, list):
        return unwrap(payload)
    return []


def _find_camera_params(output_dir: Path) -> Path | None:
    candidates = [output_dir / "camera_params.json", *output_dir.rglob("camera_params.json")]
    return next((path for path in candidates if path.is_file()), None)


def _find_points(output_dir: Path) -> Path | None:
    preferred = [
        output_dir / "points.ply",
        output_dir / "points" / "points.ply",
        output_dir / "point_cloud.ply",
    ]
    return next((path for path in [*preferred, *output_dir.rglob("*points*.ply")] if path.is_file()), None)


def derive_worldrecon_viewer_metadata(task_dir: Path) -> dict[str, Any]:
    import numpy as np
    import trimesh

    output_dir = task_dir / "output"
    camera_path = _find_camera_params(output_dir)
    if camera_path is None:
        raise ViewerMetadataError("navigation_boundary_failed: camera_params.json is missing")
    matrices = _camera_matrices(_read_json(camera_path))
    centers: list[Any] = []
    directions: list[Any] = []
    up_vectors: list[Any] = []
    for raw in matrices:
        try:
            matrix = np.asarray(raw, dtype=np.float64)
            if matrix.shape == (3, 4):
                matrix = np.vstack([matrix, [0.0, 0.0, 0.0, 1.0]])
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                continue
            rotation = matrix[:3, :3]
            translation = matrix[:3, 3]
            # HY-World camera_params stores world-to-camera extrinsics.
            center = -rotation.T @ translation
            centers.append(center)
            directions.append(rotation.T @ np.asarray([0.0, 0.0, 1.0]))
            up_vectors.append(rotation.T @ np.asarray([0.0, -1.0, 0.0]))
        except (TypeError, ValueError):
            continue
    if len(centers) < 12:
        raise ViewerMetadataError(
            f"navigation_boundary_failed: {len(centers)} valid camera centers; at least 12 are required"
        )
    centers_array = np.asarray(centers)
    if not np.isfinite(centers_array).all():
        raise ViewerMetadataError("navigation_boundary_failed: camera centers are non-finite")
    up = np.median(np.asarray(up_vectors), axis=0)
    up_norm = float(np.linalg.norm(up))
    if not math.isfinite(up_norm) or up_norm <= 1e-8:
        up = np.asarray([0.0, 1.0, 0.0])
    else:
        up /= up_norm
    center = np.median(centers_array, axis=0)
    offsets = centers_array - center
    horizontal = offsets - np.outer(offsets @ up, up)
    _, singular, vh = np.linalg.svd(horizontal, full_matrices=False)
    axis_width = vh[0]
    axis_width -= up * float(np.dot(axis_width, up))
    axis_width /= np.linalg.norm(axis_width)
    axis_height = np.cross(up, axis_width)
    projected_width = horizontal @ axis_width
    projected_height = horizontal @ axis_height
    half_width = float(max(abs(np.percentile(projected_width, 5)), abs(np.percentile(projected_width, 95))))
    half_height = float(max(abs(np.percentile(projected_height, 5)), abs(np.percentile(projected_height, 95))))
    baseline = float(np.median(np.linalg.norm(np.diff(centers_array, axis=0), axis=1)))
    human_scale = max(baseline * 4.0, 0.1)
    if min(half_width, half_height) <= 0.1 * human_scale or len(singular) < 2 or singular[1] <= 1e-6:
        raise ViewerMetadataError("navigation_boundary_failed: camera OBB has fewer than two effective axes")

    fallback = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(fallback, up))) > 0.9:
        fallback = np.asarray([0.0, 1.0, 0.0])
    reference_right = np.cross(fallback, up)
    reference_right /= np.linalg.norm(reference_right)
    reference_forward = np.cross(up, reference_right)
    angle = math.atan2(float(np.dot(axis_width, reference_forward)), float(np.dot(axis_width, reference_right)))
    mean_direction = np.mean(np.asarray(directions), axis=0)
    if np.linalg.norm(mean_direction) <= 1e-8:
        mean_direction = axis_height
    mean_direction /= np.linalg.norm(mean_direction)

    points_path = _find_points(output_dir)
    if points_path is None:
        raise ViewerMetadataError("navigation_boundary_failed: points PLY is missing")
    cloud = trimesh.load(str(points_path), process=False)
    points = np.asarray(cloud.vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1000 or not np.isfinite(points).all():
        raise ViewerMetadataError("navigation_boundary_failed: point cloud is finite but too small or malformed")
    # Depth/normal products back the point cloud. Their presence is recorded;
    # the robust hull itself is derived from camera-adjacent point percentiles
    # so isolated depth outliers cannot create a huge air wall.
    lower = np.percentile(points, 1, axis=0)
    upper = np.percentile(points, 99, axis=0)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper - lower <= 1e-6):
        raise ViewerMetadataError("navigation_boundary_failed: point-cloud bounds are degenerate")
    camera_lower = centers_array.min(axis=0) - human_scale * 0.5
    camera_upper = centers_array.max(axis=0) + human_scale * 0.5
    air_lower = np.maximum(lower, camera_lower)
    air_upper = np.minimum(upper, camera_upper)
    if np.any(air_upper <= air_lower):
        raise ViewerMetadataError("navigation_boundary_failed: camera and point-cloud bounds do not overlap")

    hull_points = points[
        np.all((points >= lower) & (points <= upper), axis=1)
    ]
    if len(hull_points) > 200_000:
        generator = np.random.default_rng(1024)
        hull_points = hull_points[generator.choice(len(hull_points), 200_000, replace=False)]
    try:
        hull = trimesh.convex.convex_hull(hull_points)
    except Exception as exc:
        raise ViewerMetadataError(f"navigation_boundary_failed: collision hull failed: {exc}") from exc
    if len(hull.faces) < 4 or not np.isfinite(hull.vertices).all():
        raise ViewerMetadataError("navigation_boundary_failed: collision hull is invalid")
    ply_dir = output_dir / "ply"
    ply_dir.mkdir(parents=True, exist_ok=True)
    collision_path = ply_dir / "fuse_simplified.ply"
    temporary_collision = ply_dir / ".fuse_simplified.tmp.ply"
    hull.export(str(temporary_collision))
    temporary_collision.replace(collision_path)

    metadata = {
        "version": CONTRACT_VERSION,
        "center_point": [float(value) for value in center],
        "facing_direction": [float(value) for value in center + mean_direction],
        "up_direction": [float(value) for value in up],
        "human_scale": human_scale,
        "kwargs": {
            "camera_obb": [
                float(center[0]), float(center[1]), float(center[2]),
                half_width, half_height, angle,
            ],
        },
        "air_wall": {
            "bbox": [
                float(air_lower[0]), float(air_upper[0]),
                float(air_lower[1]), float(air_upper[1]),
                float(air_lower[2]), float(air_upper[2]),
            ],
        },
        "provenance": {
            "camera_params": str(camera_path.relative_to(task_dir)),
            "points": str(points_path.relative_to(task_dir)),
            "depth_products": len(list(output_dir.rglob("*depth*"))),
            "normal_products": len(list(output_dir.rglob("*normal*"))),
            "valid_camera_count": len(centers),
        },
    }
    metadata_path = ply_dir / "position_meta_info.json"
    write_json_atomic(metadata_path, metadata)
    report = {
        "version": CONTRACT_VERSION,
        "state": "passed",
        "metadata_path": str(metadata_path.relative_to(task_dir)),
        "collision_mesh_path": str(collision_path.relative_to(task_dir)),
        "valid_camera_count": len(centers),
        "camera_obb_half_width": half_width,
        "camera_obb_half_height": half_height,
        "human_scale": human_scale,
        "air_wall_bbox": metadata["air_wall"]["bbox"],
    }
    write_json_atomic(task_dir / "viewer_metadata_validation.json", report)
    return report
