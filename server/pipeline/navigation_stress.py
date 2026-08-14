"""Deterministic off-screen navigation stress gate for free-view delivery."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .preflight import CONTRACT_VERSION, write_json_atomic
from .viewer_metadata import ViewerMetadataError, derive_worldrecon_viewer_metadata


NAVIGATION_SEED = 1024
NAVIGATION_STEPS = 1000
REPRESENTATIVE_VIEWS = 32
MICRO_TRANSLATION_PAIRS = 16
MAXIMUM_GIANT_RADIUS_RATIO = 0.50
MAXIMUM_TRANSPARENT_RATIO = 0.90
MAXIMUM_BLACK_WHITE_CLIP_RATIO = 0.95
MAXIMUM_MICRO_LPIPS = 0.15
MAXIMUM_MICRO_BRIGHTNESS_DELTA = 0.10
MAXIMUM_REPORTED_GIANT_IDS_PER_VIEW = 2048
MAXIMUM_AUTOREPAIR_GAUSSIANS = 500
MAXIMUM_AUTOREPAIR_FRACTION = 0.0005


class NavigationStressError(RuntimeError):
    """The model is structurally unsafe for free WASD navigation."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _metadata_path(task_dir: Path) -> Path | None:
    preferred = task_dir / "output" / "ply" / "position_meta_info.json"
    if preferred.is_file():
        return preferred
    return next((path for path in (task_dir / "output").rglob("position_meta_info.json") if path.is_file()), None)


def _gaussian_path(task_dir: Path) -> Path | None:
    spz = sorted((task_dir / "output").rglob("*.spz"))
    if spz:
        return spz[-1]
    candidates = [
        path for path in (task_dir / "output").rglob("*.ply")
        if "gaussian" in path.name.lower() or "point_cloud" in path.name.lower()
    ]
    return sorted(candidates)[-1] if candidates else None


def _sigmoid(values: Any) -> Any:
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _load_gaussians(path: Path) -> dict[str, Any]:
    import numpy as np

    if path.suffix.lower() == ".spz":
        import spz
        cloud = spz.load_spz(str(path), spz.UnpackOptions())
        positions = np.asarray(cloud.positions, dtype=np.float64).reshape(-1, 3)
        scales = np.asarray(cloud.scales, dtype=np.float64).reshape(-1, 3)
        opacities = _sigmoid(np.asarray(cloud.alphas, dtype=np.float64).reshape(-1))
        # SPZ exposes quaternions as x/y/z/w while gsplat consumes w/x/y/z.
        rotations = np.asarray(cloud.rotations, dtype=np.float64).reshape(-1, 4)[:, [3, 0, 1, 2]]
        colors = np.asarray(cloud.colors, dtype=np.float64)
        if colors.ndim == 1:
            if colors.size == positions.shape[0] * 3:
                colors = colors.reshape(positions.shape[0], 3)
            elif colors.size == positions.shape[0]:
                colors = np.repeat(colors[:, None], 3, axis=1)
            else:
                raise NavigationStressError(
                    "navigation_stress_failed: SPZ color count does not match Gaussian count"
                )
        elif colors.shape[0] != positions.shape[0] and colors.size % positions.shape[0] == 0:
            colors = colors.reshape(positions.shape[0], -1)
        if colors.shape[0] != positions.shape[0] or colors.shape[1] < 3:
            raise NavigationStressError(
                "navigation_stress_failed: SPZ colors do not have one RGB value per Gaussian"
            )
        # SPZ exposes degree-zero spherical-harmonic coefficients, matching
        # gsplat's f_dc fields rather than display-ready RGB.
        colors = 0.5 + 0.28209479177387814 * colors[:, :3]
    else:
        try:
            from plyfile import PlyData
            vertex = PlyData.read(str(path))["vertex"].data
            names = set(vertex.dtype.names or ())
            positions = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
            scale_names = ("scale_0", "scale_1", "scale_2")
            if not set(scale_names).issubset(names):
                raise NavigationStressError("navigation_stress_failed: Gaussian PLY has no scale properties")
            scales = np.column_stack([vertex[name] for name in scale_names]).astype(np.float64)
            rotation_names = ("rot_0", "rot_1", "rot_2", "rot_3")
            if not set(rotation_names).issubset(names):
                raise NavigationStressError("navigation_stress_failed: Gaussian PLY has no rotation properties")
            # HY-World/gsplat PLY checkpoints already store w/x/y/z.
            rotations = np.column_stack([vertex[name] for name in rotation_names]).astype(np.float64)
            opacity_name = "opacity" if "opacity" in names else "alpha" if "alpha" in names else None
            if opacity_name is None:
                raise NavigationStressError("navigation_stress_failed: Gaussian PLY has no opacity property")
            opacities = _sigmoid(np.asarray(vertex[opacity_name], dtype=np.float64))
            sh_colors = "f_dc_0" in names
            color_names = ("f_dc_0", "f_dc_1", "f_dc_2") if sh_colors else ("red", "green", "blue")
            if not set(color_names).issubset(names):
                raise NavigationStressError("navigation_stress_failed: Gaussian PLY has no color properties")
            colors = np.column_stack([vertex[name] for name in color_names]).astype(np.float64)
            if sh_colors:
                colors = 0.5 + 0.28209479177387814 * colors
            elif colors.size and colors.max() > 1.0:
                colors /= 255.0
        except NavigationStressError:
            raise
        except Exception as exc:
            raise NavigationStressError(f"navigation_stress_failed: Gaussian PLY decode failed: {exc}") from exc
    if not all(np.isfinite(array).all() for array in (positions, scales, rotations, opacities, colors)):
        raise NavigationStressError("navigation_stress_failed: Gaussian attributes are non-finite")
    if not (
        len(positions) == len(scales) == len(rotations) == len(opacities) == len(colors)
    ):
        raise NavigationStressError("navigation_stress_failed: Gaussian attribute counts disagree")
    rotation_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    if np.any(rotation_norms <= 1e-8):
        raise NavigationStressError("navigation_stress_failed: Gaussian rotations are degenerate")
    return {
        "positions": positions,
        "scales": scales,
        "rotations": rotations / rotation_norms,
        "opacities": opacities,
        "colors": np.clip(colors, 0.0, 1.0),
    }


def _viewer_basis(metadata: dict[str, Any]) -> tuple[Any, Any, Any]:
    import numpy as np
    up = np.asarray(metadata["up_direction"], dtype=np.float64)
    up /= np.linalg.norm(up)
    fallback = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(fallback, up))) > 0.9:
        fallback = np.asarray([0.0, 1.0, 0.0])
    right = np.cross(fallback, up)
    right /= np.linalg.norm(right)
    forward = np.cross(up, right)
    forward /= np.linalg.norm(forward)
    return right, up, forward


def _inside_bounds(point: Any, metadata: dict[str, Any]) -> bool:
    import numpy as np
    point = np.asarray(point, dtype=np.float64)
    bbox = metadata.get("air_wall", {}).get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 6:
        return False
    lower = np.asarray([bbox[0], bbox[2], bbox[4]], dtype=np.float64)
    upper = np.asarray([bbox[1], bbox[3], bbox[5]], dtype=np.float64)
    if np.any(point < np.minimum(lower, upper)) or np.any(point > np.maximum(lower, upper)):
        return False
    obb = metadata.get("kwargs", {}).get("camera_obb")
    if not isinstance(obb, list) or len(obb) < 6:
        return False
    center = np.asarray(obb[:3], dtype=np.float64)
    half_width, half_height, angle = [float(value) for value in obb[3:6]]
    right, up, forward = _viewer_basis(metadata)
    delta = point - center
    delta -= up * float(np.dot(delta, up))
    u = float(np.dot(delta, right))
    v = float(np.dot(delta, forward))
    local_width = u * math.cos(angle) + v * math.sin(angle)
    local_height = -u * math.sin(angle) + v * math.cos(angle)
    return abs(local_width) <= half_width and abs(local_height) <= half_height


def _stress_walk(metadata: dict[str, Any], collision_path: Path | None = None) -> dict[str, Any]:
    import numpy as np
    collision = None
    collision_mode = None
    if collision_path and collision_path.is_file():
        try:
            import trimesh
            collision = trimesh.load(str(collision_path), process=False)
            collision_mode = "closed_volume" if bool(collision.is_watertight) else "surface_raycast"
        except Exception as exc:
            raise NavigationStressError(f"navigation_stress_failed: collision mesh decode failed: {exc}") from exc
    generator = np.random.default_rng(NAVIGATION_SEED)
    position = np.asarray(metadata["center_point"], dtype=np.float64)
    right, up, forward = _viewer_basis(metadata)
    human_scale = float(metadata.get("human_scale", 1.0))
    accepted = 0
    rejected = 0
    finite = True
    height = float(np.dot(position, up))
    for _ in range(NAVIGATION_STEPS):
        direction = generator.normal(size=2)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        distance = generator.uniform(0.01, 0.12) * human_scale
        candidate = position + (direction[0] * right + direction[1] * forward) * distance
        if not np.isfinite(candidate).all():
            finite = False
            break
        collision_safe = True
        if collision is not None:
            try:
                if collision_mode == "closed_volume":
                    collision_safe = bool(collision.contains([candidate])[0])
                else:
                    displacement = candidate - position
                    move_distance = float(np.linalg.norm(displacement))
                    locations, _, _ = collision.ray.intersects_location(
                        [position], [displacement / max(move_distance, 1e-9)], multiple_hits=False,
                    )
                    collision_safe = not any(
                        float(np.linalg.norm(location - position)) <= move_distance + human_scale * 0.02
                        for location in locations
                    )
            except Exception as exc:
                raise NavigationStressError(f"navigation_stress_failed: collision query failed: {exc}") from exc
        if _inside_bounds(candidate, metadata) and collision_safe:
            position = candidate
            accepted += 1
        else:
            rejected += 1
        if abs(float(np.dot(position, up)) - height) > 1e-7:
            finite = False
            break
    return {
        "seed": NAVIGATION_SEED,
        "requested_steps": NAVIGATION_STEPS,
        "completed_steps": accepted + rejected,
        "accepted_steps": accepted,
        "boundary_rejected_steps": rejected,
        "all_camera_positions_finite": finite,
        "height_preserved": finite,
        "final_position_inside_bounds": _inside_bounds(position, metadata),
        "collision_mesh_checked": collision is not None,
        "collision_mode": collision_mode,
    }


def _camera_poses(metadata: dict[str, Any]) -> list[tuple[Any, Any]]:
    import numpy as np
    center = np.asarray(metadata["center_point"], dtype=np.float64)
    right, up, forward = _viewer_basis(metadata)
    obb = metadata["kwargs"]["camera_obb"]
    half_width, half_height, angle = [float(value) for value in obb[3:6]]
    width_axis = right * math.cos(angle) + forward * math.sin(angle)
    height_axis = -right * math.sin(angle) + forward * math.cos(angle)
    poses = []
    for index in range(REPRESENTATIVE_VIEWS):
        ring = 0.25 + 0.65 * ((index % 4) / 3.0)
        theta = 2.0 * math.pi * index / REPRESENTATIVE_VIEWS
        position = center + width_axis * (math.cos(theta) * half_width * ring) + height_axis * (math.sin(theta) * half_height * ring)
        target = center + up * (0.05 * float(metadata.get("human_scale", 1.0)))
        poses.append((position, target))
    return poses


def _prepare_raster_state(gaussian: dict[str, Any]) -> dict[str, Any]:
    """Move one validated Gaussian model to the production gsplat renderer."""
    import numpy as np
    try:
        import torch
        from gsplat.rendering import rasterization
    except Exception as exc:
        raise NavigationStressError(
            f"navigation_stress_failed: gsplat rasterizer unavailable: {exc}"
        ) from exc
    if not torch.cuda.is_available():
        raise NavigationStressError("navigation_stress_failed: CUDA gsplat rasterizer unavailable")
    device = torch.device("cuda")

    def tensor(name: str) -> Any:
        values = np.asarray(gaussian[name], dtype=np.float32)
        return torch.from_numpy(np.array(values, copy=True)).to(device=device)

    return {
        "means": tensor("positions"),
        "quats": tensor("rotations"),
        "scales": torch.exp(tensor("scales")),
        "opacities": tensor("opacities"),
        "colors": tensor("colors"),
        "rasterization": rasterization,
    }


def _giant_splat_mask(radii: Any, opacities: Any, *, width: int, height: int) -> Any:
    """Return the visible-splat mask that violates the production radius gate."""
    import numpy as np
    projected = np.asarray(radii, dtype=np.float64)
    alpha = np.asarray(opacities, dtype=np.float64).reshape(-1)
    if projected.ndim == 1:
        projected = projected[:, None]
    if len(projected) != len(alpha):
        raise NavigationStressError("navigation_stress_failed: gsplat radius/opacity counts disagree")
    maximum_radius = np.max(projected, axis=1) if len(projected) else np.asarray([])
    threshold = math.hypot(width, height) * MAXIMUM_GIANT_RADIUS_RATIO
    return (alpha > 0.05) & (maximum_radius > threshold)


def _count_giant_splats(radii: Any, opacities: Any, *, width: int, height: int) -> int:
    """Count only splats that gsplat actually projects into the rendered view."""
    import numpy as np
    return int(np.count_nonzero(_giant_splat_mask(
        radii, opacities, width=width, height=height,
    )))


def _giant_splat_ids(
    radii: Any,
    opacities: Any,
    gaussian_ids: Any,
    *,
    width: int,
    height: int,
) -> tuple[list[int], bool]:
    """Return stable model indices for a sparse, repairable giant-splat set."""
    import numpy as np

    ids = np.asarray(gaussian_ids, dtype=np.int64).reshape(-1)
    mask = _giant_splat_mask(radii, opacities, width=width, height=height)
    if len(ids) != len(mask):
        raise NavigationStressError("navigation_stress_failed: gsplat ID/radius counts disagree")
    unique = np.unique(ids[mask])
    truncated = len(unique) > MAXIMUM_REPORTED_GIANT_IDS_PER_VIEW
    return [int(value) for value in unique[:MAXIMUM_REPORTED_GIANT_IDS_PER_VIEW]], truncated


def _render_projection(
    raster_state: dict[str, Any],
    position: Any,
    target: Any,
    up: Any,
    *,
    width: int = 256,
    height: int = 160,
) -> tuple[Any, dict[str, Any]]:
    """Render with the same gsplat projection/rasterization used by training."""
    import numpy as np
    import torch

    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    focal = width / (2.0 * math.tan(math.radians(70.0) / 2.0))

    # gsplat uses OpenCV camera coordinates: +X right, +Y down, +Z forward.
    rotation = np.stack([right, -camera_up, forward]).astype(np.float32)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ np.asarray(position, dtype=np.float32)
    intrinsics = np.asarray([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    device = raster_state["means"].device
    with torch.inference_mode():
        rendered, rendered_alpha, metadata = raster_state["rasterization"](
            raster_state["means"],
            raster_state["quats"],
            raster_state["scales"],
            raster_state["opacities"],
            raster_state["colors"],
            torch.as_tensor(view[None], device=device),
            torch.as_tensor(intrinsics[None], device=device),
            width,
            height,
            rasterize_mode="antialiased",
        )
        alpha_tensor = rendered_alpha[0, :, :, 0]
        image_tensor = rendered[0, :, :, :3] + (1.0 - alpha_tensor[:, :, None]) * 0.5
        gaussian_ids = metadata.get("gaussian_ids")
        radii = metadata.get("radii")
        if gaussian_ids is None or radii is None:
            raise NavigationStressError("navigation_stress_failed: gsplat projection metadata is missing")
        visible_opacities = raster_state["opacities"][gaussian_ids.long()]
        radius_values = radii.detach().cpu().numpy()
        opacity_values = visible_opacities.detach().cpu().numpy()
        gaussian_id_values = gaussian_ids.detach().cpu().numpy()
        image = image_tensor.detach().cpu().numpy()
        alpha = alpha_tensor.detach().cpu().numpy()

    transparent = float(np.mean(alpha < 0.01))
    clipped = float(np.mean(np.all(image < 0.02, axis=2) | np.all(image > 0.98, axis=2)))
    giant_ids, giant_ids_truncated = _giant_splat_ids(
        radius_values,
        opacity_values,
        gaussian_id_values,
        width=width,
        height=height,
    )
    return image, {
        "finite_pixels": bool(np.isfinite(image).all()),
        "transparent_ratio": transparent,
        "black_white_clip_ratio": clipped,
        "mean_brightness": float(image.mean()),
        "giant_gaussian_count": len(giant_ids) if not giant_ids_truncated else int(
            _count_giant_splats(radius_values, opacity_values, width=width, height=height)
        ),
        "giant_gaussian_ids": giant_ids,
        "giant_gaussian_ids_truncated": giant_ids_truncated,
        "maximum_projected_radius": float(np.max(radius_values)) if radius_values.size else 0.0,
        "visible_gaussian_count": int(len(gaussian_ids)),
    }


def _lpips(first: Any, second: Any) -> float:
    import numpy as np
    try:
        import torch
        from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
        left = torch.from_numpy(np.asarray(first).transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
        right = torch.from_numpy(np.asarray(second).transpose(2, 0, 1)).unsqueeze(0).float() * 2 - 1
        # The production node keeps the trainer's audited VGG16 weights on the
        # persistent TORCH_HOME volume, so stress evaluation remains offline
        # and deterministic across Pod restarts.
        return float(learned_perceptual_image_patch_similarity(left, right, net_type="vgg").item())
    except Exception as exc:
        raise NavigationStressError(f"navigation_stress_failed: LPIPS evaluator unavailable: {exc}") from exc


def validate_stress_measurements(
    walk: dict[str, Any],
    view_metrics: list[dict[str, Any]],
    pair_metrics: list[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if walk.get("completed_steps") != NAVIGATION_STEPS or not walk.get("all_camera_positions_finite"):
        reasons.append("navigation_walk_non_finite_or_incomplete")
    if int(walk.get("accepted_steps", 0)) < NAVIGATION_STEPS // 10:
        reasons.append("navigation_walk_has_insufficient_safe_translation")
    if not walk.get("height_preserved") or not walk.get("final_position_inside_bounds"):
        reasons.append("navigation_walk_left_bounds_or_changed_height")
    if not walk.get("collision_mesh_checked"):
        reasons.append("collision_mesh_not_checked")
    if len(view_metrics) != REPRESENTATIVE_VIEWS:
        reasons.append("representative_render_count_not_32")
    for index, item in enumerate(view_metrics):
        if not item.get("finite_pixels"):
            reasons.append(f"view_{index:02d}_pixels_non_finite")
        if int(item.get("giant_gaussian_count", 0)):
            reasons.append(f"view_{index:02d}_giant_gaussian")
        if float(item.get("transparent_ratio", 1.0)) > MAXIMUM_TRANSPARENT_RATIO:
            reasons.append(f"view_{index:02d}_transparent_above_90_percent")
        if float(item.get("black_white_clip_ratio", 1.0)) > MAXIMUM_BLACK_WHITE_CLIP_RATIO:
            reasons.append(f"view_{index:02d}_black_white_clipped_above_95_percent")
    if len(pair_metrics) != MICRO_TRANSLATION_PAIRS:
        reasons.append("micro_translation_pair_count_not_16")
    for index, item in enumerate(pair_metrics):
        if not math.isfinite(float(item.get("lpips", math.inf))) or float(item["lpips"]) > MAXIMUM_MICRO_LPIPS:
            reasons.append(f"micro_pair_{index:02d}_lpips_above_0.15")
        if not math.isfinite(float(item.get("brightness_delta", math.inf))) or float(item["brightness_delta"]) > MAXIMUM_MICRO_BRIGHTNESS_DELTA:
            reasons.append(f"micro_pair_{index:02d}_brightness_delta_above_0.10")
    return reasons


def run_navigation_stress(task_dir: Path, *, task_type: str) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    if task_type == "worldrecon":
        derive_worldrecon_viewer_metadata(task_dir)
    metadata_path = _metadata_path(task_dir)
    gaussian_path = _gaussian_path(task_dir)
    if metadata_path is None or gaussian_path is None:
        raise NavigationStressError("navigation_stress_failed: metadata or Gaussian artifact is missing")
    metadata = _read_json(metadata_path)
    try:
        if not _inside_bounds(metadata["center_point"], metadata):
            raise NavigationStressError("navigation_stress_failed: official camera pose is outside navigation bounds")
        gaussian = _load_gaussians(gaussian_path)
        raster_state = _prepare_raster_state(gaussian)
    except (KeyError, TypeError, ValueError, ViewerMetadataError) as exc:
        raise NavigationStressError(f"navigation_stress_failed: {exc}") from exc
    collision_path = task_dir / "output" / "ply" / "fuse_simplified.ply"
    walk = _stress_walk(metadata, collision_path if collision_path.is_file() else None)
    poses = _camera_poses(metadata)
    up = np.asarray(metadata["up_direction"], dtype=np.float64)
    right, _, _ = _viewer_basis(metadata)
    human_scale = float(metadata.get("human_scale", 1.0))
    output_dir = task_dir / "navigation_stress_renders"
    output_dir.mkdir(parents=True, exist_ok=True)
    view_metrics: list[dict[str, Any]] = []
    rendered: list[Any] = []
    for index, (position, target) in enumerate(poses):
        image, metrics = _render_projection(raster_state, position, target, up)
        Image.fromarray((image * 255).astype("uint8"), mode="RGB").save(output_dir / f"view-{index:02d}.png")
        metrics["view"] = index
        metrics["camera_position"] = [float(value) for value in position]
        view_metrics.append(metrics)
        rendered.append(image)
    pair_metrics: list[dict[str, Any]] = []
    for index, (position, target) in enumerate(poses[:MICRO_TRANSLATION_PAIRS]):
        shifted = position + right * (0.01 * human_scale)
        shifted_target = target + right * (0.01 * human_scale)
        image, metrics = _render_projection(raster_state, shifted, shifted_target, up)
        Image.fromarray((image * 255).astype("uint8"), mode="RGB").save(output_dir / f"micro-{index:02d}.png")
        pair_metrics.append({
            "pair": index,
            "lpips": _lpips(rendered[index], image),
            "brightness_delta": abs(float(rendered[index].mean()) - float(image.mean())),
            "finite_pixels": metrics["finite_pixels"],
        })
    reasons = validate_stress_measurements(walk, view_metrics, pair_metrics)
    giant_ids = sorted({
        int(gaussian_id)
        for metrics in view_metrics
        for gaussian_id in metrics.get("giant_gaussian_ids", [])
    })
    giant_ids_complete = not any(
        metrics.get("giant_gaussian_ids_truncated") for metrics in view_metrics
    )
    giant_only_failure = bool(reasons) and all(
        reason.startswith("view_") and reason.endswith("_giant_gaussian")
        for reason in reasons
    )
    repair_eligible = bool(
        giant_only_failure
        and giant_ids_complete
        and giant_ids
        and len(giant_ids) <= MAXIMUM_AUTOREPAIR_GAUSSIANS
        and len(giant_ids) / max(len(gaussian["positions"]), 1) <= MAXIMUM_AUTOREPAIR_FRACTION
        and len(gaussian["positions"]) - len(giant_ids) >= 50_000
    )
    report = {
        "version": CONTRACT_VERSION,
        "state": "passed" if not reasons else "failed",
        "free_view_eligible": not reasons,
        "seed": NAVIGATION_SEED,
        "gaussian_path": str(gaussian_path.relative_to(task_dir)),
        "metadata_path": str(metadata_path.relative_to(task_dir)),
        "gaussian_count": len(gaussian["positions"]),
        "all_gaussian_attributes_finite": True,
        "walk": walk,
        "representative_views": view_metrics,
        "micro_translation_pairs": pair_metrics,
        "thresholds": {
            "navigation_steps": NAVIGATION_STEPS,
            "representative_views": REPRESENTATIVE_VIEWS,
            "micro_translation_pairs": MICRO_TRANSLATION_PAIRS,
            "maximum_giant_projected_radius_ratio": MAXIMUM_GIANT_RADIUS_RATIO,
            "minimum_giant_opacity": 0.05,
            "maximum_transparent_ratio": MAXIMUM_TRANSPARENT_RATIO,
            "maximum_black_white_clip_ratio": MAXIMUM_BLACK_WHITE_CLIP_RATIO,
            "maximum_micro_lpips": MAXIMUM_MICRO_LPIPS,
            "maximum_micro_brightness_delta": MAXIMUM_MICRO_BRIGHTNESS_DELTA,
        },
        "failure_reasons": reasons,
        "repair_analysis": {
            "eligible": repair_eligible,
            "failure_is_giant_gaussian_only": giant_only_failure,
            "offending_gaussian_count": len(giant_ids),
            "offending_gaussian_fraction": len(giant_ids) / max(len(gaussian["positions"]), 1),
            "offending_gaussian_ids": giant_ids,
            "offending_gaussian_ids_complete": giant_ids_complete,
            "maximum_repair_gaussians": MAXIMUM_AUTOREPAIR_GAUSSIANS,
            "maximum_repair_fraction": MAXIMUM_AUTOREPAIR_FRACTION,
            "maximum_repair_attempts": 1,
        },
    }
    write_json_atomic(task_dir / "navigation_stress.json", report)
    if reasons:
        raise NavigationStressError("navigation_stress_failed: " + ", ".join(reasons[:8]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--task-type", choices=("worldgen", "worldgen_from_pano", "worldrecon"), required=True)
    args = parser.parse_args()
    run_navigation_stress(Path(args.task_dir), task_type=args.task_type)


if __name__ == "__main__":
    main()
