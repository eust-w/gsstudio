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
MINIMUM_STRICT_TRAJECTORIES = 6
MINIMUM_TRANSLATED_CAMERA_CENTERS = 12
MINIMUM_AZIMUTH_BINS = 6
MINIMUM_VALIDATION_RENDERS = 4
MINIMUM_V4_VALIDATION_RENDERS = 8
MINIMUM_VALIDATION_PSNR = 28.0
MINIMUM_VALIDATION_SSIM = 0.90
MAXIMUM_VALIDATION_LPIPS = 0.10


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
    if tier != "strict_3d":
        raise DeliveryValidationError(
            "World generation can train only after strict free-view trajectory coverage is satisfied"
        )
    return "accepted_trajectories"


def build_training_input_manifest(task_dir: Path, scene_dir: Path, delivery_tier: str) -> dict[str, Any]:
    alignment = _read_object(task_dir / "alignment_quality.json")
    accepted = sorted({str(item) for item in alignment.get("accepted_trajectory_ids", [])})
    if delivery_tier != "strict_3d" or len(accepted) < MINIMUM_STRICT_TRAJECTORIES:
        raise DeliveryValidationError(
            f"Strict free-view 3DGS requires at least {MINIMUM_STRICT_TRAJECTORIES} accepted trajectories; "
            f"only {len(accepted)} passed"
        )
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


def _navigation_validation(task_dir: Path, metadata_path: Path) -> dict[str, Any]:
    """Validate translated, multi-angle camera coverage and held-out renders."""
    try:
        import numpy as np
    except Exception as exc:
        raise DeliveryValidationError(f"Navigation validation requires NumPy: {exc}") from exc

    alignment = _read_object(task_dir / "alignment_quality.json")
    accepted = sorted({str(item) for item in alignment.get("accepted_trajectory_ids", [])})
    manifest = _read_object(task_dir / "training_input_manifest.json")
    metadata = _read_object(metadata_path)
    cameras_path = task_dir / "scene" / "gs_data" / "cameras.json"
    cameras = _read_object(cameras_path)

    human_scale = metadata.get("human_scale", 1.0)
    try:
        human_scale = float(human_scale)
        if not math.isfinite(human_scale) or human_scale <= 0:
            human_scale = 1.0
    except (TypeError, ValueError):
        human_scale = 1.0
    camera_obb = metadata.get("kwargs", {}).get("camera_obb") if isinstance(metadata.get("kwargs"), dict) else None
    obb_half_width = 0.0
    obb_half_height = 0.0
    if isinstance(camera_obb, list) and len(camera_obb) >= 5:
        try:
            obb_half_width = abs(float(camera_obb[3]))
            obb_half_height = abs(float(camera_obb[4]))
            if not math.isfinite(obb_half_width) or not math.isfinite(obb_half_height):
                obb_half_width = 0.0
                obb_half_height = 0.0
        except (TypeError, ValueError):
            pass

    try:
        up = np.asarray(metadata["up_direction"], dtype=np.float64)
        up_norm = float(np.linalg.norm(up))
        if up.shape != (3,) or not np.isfinite(up).all() or up_norm <= 1e-9:
            raise ValueError("invalid up direction")
        up /= up_norm
    except (KeyError, TypeError, ValueError, FloatingPointError):
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(fallback, up))) > 0.9:
        fallback = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    reference = fallback - up * float(np.dot(fallback, up))
    reference /= np.linalg.norm(reference)
    reference_cross = np.cross(up, reference)

    centers: list[Any] = []
    azimuth_bins: set[int] = set()
    trajectory_camera_count = 0
    accepted_prefixes = tuple(f"{trajectory_id.replace('/', '-')}_" for trajectory_id in accepted)
    rejected_camera_ids: list[str] = []
    for camera_id, value in cameras.items():
        if not isinstance(value, dict) or "extrinsic" not in value:
            continue
        if value.get("source_type") in {"panorama", "polar_up", "polar_down"}:
            continue
        if not isinstance(camera_id, str) or not camera_id.startswith(accepted_prefixes):
            rejected_camera_ids.append(str(camera_id))
            continue
        try:
            extrinsic = np.asarray(value["extrinsic"], dtype=np.float64)
            if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
                continue
            rotation = extrinsic[:3, :3]
            translation = extrinsic[:3, 3]
            center = -rotation.T @ translation
            forward = rotation.T @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            horizontal = forward - up * float(np.dot(forward, up))
            horizontal_norm = float(np.linalg.norm(horizontal))
            if horizontal_norm <= 1e-9:
                continue
            horizontal /= horizontal_norm
        except (TypeError, ValueError):
            continue
        centers.append(center)
        angle = math.atan2(float(np.dot(horizontal, reference_cross)), float(np.dot(horizontal, reference)))
        azimuth_bins.add(int(((angle + 2 * math.pi) % (2 * math.pi)) / (math.pi / 4)))
        trajectory_camera_count += 1

    center_quantum = max(human_scale * 1e-4, 1e-6)
    unique_centers = {
        tuple(int(round(float(component) / center_quantum)) for component in center)
        for center in centers
    }

    stat_files = sorted((task_dir / "output" / "stats").glob("val_step*.json"))
    metrics = _read_object(stat_files[-1]) if stat_files else {}
    metric_values: dict[str, float | None] = {}
    for name in ("psnr", "ssim", "lpips"):
        try:
            value = float(metrics[name])
            metric_values[name] = value if math.isfinite(value) else None
        except (KeyError, TypeError, ValueError):
            metric_values[name] = None
    validation_render_count = len(list((task_dir / "output" / "renders").glob("val_step*.png")))

    reasons: list[str] = []
    if len(accepted) < MINIMUM_STRICT_TRAJECTORIES:
        reasons.append("strict_trajectory_count_below_6")
    if manifest.get("delivery_tier") != "strict_3d" or manifest.get("training_mode") != "accepted_trajectories":
        reasons.append("training_manifest_is_not_strict")
    if sorted(manifest.get("accepted_trajectory_ids", [])) != accepted:
        reasons.append("training_manifest_accepted_set_mismatch")
    if manifest.get("rejected_trajectories_consumed") is not False:
        reasons.append("rejected_trajectory_consumption_not_proven_false")
    if rejected_camera_ids:
        reasons.append("rejected_or_unapproved_trajectory_cameras_consumed")
    if len(unique_centers) < MINIMUM_TRANSLATED_CAMERA_CENTERS:
        reasons.append("translated_camera_centers_below_12")
    if min(obb_half_width, obb_half_height) <= human_scale * 0.1:
        reasons.append("camera_obb_is_not_navigable")
    if len(azimuth_bins) < MINIMUM_AZIMUTH_BINS:
        reasons.append("azimuth_coverage_below_6_bins")
    contract = _read_object(task_dir / "task_contract.json")
    minimum_validation_renders = (
        MINIMUM_V4_VALIDATION_RENDERS
        if int(contract.get("contract_version", 0) or 0) >= 4
        else MINIMUM_VALIDATION_RENDERS
    )
    if validation_render_count < minimum_validation_renders:
        reasons.append(f"validation_render_count_below_{minimum_validation_renders}")
    if metric_values["psnr"] is None or metric_values["psnr"] < MINIMUM_VALIDATION_PSNR:
        reasons.append("validation_psnr_below_28")
    if metric_values["ssim"] is None or metric_values["ssim"] < MINIMUM_VALIDATION_SSIM:
        reasons.append("validation_ssim_below_0.90")
    if metric_values["lpips"] is None or metric_values["lpips"] > MAXIMUM_VALIDATION_LPIPS:
        reasons.append("validation_lpips_above_0.10")

    report = {
        "version": 1,
        "free_view_eligible": not reasons,
        "accepted_trajectory_count": len(accepted),
        "trajectory_camera_count": trajectory_camera_count,
        "rejected_or_unapproved_camera_count": len(rejected_camera_ids),
        "rejected_or_unapproved_camera_ids": rejected_camera_ids[:50],
        "unique_translated_camera_centers": len(unique_centers),
        "camera_obb_half_width": obb_half_width,
        "camera_obb_half_height": obb_half_height,
        "human_scale": human_scale,
        "azimuth_bin_count": len(azimuth_bins),
        "azimuth_bins": sorted(azimuth_bins),
        "validation_render_count": validation_render_count,
        "metrics": metric_values,
        "thresholds": {
            "minimum_accepted_trajectories": MINIMUM_STRICT_TRAJECTORIES,
            "minimum_translated_camera_centers": MINIMUM_TRANSLATED_CAMERA_CENTERS,
            "minimum_camera_obb_half_extent_in_human_scales": 0.1,
            "minimum_azimuth_45_degree_bins": MINIMUM_AZIMUTH_BINS,
            "minimum_validation_renders": minimum_validation_renders,
            "minimum_psnr": MINIMUM_VALIDATION_PSNR,
            "minimum_ssim": MINIMUM_VALIDATION_SSIM,
            "maximum_lpips": MAXIMUM_VALIDATION_LPIPS,
        },
        "failure_reasons": reasons,
    }
    _write_atomic(task_dir / "navigation_validation.json", report)
    return report


def _find_viewer_metadata(task_dir: Path) -> Path | None:
    # Delivery validation must bind viewer metadata to the current result.
    # A task can contain older valid metadata under recovery_attempts/, which
    # must never satisfy validation for a newly trained output.
    preferred = [
        task_dir / "output" / "ply" / "position_meta_info.json",
        task_dir / "output" / "position_meta_info.json",
        task_dir / "scene" / "position_meta_info.json",
        task_dir / "position_meta_info.json",
    ]
    candidates = [
        *preferred,
        *(task_dir / "output").rglob("position_meta_info.json"),
        *(task_dir / "scene").rglob("position_meta_info.json"),
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        value = _read_object(path)
        if all(key in value for key in ("center_point", "facing_direction", "up_direction")):
            return path
    return None


def validate_completed_delivery(
    task_dir: Path,
    *,
    task_type: str,
    require_navigation_stress: bool = True,
) -> dict[str, Any]:
    contract = _read_object(task_dir / "task_contract.json")
    contract_version = int(contract.get("contract_version", 0) or 0)
    input_preflight = _read_object(task_dir / "input_preflight.json")
    post_panorama_preflight = _read_object(task_dir / "post_panorama_preflight.json")
    navigation_stress = _read_object(task_dir / "navigation_stress.json")
    if task_type == "worldrecon":
        if contract_version < 4:
            return {
                "required": False,
                "valid": True,
                "legacy_unvalidated": True,
                "free_view_eligible": False,
            }
        if input_preflight.get("state") != "passed":
            raise DeliveryValidationError("Completed reconstruction requires a passed v4 input preflight")
        if require_navigation_stress and navigation_stress.get("free_view_eligible") is not True:
            raise DeliveryValidationError("Completed reconstruction requires a passed navigation stress test")
        gaussian = next(
            (path for path in sorted((task_dir / "output").rglob("*.ply")) if "gaussian" in path.name.lower()),
            None,
        )
        metadata = _find_viewer_metadata(task_dir)
        if gaussian is None or metadata is None:
            raise DeliveryValidationError("Completed reconstruction requires Gaussian PLY and viewer metadata")
        result = {
            "required": True,
            "valid": True,
            "contract_version": contract_version,
            "ply_path": str(gaussian.relative_to(task_dir)),
            "metadata_path": str(metadata.relative_to(task_dir)),
            "free_view_eligible": bool(require_navigation_stress and navigation_stress.get("free_view_eligible") is True),
            "navigation_stress": navigation_stress or None,
        }
        _write_atomic(task_dir / "artifact_validation.json", result)
        return result
    if task_type not in {"worldgen", "worldgen_from_pano"}:
        return {"required": False, "valid": True}
    if contract_version >= 4:
        if input_preflight.get("state") != "passed":
            raise DeliveryValidationError("Completed world generation requires a passed v4 input preflight")
        if post_panorama_preflight.get("state") != "passed":
            raise DeliveryValidationError(
                "Completed world generation requires a passed generated-panorama preflight"
            )
        pilot = _read_object(task_dir / "pilot_validation.json")
        if pilot.get("state") != "passed":
            raise DeliveryValidationError("Completed world generation requires a passed 500-step pilot")
        if require_navigation_stress and navigation_stress.get("free_view_eligible") is not True:
            raise DeliveryValidationError("Completed world generation requires a passed navigation stress test")
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

    navigation_validation = _navigation_validation(task_dir, metadata)
    if not navigation_validation["free_view_eligible"]:
        raise DeliveryValidationError(
            "Completed world generation requires strict free-view coverage: "
            + ", ".join(navigation_validation["failure_reasons"])
        )

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
        ordered_extent = np.sort(extent)
        if (
            not np.isfinite(extent).all()
            or ordered_extent[-1] <= 1e-6
            or ordered_extent[-2] <= ordered_extent[-1] * 1e-3
        ):
            raise DeliveryValidationError("SPZ has a degenerate spatial bounding box")
    except DeliveryValidationError:
        raise
    except Exception as exc:
        raise DeliveryValidationError(f"SPZ could not be decoded: {exc}") from exc

    result = {
        "required": True,
        "valid": True,
        "contract_version": contract_version or 3,
        "spz_path": str(spz_files[-1].relative_to(task_dir)),
        "ply_path": str(ply_files[-1].relative_to(task_dir)),
        "checkpoint_path": str(checkpoints[-1].relative_to(task_dir)),
        "metadata_path": str(metadata.relative_to(task_dir)),
        "gaussian_count": point_count,
        "finite_attributes": True,
        "finite_opacities": True,
        "opacity_logit_endpoint_count": opacity_logit_endpoint_count,
        "non_degenerate_bounds": True,
        "free_view_eligible": bool(
            navigation_validation["free_view_eligible"]
            and (contract_version < 4 or navigation_stress.get("free_view_eligible") is True)
        ),
        "navigation_validation": navigation_validation,
        "navigation_stress": navigation_stress or None,
    }
    _write_atomic(task_dir / "artifact_validation.json", result)
    return result


def prepare_safe_training_retry(task_dir: Path, scene_dir: Path, *, maximum_points: int = 1_000_000) -> dict[str, Any]:
    """Reduce initialization memory only for a strict free-view training set."""
    manifest = _read_object(task_dir / "training_input_manifest.json")
    accepted = manifest.get("accepted_trajectory_ids", [])
    if (
        manifest.get("delivery_tier") != "strict_3d"
        or manifest.get("training_mode") != "accepted_trajectories"
        or not isinstance(accepted, list)
        or len(accepted) < MINIMUM_STRICT_TRAJECTORIES
    ):
        raise DeliveryValidationError("Safe training retry requires a strict free-view training manifest")
    points_path = scene_dir / "gs_data" / "points.ply"
    if not points_path.is_file():
        raise DeliveryValidationError("Safe training retry requires gs_data/points.ply")
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
