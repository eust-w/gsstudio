"""Version-4 input preflight and immutable strict-free-view contract.

This module deliberately runs before any expensive reconstruction or
WorldStereo work.  It keeps rejected inputs as evidence, writes reports
atomically, and only promotes a validated subset into ``preflight_selected``.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import math
import mimetypes
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterable
import urllib.request


CONTRACT_VERSION = 4
SUCCESS_POLICY = "strict_free_view"
PRODUCTION_LANE = "production_multiview"
GENERATIVE_LANE = "generative_worldgen"
SOURCE_CANDIDATE_COUNT = 4
MAXIMUM_SOURCE_ATTEMPTS = 3
MAXIMUM_WORLDSTEREO_TRAJECTORIES = 24
MAXIMUM_TASK_SECONDS = 4 * 60 * 60
MINIMUM_MULTIVIEW_FRAMES = 12
VIDEO_CANDIDATE_COUNT = 24
MAXIMUM_SELECTED_FRAMES = 48
MINIMUM_SHORT_EDGE = 512
MINIMUM_COMPOSITE_SCORE = 0.70
CV_WEIGHT = 0.55
VLM_WEIGHT = 0.45


class InputPreflightError(RuntimeError):
    """The input cannot enter strict 3D production."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def strict_contract(task_type: str) -> dict[str, Any]:
    lane = PRODUCTION_LANE if task_type == "worldrecon" else GENERATIVE_LANE
    return {
        "contract_version": CONTRACT_VERSION,
        "success_policy": SUCCESS_POLICY,
        "input_lane": lane,
        "delivery_tier": "strict_3d",
        "source_candidate_count": SOURCE_CANDIDATE_COUNT if task_type == "worldgen" else 1,
        "maximum_source_attempts": MAXIMUM_SOURCE_ATTEMPTS if task_type == "worldgen" else 1,
        "maximum_worldstereo_trajectories": MAXIMUM_WORLDSTEREO_TRAJECTORIES,
        "maximum_task_seconds": MAXIMUM_TASK_SECONDS,
        "minimum_multiview_frames": MINIMUM_MULTIVIEW_FRAMES,
        "video_candidate_count": VIDEO_CANDIDATE_COUNT,
        "maximum_selected_frames": MAXIMUM_SELECTED_FRAMES,
        "preflight_cv_weight": CV_WEIGHT,
        "preflight_vlm_weight": VLM_WEIGHT,
        "minimum_preflight_score": MINIMUM_COMPOSITE_SCORE,
        "post_panorama_preflight_required": task_type in {"worldgen", "worldgen_from_pano"},
        "minimum_accepted_trajectories": 6,
        "target_accepted_trajectories": 10,
        "initial_regular_trajectories": 8,
        "trajectory_refill_count": 4,
        "maximum_trajectories_per_source": 16,
        "pilot_steps": 500,
        "pilot_minimum_renders": 4,
        "pilot_minimum_psnr": 22.0,
        "pilot_minimum_ssim": 0.75,
        "pilot_maximum_lpips": 0.25,
        "pilot_minimum_gaussians": 50_000,
        "training_steps": 2000,
        "final_minimum_renders": 8,
        "final_minimum_psnr": 28.0,
        "final_minimum_ssim": 0.90,
        "final_maximum_lpips": 0.10,
        "minimum_translated_camera_centers": 12,
        "minimum_azimuth_bins": 6,
        "minimum_obb_half_extent_human_scale": 0.1,
        "navigation_seed": 1024,
        "navigation_steps": 1000,
        "navigation_render_views": 32,
        "navigation_micro_translation_pairs": 16,
        "navigation_maximum_giant_radius_ratio": 0.50,
        "navigation_maximum_transparent_ratio": 0.90,
        "navigation_maximum_black_white_clip_ratio": 0.95,
        "navigation_maximum_micro_lpips": 0.15,
        "navigation_maximum_brightness_delta": 0.10,
        "gaussian_repair_maximum_attempts": 1,
        "gaussian_repair_maximum_count": 500,
        "gaussian_repair_maximum_fraction": 0.0005,
    }


def normalize_task_contract(config: dict[str, Any] | None, task_type: str) -> dict[str, Any]:
    """Apply immutable server policy after parsing user JSON.

    User-supplied advanced JSON is intentionally unable to raise budgets or
    weaken any gate.  Unknown, non-policy keys are retained for model/runtime
    configuration.
    """
    normalized = dict(config or {})
    normalized.update(strict_contract(task_type))
    if task_type == "worldrecon":
        normalized["quality_preset"] = "high"
    return normalized


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _direction_groups(lines: Any) -> int:
    if lines is None:
        return 0
    angles: list[float] = []
    for line in lines[:200]:
        x1, y1, x2, y2 = [float(value) for value in line[0]]
        angle = math.atan2(y2 - y1, x2 - x1) % math.pi
        angles.append(angle)
    occupied = {
        int(angle / (math.pi / 12)) % 12
        for angle in angles
    }
    groups = 0
    remaining = set(occupied)
    while remaining:
        seed = remaining.pop()
        adjacent = {(seed - 1) % 12, (seed + 1) % 12}
        remaining.difference_update(adjacent)
        groups += 1
    return groups


def analyze_image_geometry(path: Path) -> dict[str, Any]:
    """Compute deterministic image-only reconstruction suitability metrics."""
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"usable": False, "hard_reject_reasons": ["decode_failed"], "cv_score": 0.0}
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(35, min(width, height) // 12),
        minLineLength=max(24, min(width, height) // 10), maxLineGap=12,
    )
    line_groups = _direction_groups(lines)
    detector = cv2.ORB_create(nfeatures=3000)
    keypoints = detector.detect(gray, None)
    feature_count = len(keypoints)

    reasons: list[str] = []
    if min(width, height) < MINIMUM_SHORT_EDGE:
        reasons.append("short_edge_below_512")
    if clipped > 0.15:
        reasons.append("clipped_pixels_above_15_percent")
    if not 0.03 <= edge_density <= 0.35:
        reasons.append("edge_density_outside_3_to_35_percent")
    if line_groups < 2:
        reasons.append("fewer_than_two_major_line_directions")
    if laplacian_variance < 45.0:
        reasons.append("image_too_blurry")
    if feature_count < 250:
        reasons.append("insufficient_feature_points")

    resolution_score = _clip01(min(width, height) / 832.0)
    blur_score = _clip01((laplacian_variance - 25.0) / 175.0)
    exposure_score = _clip01(1.0 - clipped / 0.15)
    edge_score = _clip01(1.0 - abs(edge_density - 0.14) / 0.14)
    feature_score = _clip01(feature_count / 1200.0)
    direction_score = _clip01(line_groups / 3.0)
    cv_score = (
        0.15 * resolution_score
        + 0.20 * blur_score
        + 0.15 * exposure_score
        + 0.15 * edge_score
        + 0.20 * feature_score
        + 0.15 * direction_score
    )
    return {
        "usable": not reasons,
        "width": width,
        "height": height,
        "short_edge": min(width, height),
        "laplacian_variance": round(laplacian_variance, 4),
        "clipped_pixel_ratio": round(clipped, 6),
        "edge_density": round(edge_density, 6),
        "feature_count": feature_count,
        "major_line_direction_groups": line_groups,
        "cv_score": round(cv_score, 6),
        "hard_reject_reasons": reasons,
    }


def analyze_pair_geometry(first_path: Path, second_path: Path) -> dict[str, Any]:
    """Measure verified feature/parallax evidence and reject pure rotation."""
    import cv2
    import numpy as np

    first = cv2.imread(str(first_path), cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(str(second_path), cv2.IMREAD_GRAYSCALE)
    if first is None or second is None:
        return {"usable": False, "reason": "decode_failed"}
    if first.shape != second.shape:
        second = cv2.resize(second, (first.shape[1], first.shape[0]))
    detector = cv2.ORB_create(nfeatures=4000)
    kp1, des1 = detector.detectAndCompute(first, None)
    kp2, des2 = detector.detectAndCompute(second, None)
    if des1 is None or des2 is None:
        return {"usable": False, "reason": "missing_descriptors", "match_inliers": 0}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [left for left, right in pairs if left.distance < 0.75 * right.distance]
    if len(good) < 16:
        return {"usable": False, "reason": "insufficient_feature_matches", "match_inliers": len(good)}
    points1 = np.float32([kp1[item.queryIdx].pt for item in good])
    points2 = np.float32([kp2[item.trainIdx].pt for item in good])
    homography, homography_mask = cv2.findHomography(points1, points2, cv2.RANSAC, 3.0)
    fundamental, fundamental_mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 1.5, 0.995)
    h_inliers = int(homography_mask.sum()) if homography is not None and homography_mask is not None else 0
    f_inliers = int(fundamental_mask.sum()) if fundamental is not None and fundamental_mask is not None else 0
    flow = np.linalg.norm(points2 - points1, axis=1)
    median_flow = float(np.median(flow))
    diagonal = math.hypot(first.shape[0], first.shape[1])
    normalized_flow = median_flow / max(diagonal, 1.0)
    # A homography explaining almost every correspondence with little flow is
    # the observable signature of a panorama-like pure rotation sequence.
    homography_ratio = h_inliers / max(len(good), 1)
    parallax_ratio = f_inliers / max(h_inliers, 1)
    pure_rotation = bool(homography_ratio > 0.90 and normalized_flow < 0.012) or parallax_ratio < 0.25
    usable = f_inliers >= 20 and normalized_flow >= 0.004 and not pure_rotation
    return {
        "usable": usable,
        "feature_matches": len(good),
        "match_inliers": f_inliers,
        "homography_inliers": h_inliers,
        "homography_inlier_ratio": round(homography_ratio, 6),
        "fundamental_to_homography_inlier_ratio": round(parallax_ratio, 6),
        "median_optical_flow": round(median_flow, 4),
        "normalized_optical_flow": round(normalized_flow, 6),
        "pure_rotation": pure_rotation,
        "reason": None if usable else "non_translational_or_insufficient_parallax",
    }


VLM_SYSTEM_PROMPT = """You are a strict photogrammetry input inspector. Return one JSON object only.
Score reconstruction_suitability from 0 to 1 and set each boolean flag accurately:
person, mirror_or_glass, large_reflection, curved_walls, unwalkable_floor,
perspective_conflict. Also provide a short reasons array. Do not use markdown."""

PANORAMA_VLM_SYSTEM_PROMPT = """You are a strict 3D reconstruction inspector evaluating an
equirectangular panorama after panorama generation. Return one JSON object only. Score
reconstruction_suitability from 0 to 1 and set each boolean flag accurately: person,
mirror_or_glass, large_reflection, curved_walls, unwalkable_floor, perspective_conflict.
Account for normal equirectangular projection across the entire panorama: straight physical
ceiling and floor boundaries can appear as smooth curves away from the horizon, not only at
the left/right seam. Do not set curved_walls merely because ceiling or floor lines have this
expected projection curve. Set curved_walls only when the wall surface itself is physically
bowed, cylindrical, or structurally inconsistent after accounting for equirectangular
projection. Still flag duplicated structures, inconsistent vanishing geometry, people,
reflections, or unusable walking space. Also provide a short reasons array. Do not use
markdown."""


def qwen_vlm_score(
    image_path: Path,
    *,
    host: str,
    port: int,
    model: str,
    timeout: int = 180,
    system_prompt: str = VLM_SYSTEM_PROMPT,
) -> dict[str, Any]:
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 300,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            ],
        }],
    }
    request = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    content = payload["choices"][0]["message"]["content"].strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    # Qwen3-VL can still wrap JSON in a short reasoning preamble even at
    # temperature zero.  Accept exactly one outer object, but never silently
    # fall back to a prose-derived score.
    object_start = content.find("{")
    object_end = content.rfind("}")
    if object_start < 0 or object_end <= object_start:
        raise InputPreflightError("Qwen3-VL response did not contain a JSON object")
    result = json.loads(content[object_start:object_end + 1])
    flags = {
        name: bool(result.get(name, False))
        for name in (
            "person", "mirror_or_glass", "large_reflection", "curved_walls",
            "unwalkable_floor", "perspective_conflict",
        )
    }
    score = _clip01(float(result.get("reconstruction_suitability", 0.0)))
    return {"vlm_score": score, "hard_flags": flags, "reasons": result.get("reasons", [])}


def _default_vlm_failure(exc: Exception) -> dict[str, Any]:
    # VLM evidence is mandatory in v4. Infrastructure failures may be retried
    # by the worker, but are never silently converted into a passing score.
    return {
        "vlm_score": 0.0,
        "hard_flags": {},
        "reasons": [f"vlm_scoring_failed:{type(exc).__name__}"],
        "infrastructure_error": str(exc),
    }


def score_source_candidates(
    task_dir: Path,
    candidates: Iterable[dict[str, Any]],
    *,
    vlm_scorer: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        path = task_dir / str(item["path"])
        cv = analyze_image_geometry(path)
        try:
            vlm = vlm_scorer(path)
        except Exception as exc:
            vlm = _default_vlm_failure(exc)
        hard_flags = sorted(name for name, active in vlm.get("hard_flags", {}).items() if active)
        reasons = [*cv.get("hard_reject_reasons", []), *hard_flags]
        if vlm.get("infrastructure_error"):
            reasons.append("vlm_scoring_unavailable")
        composite = CV_WEIGHT * float(cv.get("cv_score", 0.0)) + VLM_WEIGHT * float(vlm.get("vlm_score", 0.0))
        if composite < MINIMUM_COMPOSITE_SCORE:
            reasons.append("composite_score_below_0.70")
        item.update({
            "cv": cv,
            "vlm": vlm,
            "composite_score": round(composite, 6),
            "accepted": not reasons,
            "rejection_reasons": sorted(set(reasons)),
        })
        scored.append(item)
    scored.sort(key=lambda item: (-float(item["composite_score"]), int(item.get("seed", 0))))
    accepted = [item for item in scored if item["accepted"]]
    return {
        "version": CONTRACT_VERSION,
        "state": "passed" if accepted else "failed",
        "weights": {"opencv": CV_WEIGHT, "qwen3_vl": VLM_WEIGHT},
        "minimum_composite_score": MINIMUM_COMPOSITE_SCORE,
        "candidates": scored,
        "ranking": [item.get("id") for item in scored],
        "selected_candidate_id": accepted[0].get("id") if accepted else None,
        "accepted_candidate_count": len(accepted),
        "updated_at": _utcnow(),
    }


def _candidate_files(input_dir: Path) -> list[Path]:
    pool_dir = input_dir / "source_candidates"
    generated = sorted(pool_dir.glob("candidate-*.png")) if pool_dir.is_dir() else []
    if generated:
        return generated
    source = input_dir / "input.png"
    if source.is_file():
        return [source]
    panorama = input_dir / "panorama.png"
    return [panorama] if panorama.is_file() else []


def run_generative_preflight(
    task_dir: Path,
    input_dir: Path,
    *,
    host: str,
    port: int,
    model: str,
) -> dict[str, Any]:
    pool_path = task_dir / "source_pool.json"
    prior = _read_object(pool_path)
    prior_by_path = {
        str(item.get("path")): item
        for item in prior.get("candidates", [])
        if isinstance(item, dict)
    }
    candidates = []
    for index, path in enumerate(_candidate_files(input_dir)):
        relative = str(path.relative_to(task_dir))
        old = prior_by_path.get(relative, {})
        candidates.append({
            "id": old.get("id") or f"source-{index:02d}",
            "path": relative,
            "seed": int(old.get("seed", 0)),
            "generation": old.get("generation"),
        })
    report = score_source_candidates(
        task_dir,
        candidates,
        vlm_scorer=lambda path: qwen_vlm_score(path, host=host, port=port, model=model),
    )
    report["input_lane"] = GENERATIVE_LANE
    write_json_atomic(pool_path, report)
    write_json_atomic(task_dir / "input_preflight.json", report)
    selected_id = report.get("selected_candidate_id")
    selected = next((item for item in report["candidates"] if item["id"] == selected_id), None)
    if not selected:
        raise InputPreflightError("source_replacement_required: no source candidate passed input preflight")
    selected_path = task_dir / selected["path"]
    temporary = input_dir / ".selected_source.tmp.png"
    shutil.copy2(selected_path, temporary)
    os.replace(temporary, input_dir / "selected_source.png")
    return report


def run_post_panorama_preflight(
    task_dir: Path,
    panorama_path: Path,
    *,
    host: str,
    port: int,
    model: str,
    vlm_scorer: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Re-check the actual generated panorama before any trajectory GPU work.

    Text/source preflight cannot detect people, reflections, or perspective
    conflicts introduced by panorama expansion. This second immutable report
    evaluates the exact panorama that WorldNav and WorldStereo will consume.
    """
    if not panorama_path.is_file():
        report = {
            "version": CONTRACT_VERSION,
            "stage": "post_panorama_preflight",
            "state": "failed",
            "panorama_path": str(panorama_path),
            "failure_reasons": ["generated_panorama_missing"],
            "updated_at": _utcnow(),
        }
        write_json_atomic(task_dir / "post_panorama_preflight.json", report)
        raise InputPreflightError("post_panorama_preflight_failed: generated panorama is missing")
    try:
        relative = str(panorama_path.relative_to(task_dir))
    except ValueError:
        relative = str(panorama_path)
    scorer = vlm_scorer or (
        lambda path: qwen_vlm_score(
            path,
            host=host,
            port=port,
            model=model,
            system_prompt=PANORAMA_VLM_SYSTEM_PROMPT,
        )
    )
    scored = score_source_candidates(
        task_dir,
        [{"id": "generated-panorama", "path": relative, "seed": 0}],
        vlm_scorer=scorer,
    )
    candidate = scored["candidates"][0]
    report = {
        "version": CONTRACT_VERSION,
        "stage": "post_panorama_preflight",
        "input_lane": GENERATIVE_LANE,
        "state": "passed" if candidate.get("accepted") else "failed",
        "panorama_path": relative,
        "composite_score": candidate.get("composite_score"),
        "cv": candidate.get("cv"),
        "vlm": candidate.get("vlm"),
        "failure_reasons": candidate.get("rejection_reasons", []),
        "updated_at": _utcnow(),
    }
    write_json_atomic(task_dir / "post_panorama_preflight.json", report)
    if report["state"] != "passed":
        raise InputPreflightError(
            "source_replacement_required: post_panorama_preflight_failed: "
            + ", ".join(report["failure_reasons"])
        )
    return report


def extract_video_candidates(video_path: Path, output_dir: Path, *, count: int = VIDEO_CANDIDATE_COUNT) -> list[Path]:
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < count:
        capture.release()
        raise InputPreflightError(f"input_preflight_failed: video has {frame_count} frames; at least {count} are required")
    indices = sorted({round(index * (frame_count - 1) / (count - 1)) for index in range(count)})
    paths: list[Path] = []
    for output_index, frame_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        destination = output_dir / f"video-frame-{output_index:04d}.png"
        if not cv2.imwrite(str(destination), frame):
            continue
        paths.append(destination)
    capture.release()
    if len(paths) < count:
        raise InputPreflightError(f"input_preflight_failed: extracted only {len(paths)} of {count} video candidates")
    return paths


def _uploaded_multiview_files(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file()
        and path.name.startswith("image_")
        and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    )


def run_multiview_preflight(
    task_dir: Path,
    input_dir: Path,
    *,
    host: str,
    port: int,
    model: str,
    vlm_scorer: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    video = input_dir / "input_video.mp4"
    if video.is_file():
        candidates = extract_video_candidates(video, input_dir / "video_candidates")
        input_mode = "video"
    else:
        candidates = _uploaded_multiview_files(input_dir)
        input_mode = "images"
    if len(candidates) < MINIMUM_MULTIVIEW_FRAMES:
        report = {
            "version": CONTRACT_VERSION,
            "input_lane": PRODUCTION_LANE,
            "input_mode": input_mode,
            "state": "failed",
            "candidate_frame_count": len(candidates),
            "selected_frame_count": 0,
            "failure_reasons": ["usable_frame_count_below_12"],
            "updated_at": _utcnow(),
        }
        write_json_atomic(task_dir / "input_preflight.json", report)
        raise InputPreflightError("input_preflight_failed: at least 12 usable multi-view frames are required")

    scorer = vlm_scorer or (lambda path: qwen_vlm_score(path, host=host, port=port, model=model))
    frame_reports: list[dict[str, Any]] = []
    # Every retained candidate receives an independent zero-temperature Qwen
    # decision. A nearby frame can hide a person or reflection that is absent
    # from its neighbor, so scores are never inherited across frames.
    vlm_by_index: dict[int, dict[str, Any]] = {}
    for index in range(len(candidates)):
        try:
            vlm_by_index[index] = scorer(candidates[index])
        except Exception as exc:
            vlm_by_index[index] = _default_vlm_failure(exc)
    for index, path in enumerate(candidates):
        cv = analyze_image_geometry(path)
        vlm = vlm_by_index[index]
        hard_flags = sorted(name for name, active in vlm.get("hard_flags", {}).items() if active)
        score = CV_WEIGHT * float(cv.get("cv_score", 0.0)) + VLM_WEIGHT * float(vlm.get("vlm_score", 0.0))
        reasons = [*cv.get("hard_reject_reasons", []), *hard_flags]
        if vlm.get("infrastructure_error"):
            reasons.append("vlm_scoring_unavailable")
        if score < MINIMUM_COMPOSITE_SCORE:
            reasons.append("composite_score_below_0.70")
        frame_reports.append({
            "id": f"frame-{index:04d}",
            "path": str(path.relative_to(task_dir)),
            "cv": cv,
            "vlm": vlm,
            "vlm_reference_frame": index,
            "composite_score": round(score, 6),
            "accepted": not reasons,
            "rejection_reasons": sorted(set(reasons)),
        })

    accepted_indices = [index for index, item in enumerate(frame_reports) if item["accepted"]]
    pair_reports: list[dict[str, Any]] = []
    adjacency: dict[int, set[int]] = {index: set() for index in accepted_indices}
    for left, right in zip(accepted_indices, accepted_indices[1:]):
        pair = analyze_pair_geometry(candidates[left], candidates[right])
        pair.update({"left": left, "right": right})
        pair_reports.append(pair)
        if pair["usable"]:
            adjacency[left].add(right)
            adjacency[right].add(left)
    # Reconstruction needs one coherent translated-camera graph. A union of
    # disconnected good pairs can meet the numeric frame minimum while still
    # making SfM fail, so keep only the largest deterministic component.
    components: list[list[int]] = []
    unseen = set(accepted_indices)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: list[int] = []
        unseen.remove(root)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    largest_component = max(components, key=lambda value: (len(value), [-item for item in value]), default=[])
    selected_indices = largest_component[:MAXIMUM_SELECTED_FRAMES]
    failure_reasons: list[str] = []
    if len(selected_indices) < MINIMUM_MULTIVIEW_FRAMES:
        failure_reasons.append("usable_translational_frame_count_below_12")
    if pair_reports and not any(not item.get("pure_rotation", False) and item.get("usable") for item in pair_reports):
        failure_reasons.append("pure_rotation_or_insufficient_parallax")

    report = {
        "version": CONTRACT_VERSION,
        "input_lane": PRODUCTION_LANE,
        "input_mode": input_mode,
        "state": "passed" if not failure_reasons else "failed",
        "candidate_frame_count": len(candidates),
        "selected_frame_count": len(selected_indices),
        "frames": frame_reports,
        "pair_geometry": pair_reports,
        "failure_reasons": failure_reasons,
        "thresholds": {
            "minimum_frames": MINIMUM_MULTIVIEW_FRAMES,
            "minimum_short_edge": MINIMUM_SHORT_EDGE,
            "minimum_composite_score": MINIMUM_COMPOSITE_SCORE,
            "maximum_frames": MAXIMUM_SELECTED_FRAMES,
        },
        "updated_at": _utcnow(),
    }
    write_json_atomic(task_dir / "source_pool.json", {
        "version": CONTRACT_VERSION,
        "input_lane": PRODUCTION_LANE,
        "candidates": frame_reports,
        "selected_ids": [frame_reports[index]["id"] for index in selected_indices],
        "updated_at": _utcnow(),
    })
    write_json_atomic(task_dir / "input_preflight.json", report)
    if failure_reasons:
        raise InputPreflightError("input_preflight_failed: " + ", ".join(failure_reasons))

    selected_dir = input_dir / "preflight_selected"
    temporary_dir = input_dir / ".preflight_selected.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    for output_index, source_index in enumerate(selected_indices):
        source = candidates[source_index]
        shutil.copy2(source, temporary_dir / f"frame-{output_index:04d}{source.suffix.lower()}")
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    os.replace(temporary_dir, selected_dir)
    return report


def select_ranked_source_for_attempt(task_dir: Path, attempt: int, destination: Path) -> dict[str, Any]:
    pool = _read_object(task_dir / "source_pool.json")
    accepted = [item for item in pool.get("candidates", []) if isinstance(item, dict) and item.get("accepted")]
    if attempt < 0 or attempt >= min(len(accepted), MAXIMUM_SOURCE_ATTEMPTS):
        raise InputPreflightError("source_replacement_required: no validated source candidate remains")
    selected = accepted[attempt]
    source = task_dir / str(selected["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generative", "multiview", "panorama"))
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--panorama")
    parser.add_argument("--vllm-host", default="localhost")
    parser.add_argument("--vllm-port", type=int, default=8100)
    parser.add_argument("--vllm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()
    kwargs = {
        "task_dir": Path(args.task_dir),
        "input_dir": Path(args.input_dir),
        "host": args.vllm_host,
        "port": args.vllm_port,
        "model": args.vllm_model,
    }
    if args.mode == "generative":
        run_generative_preflight(**kwargs)
    elif args.mode == "multiview":
        run_multiview_preflight(**kwargs)
    else:
        if not args.panorama:
            parser.error("--panorama is required for panorama mode")
        run_post_panorama_preflight(
            Path(args.task_dir),
            Path(args.panorama),
            host=args.vllm_host,
            port=args.vllm_port,
            model=args.vllm_model,
        )


if __name__ == "__main__":
    main()
