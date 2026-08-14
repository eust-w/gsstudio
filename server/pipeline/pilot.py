"""Independent 500-step 3DGS pilot gate for v4 Worldgen tasks."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .preflight import CONTRACT_VERSION, write_json_atomic


MINIMUM_PILOT_RENDERS = 4
MINIMUM_PILOT_PSNR = 22.0
MINIMUM_PILOT_SSIM = 0.75
MAXIMUM_PILOT_LPIPS = 0.25
MINIMUM_PILOT_GAUSSIANS = 50_000


class PilotValidationError(RuntimeError):
    """Pilot output indicates a content-quality failure."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[1])
            if line == "end_header":
                break
    return 0


def validate_pilot_output(task_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    output = output_dir or task_dir / "pilot_output"
    stat_files = sorted((output / "stats").glob("val_step*.json"))
    metrics = _read_json(stat_files[-1]) if stat_files else {}
    values: dict[str, float | None] = {}
    for key in ("psnr", "ssim", "lpips"):
        try:
            value = float(metrics[key])
            values[key] = value if math.isfinite(value) else None
        except (KeyError, TypeError, ValueError):
            values[key] = None
    renders = sorted((output / "renders").glob("val_step*.png"))
    plys = sorted(output.rglob("*.ply"))
    checkpoints = sorted((output / "ckpts").glob("ckpt_*.pt"))
    gaussian_count = _ply_vertex_count(plys[-1]) if plys else 0
    finite_bounds = False
    extent = None
    if plys:
        try:
            import numpy as np
            import trimesh

            cloud = trimesh.load(str(plys[-1]), process=False)
            vertices = np.asarray(cloud.vertices, dtype=np.float64)
            if vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices):
                span = vertices.max(axis=0) - vertices.min(axis=0)
                ordered_extent = np.sort(span)
                finite_bounds = bool(
                    np.isfinite(vertices).all()
                    and np.isfinite(span).all()
                    and ordered_extent[-1] > 1e-6
                    and ordered_extent[-2] > ordered_extent[-1] * 1e-3
                )
                extent = [float(value) for value in span]
        except Exception:
            finite_bounds = False

    reasons: list[str] = []
    if len(renders) < MINIMUM_PILOT_RENDERS:
        reasons.append("pilot_validation_render_count_below_4")
    if values["psnr"] is None or values["psnr"] < MINIMUM_PILOT_PSNR:
        reasons.append("pilot_psnr_below_22")
    if values["ssim"] is None or values["ssim"] < MINIMUM_PILOT_SSIM:
        reasons.append("pilot_ssim_below_0.75")
    if values["lpips"] is None or values["lpips"] > MAXIMUM_PILOT_LPIPS:
        reasons.append("pilot_lpips_above_0.25")
    if gaussian_count < MINIMUM_PILOT_GAUSSIANS:
        reasons.append("pilot_gaussian_count_below_50000")
    if not finite_bounds:
        reasons.append("pilot_spatial_bounds_degenerate_or_non_finite")
    if not checkpoints:
        reasons.append("pilot_checkpoint_missing")

    report = {
        "version": CONTRACT_VERSION,
        "state": "passed" if not reasons else "failed",
        "evaluation_only_checkpoint": True,
        "full_training_starts_from_scratch": True,
        "validation_render_count": len(renders),
        "metrics": values,
        "gaussian_count": gaussian_count,
        "finite_non_degenerate_bounds": finite_bounds,
        "spatial_extent": extent,
        "checkpoint_path": str(checkpoints[-1].relative_to(task_dir)) if checkpoints else None,
        "ply_path": str(plys[-1].relative_to(task_dir)) if plys else None,
        "thresholds": {
            "steps": 500,
            "minimum_validation_renders": MINIMUM_PILOT_RENDERS,
            "minimum_psnr": MINIMUM_PILOT_PSNR,
            "minimum_ssim": MINIMUM_PILOT_SSIM,
            "maximum_lpips": MAXIMUM_PILOT_LPIPS,
            "minimum_gaussians": MINIMUM_PILOT_GAUSSIANS,
        },
        "failure_reasons": reasons,
    }
    write_json_atomic(task_dir / "pilot_validation.json", report)
    if reasons:
        raise PilotValidationError("pilot_content_failed: " + ", ".join(reasons))
    return report
