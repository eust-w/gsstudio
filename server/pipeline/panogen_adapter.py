"""Single-load HunyuanImage-3 source and panorama generation adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_SOURCE_IMAGE_HEIGHT = 832
DEFAULT_SOURCE_IMAGE_WIDTH = 1216
DEFAULT_SOURCE_CANDIDATE_COUNT = 4
SOURCE_PROMPT_MODE = "direct_image_v1"
WORLD_GEOMETRY_PROMPT_SUFFIX = (
    "写实的矩形现代机器人研发办公室，横向构图；"
    "使用24毫米移轴建筑摄影镜头的直线投影，水平视角约70度，"
    "非全景、非360度、非鱼眼、非超广角；"
    "四面墙均为笔直平面且互相垂直，平直天花板与平整地面，"
    "禁止拱门、圆柱、弧形隔断和曲面吊顶；"
    "清晰稳定的桌椅布局和宽阔可行走地面；"
    "一致自然光照，无人物、无窗墙、无玻璃门、无玻璃柜、无镜面、"
    "无玻璃材质、无玻璃反射、无反光地面；"
    "所有物体具有明确边界、合理尺度和一致透视；"
    "使用校正后的直线建筑摄影，无鱼眼畸变、无弯曲墙面，所有表面采用不透明哑光材质，"
    "适合多视角扩展与三维重建。"
)


def expand_worldgen_prompt(prompt: str) -> str:
    """Add deterministic geometry guidance without replacing user intent."""
    base = prompt.strip()
    if not base:
        return WORLD_GEOMETRY_PROMPT_SUFFIX
    if WORLD_GEOMETRY_PROMPT_SUFFIX in base:
        return base
    return f"{base}\n\n{WORLD_GEOMETRY_PROMPT_SUFFIX}"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _utcnow()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def _atomic_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.png")
    image.convert("RGB").save(temporary, format="PNG")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path_for_report(path: Path, report_path: Path) -> str:
    """Return a stable task-relative path even for recovery reports.

    Recovery reports live below ``recovery_attempts/attempt-NNN`` while an
    uploaded source may remain in the canonical task ``input`` directory.
    Walk upward to their nearest shared task ancestor instead of emitting
    ``..`` segments or raising ``ValueError``.
    """
    resolved = path.resolve()
    report_parent = report_path.parent.resolve()
    for ancestor in (report_parent, *report_parent.parents):
        if ancestor == Path(ancestor.anchor):
            break
        try:
            return str(resolved.relative_to(ancestor))
        except ValueError:
            continue
    return str(resolved)


def generate_source_pool(
    pipeline: Any | None,
    *,
    input_mode: str,
    input_image: Path,
    candidate_dir: Path,
    report_path: Path,
    prompt: str,
    base_seed: int,
    source_attempt: int,
    candidate_count: int = DEFAULT_SOURCE_CANDIDATE_COUNT,
    source_image_size: tuple[int, int] = (
        DEFAULT_SOURCE_IMAGE_HEIGHT,
        DEFAULT_SOURCE_IMAGE_WIDTH,
    ),
) -> dict[str, Any]:
    """Create the immutable v4 candidate pool before Qwen/CV preflight."""
    if candidate_count != DEFAULT_SOURCE_CANDIDATE_COUNT:
        raise ValueError("The strict source pool always contains exactly four candidates")
    source_height, source_width = source_image_size
    effective_prompt = expand_worldgen_prompt(prompt) if input_mode == "prompt" else prompt.strip()
    candidate_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "version": 4,
        "state": "generating",
        "input_lane": "generative_worldgen",
        "mode": input_mode,
        "source_attempt": source_attempt,
        "base_seed": base_seed,
        "source_prompt_mode": SOURCE_PROMPT_MODE if input_mode == "prompt" else None,
        "effective_prompt": effective_prompt,
        "requested_source_width": source_width,
        "requested_source_height": source_height,
        "candidates": [],
        "started_at": _utcnow(),
    }
    _atomic_json(report_path, report)
    try:
        if input_mode == "image":
            if not input_image.is_file():
                raise FileNotFoundError(f"Input image does not exist: {input_image}")
            with Image.open(input_image) as decoded:
                decoded.verify()
            report["candidates"].append({
                "id": "source-00",
                "path": str(input_image.relative_to(report_path.parent)),
                "seed": base_seed,
                "sha256": _sha256(input_image),
                "generation": "uploaded",
            })
        elif input_mode == "prompt":
            if pipeline is None:
                raise RuntimeError("Prompt source pool requires HunyuanImage-3")
            for index in range(candidate_count):
                seed = base_seed + index
                destination = candidate_dir / f"candidate-{index:02d}.png"
                _, samples = pipeline.model.generate_image(
                    prompt=effective_prompt,
                    image=None,
                    seed=seed,
                    image_size=(source_height, source_width),
                    use_system_prompt="dynamic",
                    bot_task="image",
                    diff_infer_steps=50,
                    verbose=2,
                )
                _atomic_image(samples[0], destination)
                with Image.open(destination) as decoded:
                    width, height = decoded.size
                if (height, width) != source_image_size:
                    raise RuntimeError(
                        f"Candidate {index} has {width}x{height}; expected {source_width}x{source_height}"
                    )
                report["candidates"].append({
                    "id": f"source-{index:02d}",
                    "path": str(destination.relative_to(report_path.parent)),
                    "seed": seed,
                    "sha256": _sha256(destination),
                    "generation": "hunyuan_image_3",
                })
                _atomic_json(report_path, report)
        else:
            raise ValueError(f"Unsupported input mode: {input_mode}")
        report.update({"state": "generated", "completed_at": _utcnow()})
        _atomic_json(report_path, report)
        return report
    except Exception as exc:
        report.update({
            "state": "failed",
            "completed_at": _utcnow(),
            "error_code": "source_pool_generation_failed",
            "error_summary": str(exc),
        })
        _atomic_json(report_path, report)
        raise


def _valid_checkpoint(
    path: Path,
    report: dict[str, Any],
    *,
    mode: str,
    seed: int,
    source_attempt: int,
    effective_prompt: str,
    source_image_size: tuple[int, int],
) -> bool:
    if not path.is_file():
        return False
    if report.get("mode") != mode or int(report.get("seed", -1)) != seed:
        return False
    if int(report.get("source_attempt", -1)) != source_attempt:
        return False
    if mode == "prompt":
        expected_height, expected_width = source_image_size
        if report.get("source_prompt_mode") != SOURCE_PROMPT_MODE:
            return False
        if report.get("effective_prompt") != effective_prompt:
            return False
        if int(report.get("requested_source_width", -1)) != expected_width:
            return False
        if int(report.get("requested_source_height", -1)) != expected_height:
            return False
    try:
        if report.get("sha256") != _sha256(path):
            return False
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def generate_source_and_panorama(
    pipeline: Any,
    *,
    input_mode: str,
    input_image: Path,
    panorama_path: Path,
    report_path: Path,
    prompt: str,
    seed: int,
    source_attempt: int,
    source_image_size: tuple[int, int] = (
        DEFAULT_SOURCE_IMAGE_HEIGHT,
        DEFAULT_SOURCE_IMAGE_WIDTH,
    ),
) -> dict[str, Any]:
    """Generate/reuse a source image, then expand it with the same model."""
    started = time.monotonic()
    try:
        prior = json.loads(report_path.read_text()) if report_path.is_file() else {}
    except (json.JSONDecodeError, OSError):
        prior = {}
    source_height, source_width = source_image_size
    if source_height <= 0 or source_width <= 0:
        raise ValueError("Source image dimensions must be positive")
    effective_prompt = expand_worldgen_prompt(prompt) if input_mode == "prompt" else prompt.strip()
    report: dict[str, Any] = {
        "version": 1,
        "mode": input_mode,
        "state": "generating_source" if input_mode == "prompt" else "source_ready",
        "source_attempt": source_attempt,
        "seed": seed,
        "original_prompt": prompt.strip(),
        "effective_prompt": effective_prompt,
        "source_prompt_mode": SOURCE_PROMPT_MODE if input_mode == "prompt" else None,
        "requested_source_width": source_width if input_mode == "prompt" else None,
        "requested_source_height": source_height if input_mode == "prompt" else None,
        "artifact_path": _artifact_path_for_report(input_image, report_path),
        "started_at": prior.get("started_at") or _utcnow(),
        "completed_at": None,
        "duration_seconds": 0,
        "error_code": None,
        "error_summary": None,
    }
    _atomic_json(report_path, report)

    try:
        reusable = _valid_checkpoint(
            input_image,
            prior,
            mode=input_mode,
            seed=seed,
            source_attempt=source_attempt,
            effective_prompt=effective_prompt,
            source_image_size=source_image_size,
        )
        if input_mode == "prompt" and not reusable:
            _, samples = pipeline.model.generate_image(
                prompt=effective_prompt,
                image=None,
                seed=seed,
                image_size=(source_height, source_width),
                # Generate from the deterministic geometry prompt directly.
                # Upstream recaptioning can invert hard constraints (for
                # example by adding glass walls and reflective floors).
                use_system_prompt="dynamic",
                bot_task="image",
                diff_infer_steps=50,
                verbose=2,
            )
            _atomic_image(samples[0], input_image)
        elif input_mode != "image" and input_mode != "prompt":
            raise ValueError(f"Unsupported input mode: {input_mode}")
        elif not input_image.is_file():
            raise FileNotFoundError(f"Input image does not exist: {input_image}")

        with Image.open(input_image) as source:
            width, height = source.size
        if input_mode == "prompt" and (height, width) != source_image_size:
            raise RuntimeError(
                "Generated source image has unexpected dimensions: "
                f"{width}x{height}; expected {source_width}x{source_height}"
            )
        report.update({
            "state": "expanding_panorama",
            "sha256": _sha256(input_image),
            "width": width,
            "height": height,
        })
        _atomic_json(report_path, report)

        panorama = pipeline(str(input_image), prompt=effective_prompt, seed=seed)
        _atomic_image(panorama, panorama_path)
        report.update({
            "state": "completed",
            "completed_at": _utcnow(),
            "duration_seconds": round(time.monotonic() - started, 3),
        })
        _atomic_json(report_path, report)
        return report
    except Exception as exc:
        friendly = (
            "The source image is missing or invalid."
            if isinstance(exc, (FileNotFoundError, ValueError))
            else "The source image could not be generated."
        )
        report.update({
            "state": "failed",
            "completed_at": _utcnow(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "error_code": "source_generation_failed",
            "error_summary": friendly,
        })
        _atomic_json(report_path, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panogen-dir", required=True)
    parser.add_argument("--input-mode", choices=("image", "prompt"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--source-attempt", type=int, default=0)
    parser.add_argument("--source-pool-only", action="store_true")
    parser.add_argument("--candidate-dir")
    parser.add_argument("--pool-report")
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_SOURCE_CANDIDATE_COUNT)
    args = parser.parse_args()

    pipeline = None
    if not args.source_pool_only or args.input_mode == "prompt":
        sys.path.insert(0, args.panogen_dir)
        from pipeline import HunyuanPanoPipeline

        model_root = os.environ.get("HY_PANO_MODEL", "tencent/HY-World-2.0")
        pipeline = HunyuanPanoPipeline.from_pretrained(model_root, attn_impl="sdpa")
    if args.source_pool_only:
        if not args.candidate_dir or not args.pool_report:
            parser.error("--source-pool-only requires --candidate-dir and --pool-report")
        generate_source_pool(
            pipeline,
            input_mode=args.input_mode,
            input_image=Path(args.image),
            candidate_dir=Path(args.candidate_dir),
            report_path=Path(args.pool_report),
            prompt=args.prompt,
            base_seed=args.seed,
            source_attempt=args.source_attempt,
            candidate_count=args.candidate_count,
        )
        return
    assert pipeline is not None
    generate_source_and_panorama(
        pipeline,
        input_mode=args.input_mode,
        input_image=Path(args.image),
        panorama_path=Path(args.save),
        report_path=Path(args.report),
        prompt=args.prompt,
        seed=args.seed,
        source_attempt=args.source_attempt,
    )


if __name__ == "__main__":
    main()
