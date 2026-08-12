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


def _valid_checkpoint(
    path: Path,
    report: dict[str, Any],
    *,
    mode: str,
    seed: int,
    source_attempt: int,
) -> bool:
    if not path.is_file():
        return False
    if report.get("mode") != mode or int(report.get("seed", -1)) != seed:
        return False
    if int(report.get("source_attempt", -1)) != source_attempt:
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
) -> dict[str, Any]:
    """Generate/reuse a source image, then expand it with the same model."""
    started = time.monotonic()
    try:
        prior = json.loads(report_path.read_text()) if report_path.is_file() else {}
    except (json.JSONDecodeError, OSError):
        prior = {}
    report: dict[str, Any] = {
        "version": 1,
        "mode": input_mode,
        "state": "generating_source" if input_mode == "prompt" else "source_ready",
        "source_attempt": source_attempt,
        "seed": seed,
        "artifact_path": str(input_image.relative_to(report_path.parent)),
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
        )
        if input_mode == "prompt" and not reusable:
            _, samples = pipeline.model.generate_image(
                prompt=prompt,
                image=None,
                seed=seed,
                image_size="auto",
                use_system_prompt="en_unified",
                bot_task="think_recaption",
                diff_infer_steps=50,
                verbose=2,
                max_new_tokens=2048,
            )
            _atomic_image(samples[0], input_image)
        elif input_mode != "image" and input_mode != "prompt":
            raise ValueError(f"Unsupported input mode: {input_mode}")
        elif not input_image.is_file():
            raise FileNotFoundError(f"Input image does not exist: {input_image}")

        with Image.open(input_image) as source:
            width, height = source.size
        report.update({
            "state": "expanding_panorama",
            "sha256": _sha256(input_image),
            "width": width,
            "height": height,
        })
        _atomic_json(report_path, report)

        panorama = pipeline(str(input_image), prompt=prompt, seed=seed)
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
    args = parser.parse_args()

    sys.path.insert(0, args.panogen_dir)
    from pipeline import HunyuanPanoPipeline

    model_root = os.environ.get("HY_PANO_MODEL", "tencent/HY-World-2.0")
    pipeline = HunyuanPanoPipeline.from_pretrained(model_root, attn_impl="sdpa")
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
