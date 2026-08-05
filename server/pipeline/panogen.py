"""Panogen-specific utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Any


def prepare_panogen_input(task_dir: Path, input_image: Optional[Path], prompt: Optional[str]) -> dict[str, Any]:
    """Prepare input specification for HY-Pano-2 panorama generation.

    Returns a dict with input_image path and optional prompt.
    """
    spec: dict[str, Any] = {}

    if input_image and input_image.exists():
        spec["input_image"] = str(input_image)

    if prompt:
        spec["prompt"] = prompt

    return spec
