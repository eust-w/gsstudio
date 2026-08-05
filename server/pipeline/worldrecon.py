"""World Reconstruction pipeline (WorldMirror-2)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def get_worldrecon_commands(
    task_dir: Path,
    input_dir: Path,
    config: dict[str, Any],
    hyworld_repo: Path,
) -> list[tuple[str, list[str], dict[str, str], Path]]:
    """Build command list for WorldMirror-2 reconstruction.

    Returns list with single (stage_name, cmd, env_overrides, cwd).
    """
    output_dir = task_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "hyworld2.worldrecon.pipeline",
        "--input_path", str(input_dir),
        "--output_path", str(output_dir),
        "--enable_bf16",
        "--save_gs",
        "--save_points",
        "--save_camera",
        "--save_depth",
        "--save_normal",
    ]

    # Optional: multi-GPU FSDP for large image sets
    num_gpus = config.get("recon_gpus", 1)
    if num_gpus > 1:
        cmd = [
            "torchrun", "--nproc_per_node", str(num_gpus),
            "-m", "hyworld2.worldrecon.pipeline",
            "--input_path", str(input_dir),
            "--output_path", str(output_dir),
            "--use_fsdp",
            "--enable_bf16",
            "--save_gs",
            "--save_points",
            "--save_camera",
            "--save_depth",
            "--save_normal",
        ]

    gpu_ids = ",".join(str(i) for i in range(num_gpus))
    env = {
        "CUDA_VISIBLE_DEVICES": gpu_ids,
        "HF_HOME": config.get("hf_home", "/models/hf-cache"),
    }

    return [("worldrecon", cmd, env, hyworld_repo)]
