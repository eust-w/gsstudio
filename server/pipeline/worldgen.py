"""World Generation pipeline orchestrator (panogen + 5-stage worldgen)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .quality import normalize_worldgen_config


WORLDGEN_STAGES = [
    {"name": "panogen", "description": "Panorama generation (HY-Pano-2 80B)", "gpu": "0,1,2,3,4"},
    {"name": "traj_generate", "description": "Trajectory planning (WorldNav)", "gpu": "0"},
    {"name": "traj_render", "description": "Trajectory rendering", "gpu": "0,1,2,3,4"},
    {"name": "video_gen", "description": "World expansion (WorldStereo-2)", "gpu": "0,1,2,3,4"},
    {"name": "gen_gs_data", "description": "GS data preparation", "gpu": "0,1,2,3,4"},
    {"name": "gs_training", "description": "3DGS training", "gpu": "0,1,2,3,4"},
]

# Stages for worldgen_from_pano (skip panogen)
WORLDGEN_FROM_PANO_STAGES = WORLDGEN_STAGES[1:]


def get_stage_names(task_type: str) -> list[str]:
    if task_type == "worldgen":
        return [s["name"] for s in WORLDGEN_STAGES]
    elif task_type == "worldgen_from_pano":
        return [s["name"] for s in WORLDGEN_FROM_PANO_STAGES]
    elif task_type == "worldrecon":
        return ["worldrecon"]
    return []


def get_worldgen_commands(
    task_dir: Path,
    scene_dir: Path,
    input_dir: Path,
    config: dict[str, Any],
    hyworld_repo: Path,
    vllm_host: str,
    vllm_port: int,
    vllm_model: str,
    task_type: str = "worldgen",
) -> list[tuple[str, list[str], dict[str, str], Path]]:
    """Build command lists for each worldgen stage.

    Returns list of (stage_name, cmd, env_overrides, cwd).
    """
    config = normalize_worldgen_config(config, form_prompt=None)
    worldgen_dir = hyworld_repo / "hyworld2" / "worldgen"
    panogen_dir = hyworld_repo / "hyworld2" / "panogen"
    quality_preset = config["quality_preset"]
    model_type = config["worldstereo_model_type"]
    # High quality is deliberately invariant so its output is comparable over
    # time and matches this deployment's five-A800 capacity.
    num_gpus = 5 if quality_preset == "high" else int(config.get("num_gpus", 5))
    gpu_list = ",".join(str(i) for i in range(num_gpus))
    # WorldStereo has 16 attention heads, so sequence-parallel world size must
    # divide 16. Use the largest valid subset of the available GPUs.
    video_num_gpus = max(size for size in (1, 2, 4, 8, 16) if size <= num_gpus)
    video_gpu_list = ",".join(str(i) for i in range(video_num_gpus))

    if quality_preset == "high":
        max_steps = 2000
        refine_start = 200
        refine_stop = 1000
        refine_every = 133
        refine_scale2d_stop = 1000
    else:
        # Fast DMD keeps the former resource-scaled training schedule.
        step_factor = 8.0 / num_gpus
        max_steps = int(1500 * step_factor)
        refine_start = int(150 * step_factor)
        refine_stop = int(750 * step_factor)
        refine_every = int(100 * step_factor)
        refine_scale2d_stop = int(750 * step_factor)

    commands = []

    # Stage 0: Panogen (only for worldgen type)
    if task_type == "worldgen":
        input_image = input_dir / config.get("input_image", "input.png")
        pano_output = scene_dir / "panorama.png"
        panogen_cmd = [
            config.get("panogen_python", "/opt/panogen/bin/python"),
            str(panogen_dir / "pipeline.py"),
            "--image", str(input_image),
            "--save", str(pano_output),
            "--attn-impl", "sdpa",
        ]
        if config.get("prompt"):
            panogen_cmd.extend(["--prompt", config["prompt"]])
        if config.get("seed"):
            panogen_cmd.extend(["--seed", str(config["seed"])])
        panogen_env = {
            "CUDA_VISIBLE_DEVICES": gpu_list,
            "HF_HOME": config.get("hf_home", "/models/hf-cache"),
        }
        commands.append(("panogen", panogen_cmd, panogen_env, panogen_dir))

    # Stage 1: Trajectory generation
    traj_gen_cmd = [
        "python", str(worldgen_dir / "traj_generate.py"),
        "--target_path", str(scene_dir),
        "--llm_addr", vllm_host,
        "--llm_port", str(vllm_port),
        "--llm_name", vllm_model,
        "--apply_nav_traj",
        "--apply_up_route",
        "--apply_recon_iteration",
        "--force_vlm",
    ]
    traj_gen_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": config.get("hf_home", "/models/hf-cache"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    commands.append(("traj_generate", traj_gen_cmd, traj_gen_env, worldgen_dir))

    # Stage 2: Trajectory rendering (multi-GPU)
    traj_render_cmd = [
        "torchrun", "--nproc_per_node", str(num_gpus),
        str(worldgen_dir / "traj_render.py"),
        "--target_path", str(scene_dir),
        "--llm_addr", vllm_host,
        "--llm_port", str(vllm_port),
        "--llm_name", vllm_model,
    ]
    traj_render_env = {
        "CUDA_VISIBLE_DEVICES": gpu_list,
        "HF_HOME": config.get("hf_home", "/models/hf-cache"),
    }
    commands.append(("traj_render", traj_render_cmd, traj_render_env, worldgen_dir))

    # Stage 3: Video generation (WorldStereo-2, multi-GPU FSDP)
    video_gen_cmd = [
        "torchrun", "--nproc_per_node", str(video_num_gpus),
        str(worldgen_dir / "video_gen.py"),
        "--target_path", str(scene_dir),
        "--fsdp",
        "--local_files_only",
        "--model_type", model_type,
    ]
    if quality_preset == "high":
        # The automatic recovery journal writes a bounded, regular-first
        # whitelist before each full WorldStereo invocation. `--skip_exist`
        # resumes an interrupted batch without regenerating completed videos.
        video_gen_cmd.extend([
            "--trajectory_whitelist", str(task_dir / "video_trajectory_whitelist.json"),
            "--skip_exist",
        ])
    video_gen_env = {
        "CUDA_VISIBLE_DEVICES": video_gpu_list,
        "HF_HOME": config.get("hf_home", "/models/hf-cache"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    commands.append(("video_gen", video_gen_cmd, video_gen_env, worldgen_dir))

    # Stage 4: GS data preparation (multi-GPU)
    gen_gs_cmd = [
        "torchrun", "--nproc_per_node", str(num_gpus),
        str(worldgen_dir / "gen_gs_data.py"),
        "--root_path", str(scene_dir),
        "--result_name", model_type,
        "--save_normal",
        "--split_sky",
    ]
    if quality_preset == "high":
        gen_gs_cmd.extend(["--trajectory_whitelist", str(task_dir / "trajectory_whitelist.json")])
    gen_gs_env = {
        "CUDA_VISIBLE_DEVICES": gpu_list,
        "HF_HOME": config.get("hf_home", "/models/hf-cache"),
    }
    commands.append(("gen_gs_data", gen_gs_cmd, gen_gs_env, worldgen_dir))

    # Stage 5: 3DGS training
    output_dir = task_dir / "output"
    gs_train_cmd = [
        "python", "-m", "world_gs_trainer", "default",
        "--data_dir", str(scene_dir / "gs_data"),
        "--result_dir", str(output_dir),
        "--max_steps", str(max_steps),
        "--save_steps", str(max_steps),
        "--eval_steps", str(max_steps),
        "--ply_steps", str(max_steps),
        "--save_ply",
        "--convert_to_spz",
        "--disable_video",
        "--use_scale_regularization",
        "--antialiased",
        "--depth_loss",
        "--normal_loss",
        "--sky_depth_from_pcd",
        "--use_mask_gaussian",
        "--mask_export_stochastic",
        "--no-mask-export-anchor-protection",
        "--use_anchor_protection",
        "--export_mesh",
        "--strategy.refine-start-iter", str(refine_start),
        "--strategy.refine-stop-iter", str(refine_stop),
        "--strategy.refine-every", str(refine_every),
        "--strategy.refine-scale2d-stop-iter", str(refine_scale2d_stop),
        "--strategy.reset-every", "99990",
        "--strategy.grow-grad2d", "0.0001",
        "--strategy.prune-scale3d", "0.1",
    ]
    gs_train_env = {
        "CUDA_VISIBLE_DEVICES": gpu_list,
        "HF_HOME": config.get("hf_home", "/models/hf-cache"),
        # The perceptual-loss checkpoint is required by the trainer.  Keep it
        # on the persistent model volume so every worker starts offline.
        "TORCH_HOME": config.get("torch_home", "/models/torch-cache"),
    }
    commands.append(("gs_training", gs_train_cmd, gs_train_env, worldgen_dir))

    return commands
