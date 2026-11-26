"""Read-only dependency preflight for the real HY-World execution runtime.

These checks never download models, allocate a training job, or invent GPU
telemetry. Passing preflight establishes prerequisites, not output quality;
the pipeline's input, model-loading, and delivery gates remain authoritative.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.request

from fastapi import HTTPException

from .config import Config

TASK_TYPES = ("worldgen", "worldgen_from_pano", "worldrecon")
_cache_lock = threading.Lock()
_cache: dict[tuple, tuple[float, dict]] = {}
_CACHE_SECONDS = 15


def clear_runtime_cache() -> None:
    with _cache_lock:
        _cache.clear()


def probe_gpus() -> dict[str, Any]:
    """Report only NVML telemetry observed on this server, including in demo."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"gpus": [], "available": False, "error": "NVIDIA GPU telemetry is unavailable: nvidia-smi was not found."}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, check=False,
        )
        if result.returncode:
            raise RuntimeError("nvidia-smi could not communicate with the GPU driver")
        gpus = []
        for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            if not row or all(not item.strip() for item in row):
                continue
            if len(row) != 4:
                raise ValueError("unexpected GPU telemetry format")
            index, name, total, used = [item.strip() for item in row]
            total_gb, used_gb = float(total) / 1024, float(used) / 1024
            if not all(math.isfinite(value) and value >= 0 for value in (total_gb, used_gb)):
                raise ValueError("invalid GPU memory telemetry")
            gpus.append({"index": int(index), "name": name, "vram_total_gb": total_gb,
                         "vram_used_gb": used_gb, "used_by": None})
        return {"gpus": gpus, "available": bool(gpus), "error": None if gpus else "No NVIDIA GPUs were detected."}
    except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError):
        return {"gpus": [], "available": False, "error": "NVIDIA GPU telemetry failed; check the driver and device access."}


_PYTHON_PROBE = """
import importlib, json
names = ['torch', 'numpy', 'PIL', 'cv2', 'gsplat.rendering', 'plyfile', 'spz', 'trimesh', 'torchmetrics.functional.image.lpips']
missing = []
for name in names:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
try:
    import torch
    cuda_available = torch.cuda.is_available()
    cuda_devices = torch.cuda.device_count() if cuda_available else 0
except Exception:
    cuda_available, cuda_devices = False, 0
try:
    import rtree
    rtree_available = True
except Exception:
    rtree_available = False
try:
    import torch.distributed.run
    distributed_available = True
except Exception:
    distributed_available = False
print(json.dumps(dict(missing=missing, cuda_available=cuda_available, cuda_devices=cuda_devices, rtree_available=rtree_available, distributed_available=distributed_available)))
"""


def _probe_python() -> dict[str, Any]:
    executable = sys.executable
    try:
        result = subprocess.run([executable, "-c", _PYTHON_PROBE], capture_output=True, text=True, timeout=20, check=False)
        if result.returncode:
            raise ValueError("Python probe failed")
        value = json.loads(result.stdout.strip().splitlines()[-1])
        return value
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return {"error": "The pipeline Python dependency probe failed or timed out.", "missing": [], "cuda_devices": 0}


def _probe_vlm(config: Config) -> dict[str, Any]:
    url = f"http://{config.VLLM_HOST}:{config.VLLM_PORT}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            data = json.loads(response.read(1024 * 1024))
        ids = {item.get("id") for item in data.get("data", []) if isinstance(item, dict)}
        if config.VLLM_MODEL not in ids:
            return {"ok": False, "message": "The VLM service does not serve the configured preflight model."}
        return {"ok": True, "message": f"VLM service is reachable and serves {config.VLLM_MODEL}."}
    except (OSError, ValueError, AttributeError, TypeError):
        return {"ok": False, "message": "The VLM preflight service is unreachable or returned an invalid model list."}


def _probe_panorama(config: Config) -> dict[str, Any]:
    executable = shutil.which(config.PANOGEN_PYTHON)
    directory = Path(config.HYWORLD_REPO).expanduser().resolve() / "hyworld2/panogen"
    if not executable or not (directory / "pipeline.py").is_file():
        return {"ok": False, "prompt_supported": False,
                "message": "Panorama generation requires GSSTUDIO_PANOGEN_PYTHON and hyworld2/panogen/pipeline.py."}
    source = """
import inspect, json, sys
sys.path.insert(0, sys.argv[1])
from pipeline import HunyuanPanoPipeline
from hunyuan_image_3.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM
parameter = inspect.signature(HunyuanImage3ForCausalMM.generate_image).parameters.get('image')
print(json.dumps({'ok': True, 'prompt_supported': parameter is not None and parameter.default is None}))
"""
    try:
        result = subprocess.run([executable, "-c", source, str(directory)], capture_output=True,
                                text=True, timeout=20, check=False)
        if result.returncode:
            raise ValueError("Panorama dependency import failed")
        value = json.loads(result.stdout.strip().splitlines()[-1])
        return {**value, "message": "The panorama interpreter imports its inference pipeline successfully."}
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return {"ok": False, "prompt_supported": False,
                "message": "The panorama interpreter could not import its inference dependencies within 20 seconds."}


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _model_roots(hf_home: str, repo_id: str) -> list[Path]:
    hub = Path(hf_home) / "hub"
    slug = repo_id.replace("/", "--")
    snapshots = hub / f"models--{slug}" / "snapshots"
    roots = [hub / slug]
    if snapshots.is_dir():
        roots.extend(path for path in snapshots.iterdir() if path.is_dir())
    return [path for path in roots if path.is_dir()]


def _has_weights(root: Path) -> bool:
    """Check referenced shards as well as single-file weights without loading."""
    for index in root.rglob("*.index.json"):
        try:
            value = json.loads(index.read_text())
            shards = value.get("weight_map", {}).values()
            if shards and not all(_nonempty(index.parent / name) for name in set(shards)):
                return False
        except (OSError, ValueError, AttributeError, TypeError):
            return False
    return any(_nonempty(path) for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt") for path in root.rglob(pattern))


def _check(code: str, label: str, ok: bool, message: str, tasks: tuple[str, ...] = TASK_TYPES) -> dict:
    return {"code": code, "label": label, "ok": bool(ok), "message": message, "task_types": list(tasks)}


def _probe_runtime(config: Config) -> dict:
    with ThreadPoolExecutor(max_workers=4) as pool:
        gpu_future = pool.submit(probe_gpus)
        python_future = pool.submit(_probe_python)
        vlm_future = pool.submit(_probe_vlm, config)
        panorama_future = pool.submit(_probe_panorama, config)
        gpu, python, vlm = gpu_future.result(), python_future.result(), vlm_future.result()
        panorama = panorama_future.result()

    checks = []
    generation = ("worldgen", "worldgen_from_pano")
    repo = Path(config.HYWORLD_REPO)
    checks.append(_check("gpu_telemetry", "NVIDIA GPU", gpu["available"], gpu["error"] or f"Detected {len(gpu['gpus'])} NVIDIA GPU(s)."))
    missing = python.get("missing", [])
    checks.append(_check("python_dependencies", "Pipeline Python", not python.get("error") and not missing,
                         python.get("error") or ("Missing or unusable Python modules: " + ", ".join(missing) if missing else "Required pipeline Python modules import successfully.")))
    checks.append(_check("cuda_runtime", "PyTorch CUDA", python.get("cuda_available", False),
                         f"PyTorch reports {python.get('cuda_devices', 0)} accessible CUDA device(s)."))
    checks.append(_check("vlm_service", "Visual input preflight", vlm["ok"], vlm["message"]))
    checks.append(_check("worldrecon_code", "WorldMirror source", (repo / "hyworld2/worldrecon/pipeline.py").is_file(),
                         "HY-World reconstruction source must be installed at GSSTUDIO_HYWORLD_REPO.", ("worldrecon",)))
    required = ("traj_generate.py", "traj_render.py", "video_gen.py", "gen_gs_data.py")
    absent = [name for name in required if not (repo / "hyworld2/worldgen" / name).is_file()]
    checks.append(_check("worldgen_code", "World generation source", not absent,
                         "Missing HY-World generation scripts: " + ", ".join(absent) if absent else "HY-World generation scripts are installed.", generation))
    checks.append(_check("distributed_launcher", "Distributed GPU launcher", python.get("distributed_available", False),
                         "torch.distributed.run must be importable in the pipeline environment.", generation))
    checks.append(_check("spatial_index", "Trajectory spatial index", python.get("rtree_available", False),
                         "The rtree module is required by trajectory planning.", generation))
    checks.append(_check("panorama_runtime", "Panorama runtime", panorama["ok"], panorama["message"], ("worldgen",)))
    checks.append(_check("prompt_source", "Text source generation", panorama["prompt_supported"],
                         "Text source generation requires the image-optional HunyuanImage-3 generation interface.", ()))
    writable = Path(config.DATA_DIR).absolute()
    while not writable.exists() and writable != writable.parent:
        writable = writable.parent
    checks.append(_check("task_storage", "Task storage", writable.is_dir() and os.access(writable, os.W_OK | os.X_OK),
                         "The task data directory or its existing parent must be writable."))

    hy_roots = _model_roots(config.HF_HOME, "tencent/HY-World-2.0")
    checks.append(_check("hyworld_weights", "HY-World model cache", any(_has_weights(root) for root in hy_roots),
                         "HY-World model weights and every indexed shard must be present in GSSTUDIO_HF_HOME; preflight does not download them."))
    sam_roots = _model_roots(config.HF_HOME, "facebook/sam3")
    checks.append(_check("sam3_weights", "SAM3 model cache", any(_nonempty(root / "config.json") and _nonempty(root / "model.safetensors") for root in sam_roots),
                         "Cache the approved facebook/sam3 model before submitting world generation.", generation))
    stereo_roots = _model_roots(config.HF_HOME, "hanshanxue/WorldStereo")
    checks.append(_check("worldstereo_weights", "WorldStereo model cache", any(_nonempty(root / "worldstereo-memory/config.json") and _has_weights(root / "worldstereo-memory") for root in stereo_roots),
                         "The full worldstereo-memory checkpoint must be cached for strict world generation.", generation))
    wan_roots = _model_roots(config.HF_HOME, "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    checks.append(_check("wan_weights", "WorldStereo base model", any(_nonempty(root / "scheduler/scheduler_config.json") and all(_has_weights(root / part) for part in ("transformer", "text_encoder", "image_encoder", "vae")) for root in wan_roots),
                         "The Wan model requires transformer, text encoder, image encoder, VAE weights and scheduler configuration.", generation))
    for repo_id, code in (("Ruicheng/moge-2-vitl-normal", "moge_weights"), ("facebook/dinov2-base", "dinov2_weights"), ("naver-iv/zim-anything-vitl", "zim_weights"), ("IDEA-Research/grounding-dino-tiny", "grounding_dino_weights")):
        checks.append(_check(code, repo_id, any(_has_weights(root) for root in _model_roots(config.HF_HOME, repo_id)),
                             f"Cache {repo_id} before submitting world generation.", generation))
    torch_home = Path(os.environ.get("TORCH_HOME", "/models/torch-cache"))
    checks.append(_check("perceptual_weights", "Perceptual validation model", _nonempty(torch_home / "hub/checkpoints/vgg16-397923af.pth"),
                         "The VGG16 checkpoint must be cached under TORCH_HOME for offline delivery validation."))
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "checks": checks, "gpu": gpu,
            "cuda_devices": int(python.get("cuda_devices", 0)),
            "distributed_available": python.get("distributed_available", False), "python": python}


def _runtime_snapshot(config: Config, *, refresh: bool = False) -> dict:
    key = (str(config.HYWORLD_REPO), config.PANOGEN_PYTHON, config.HF_HOME,
           str(config.DATA_DIR), config.VLLM_HOST, config.VLLM_PORT, config.VLLM_MODEL,
           os.environ.get("PATH"), os.environ.get("TORCH_HOME"), os.environ.get("CUDA_VISIBLE_DEVICES"))
    with _cache_lock:
        cached = _cache.get(key)
        if not refresh and cached and time.monotonic() - cached[0] < _CACHE_SECONDS:
            return deepcopy(cached[1])
        snapshot = _probe_runtime(config)
        # Keep bounded state even when settings are changed in a long session.
        _cache.clear()
        _cache[key] = (time.monotonic(), snapshot)
        return deepcopy(snapshot)


def _capability(snapshot: dict, task_type: str, task_config: dict | None = None) -> dict:
    blockers = [{"code": check["code"], "message": check["message"]}
                for check in snapshot["checks"] if task_type in check["task_types"] and not check["ok"]]
    required_gpus = int((task_config or {}).get("recon_gpus", 1)) if task_type == "worldrecon" else 5
    available_gpus = min(len(snapshot["gpu"]["gpus"]), snapshot["cuda_devices"])
    if available_gpus < required_gpus:
        blockers.append({"code": "gpu_capacity", "message": f"This task requires {required_gpus} accessible CUDA GPU(s); {available_gpus} are available."})
    if task_type == "worldrecon" and required_gpus > 1 and not snapshot.get("distributed_available", False):
        blockers.append({"code": "distributed_launcher", "message": "Multi-GPU reconstruction requires torch.distributed.run in the pipeline environment."})
    if task_type == "worldgen" and (task_config or {}).get("input_mode") == "prompt":
        blockers.extend({"code": check["code"], "message": check["message"]} for check in snapshot["checks"]
                        if check["code"] == "prompt_source" and not check["ok"])
    return {"task_type": task_type, "ready": not blockers, "required_gpus": required_gpus, "blockers": blockers}


def get_system_status(config: Config, active_task: str | None = None, *, refresh: bool = False) -> dict:
    snapshot = _runtime_snapshot(config, refresh=refresh)
    capabilities = [_capability(snapshot, task_type) for task_type in TASK_TYPES]
    return {"mode": "demo" if config.MOCK_PIPELINE else "real",
            "ready": any(item["ready"] for item in capabilities),
            "checked_at": snapshot["checked_at"], "capabilities": capabilities,
            "checks": snapshot["checks"], "gpu": {**snapshot["gpu"], "active_task": active_task},
            "repair": _repair_capability(snapshot),
            "verification_scope": "dependency_preflight",
            "message": "Demo mode runs synthetic tasks; generated results are not real reconstructions." if config.MOCK_PIPELINE else "Runtime prerequisites are checked before submission. Input and delivery quality are validated during execution."}


def require_task_capability(config: Config, task_type: str, task_config: dict | None = None) -> None:
    if config.MOCK_PIPELINE:
        return
    if task_type not in TASK_TYPES:
        raise HTTPException(400, "Unsupported task type")
    capability = _capability(_runtime_snapshot(config), task_type, task_config)
    if not capability["ready"]:
        raise HTTPException(503, detail={"code": "runtime_not_ready", "message": "The real execution runtime is not ready for this task.", "blockers": capability["blockers"]})


def _repair_capability(snapshot: dict) -> dict:
    codes = {"gpu_telemetry", "cuda_runtime", "task_storage", "perceptual_weights"}
    blockers = [{"code": item["code"], "message": item["message"]}
                for item in snapshot["checks"] if item["code"] in codes and not item["ok"]]
    python = snapshot.get("python")
    if python is None:
        blockers.extend({"code": item["code"], "message": item["message"]}
                        for item in snapshot["checks"] if item["code"] == "python_dependencies" and not item["ok"])
    else:
        missing = [module for module in python.get("missing", []) if module != "cv2"]
        if missing or python.get("error"):
            blockers.append({"code": "repair_dependencies", "message": python.get("error") or
                             "Missing or unusable repair modules: " + ", ".join(missing)})
    if not snapshot["gpu"]["gpus"] or snapshot["cuda_devices"] < 1:
        blockers.append({"code": "gpu_capacity", "message": "Model repair requires one accessible CUDA GPU."})
    return {"ready": not blockers, "required_gpus": 1, "blockers": blockers}


def require_repair_capability(config: Config) -> None:
    """Admit existing-model repair without requiring source inference models."""
    if config.MOCK_PIPELINE:
        return
    capability = _repair_capability(_runtime_snapshot(config))
    if not capability["ready"]:
        raise HTTPException(503, detail={"code": "runtime_not_ready", "message": "The model repair runtime is not ready.", "blockers": capability["blockers"]})
