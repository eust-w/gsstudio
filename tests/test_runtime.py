"""Runtime admission and telemetry must describe the machine, never a demo GPU."""
import json
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from server.config import Config
from server import runtime


@pytest.fixture
def runtime_config(tmp_path):
    cfg = Config()
    cfg.MOCK_PIPELINE = False
    cfg.DATA_DIR = tmp_path / "data"
    cfg.HYWORLD_REPO = tmp_path / "hyworld"
    cfg.HF_HOME = str(tmp_path / "models")
    cfg.PANOGEN_PYTHON = str(tmp_path / "panogen/python")
    runtime.clear_runtime_cache()
    yield cfg
    runtime.clear_runtime_cache()


def snapshot(*, gpu_count=5, checks=None):
    return {"checked_at": "2026-10-03T00:00:00+00:00", "checks": checks or [],
            "cuda_devices": gpu_count,
            "gpu": {"gpus": [{"index": i, "name": "Observed GPU", "vram_total_gb": 24,
                              "vram_used_gb": 1, "used_by": None} for i in range(gpu_count)],
                    "available": gpu_count > 0, "error": None}}


def test_gpu_unavailable_never_fabricates_hardware(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)
    observed = runtime.probe_gpus()
    assert observed["gpus"] == []
    assert observed["available"] is False
    assert "nvidia-smi" in observed["error"]


def test_gpu_observations_preserve_actual_memory_and_unknown_ownership(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='0, "NVIDIA, Test Device", 24576, 2048\n'))
    observed = runtime.probe_gpus()
    assert observed["gpus"][0] == {"index": 0, "name": "NVIDIA, Test Device", "vram_total_gb": 24,
                                   "vram_used_gb": 2, "used_by": None}


@pytest.mark.parametrize("value", ["NaN", "inf", "-1", "[Not Supported]"])
def test_gpu_invalid_telemetry_is_not_reported_as_healthy(monkeypatch, value):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=f"0, Test GPU, 24576, {value}\n"))
    assert runtime.probe_gpus()["available"] is False


def test_demo_mode_reports_actual_readiness_and_bypasses_admission(runtime_config, monkeypatch):
    runtime_config.MOCK_PIPELINE = True
    monkeypatch.setattr(runtime, "_probe_runtime", lambda _: snapshot(gpu_count=0))
    status = runtime.get_system_status(runtime_config)
    assert status["mode"] == "demo"
    assert status["ready"] is False
    assert status["gpu"]["gpus"] == []
    assert all(not item["ready"] for item in status["capabilities"])
    runtime.require_task_capability(runtime_config, "worldgen")


def test_capabilities_are_task_specific_and_admission_uses_requested_gpu_count(runtime_config, monkeypatch):
    monkeypatch.setattr(runtime, "_probe_runtime", lambda _: snapshot(gpu_count=1, checks=[
        runtime._check("panorama_runtime", "Panorama", False, "Missing panorama runtime", ("worldgen",))]))
    status = runtime.get_system_status(runtime_config)
    assert status["ready"] is True
    by_type = {item["task_type"]: item for item in status["capabilities"]}
    assert by_type["worldrecon"]["ready"] is True
    assert by_type["worldgen"]["ready"] is False
    runtime.require_task_capability(runtime_config, "worldrecon", {"recon_gpus": 1})
    with pytest.raises(HTTPException) as caught:
        runtime.require_task_capability(runtime_config, "worldrecon", {"recon_gpus": 2})
    assert caught.value.status_code == 503
    assert caught.value.detail["code"] == "runtime_not_ready"
    assert any(item["code"] == "gpu_capacity" for item in caught.value.detail["blockers"])


def test_cached_snapshots_are_isolated_and_refreshable(runtime_config, monkeypatch):
    calls = []
    def probe(_):
        calls.append(True)
        return snapshot()
    monkeypatch.setattr(runtime, "_probe_runtime", probe)
    first = runtime.get_system_status(runtime_config, "task-1")
    first["gpu"]["gpus"].clear()
    second = runtime.get_system_status(runtime_config, "task-2")
    assert len(calls) == 1
    assert len(second["gpu"]["gpus"]) == 5
    assert second["gpu"]["active_task"] == "task-2"
    runtime.get_system_status(runtime_config, refresh=True)
    assert len(calls) == 2


def test_partial_or_dangling_model_cache_is_not_ready(tmp_path):
    (tmp_path / "model-00001.safetensors").write_bytes(b"first shard")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.1": "model-00001.safetensors", "layer.2": "model-00002.safetensors"}}))
    assert runtime._has_weights(tmp_path) is False
    (tmp_path / "model-00002.safetensors").symlink_to(tmp_path / "missing-blob")
    assert runtime._has_weights(tmp_path) is False
    (tmp_path / "missing-blob").write_bytes(b"second shard")
    assert runtime._has_weights(tmp_path) is True


def test_vlm_must_serve_configured_model(runtime_config, monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, size): return b'{"data":[{"id":"different-model"}]}'
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **k: Response())
    assert runtime._probe_vlm(runtime_config)["ok"] is False


def test_prompt_interface_only_blocks_text_tasks(runtime_config, monkeypatch):
    monkeypatch.setattr(runtime, "_probe_runtime", lambda _: snapshot(checks=[
        runtime._check("prompt_source", "Prompt", False, "Missing prompt interface", ())]))
    runtime.require_task_capability(runtime_config, "worldgen", {"input_mode": "image"})
    with pytest.raises(HTTPException) as caught:
        runtime.require_task_capability(runtime_config, "worldgen", {"input_mode": "prompt"})
    assert caught.value.detail["blockers"] == [{"code": "prompt_source", "message": "Missing prompt interface"}]


def test_repair_does_not_require_source_models_vlm_or_five_gpus(runtime_config, monkeypatch):
    state = snapshot(gpu_count=1, checks=[runtime._check(code, code, False, "Unavailable")
                                         for code in ("worldgen_code", "vlm_service", "panorama_runtime", "hyworld_weights", "sam3_weights")])
    state["python"] = {"missing": ["cv2"]}
    monkeypatch.setattr(runtime, "_probe_runtime", lambda _: state)
    runtime.require_repair_capability(runtime_config)
    with pytest.raises(HTTPException):
        runtime.require_task_capability(runtime_config, "worldgen")
    state["python"]["missing"].append("gsplat.rendering")
    runtime.clear_runtime_cache()
    with pytest.raises(HTTPException) as caught:
        runtime.require_repair_capability(runtime_config)
    assert caught.value.detail["blockers"][0]["code"] == "repair_dependencies"


def test_missing_runtime_exposes_specific_blockers_without_model_downloads(runtime_config, monkeypatch):
    monkeypatch.setattr(runtime, "probe_gpus", lambda: {"gpus": [], "available": False, "error": "No GPU"})
    monkeypatch.setattr(runtime, "_probe_python", lambda: {"missing": ["torch"], "cuda_devices": 0})
    monkeypatch.setattr(runtime, "_probe_vlm", lambda _: {"ok": False, "message": "No VLM"})
    monkeypatch.setattr(runtime.shutil, "which", lambda _: None)
    status = runtime.get_system_status(runtime_config)
    assert status["mode"] == "real"
    assert status["verification_scope"] == "dependency_preflight"
    assert status["ready"] is False
    checks = {item["code"]: item for item in status["checks"]}
    assert checks["hyworld_weights"]["ok"] is False
    assert checks["panorama_runtime"]["ok"] is False
    assert checks["task_storage"]["ok"] is True
    assert not Path(runtime_config.HF_HOME).exists()


def test_dev_script_requires_explicit_demo_opt_in():
    source = (Path(__file__).parents[1] / "scripts/dev_local.sh").read_text()
    assert 'MODE="real"' in source
    assert '--demo) MODE="demo"' in source
    assert 'GSSTUDIO_MOCK_PIPELINE=true' not in source


def test_gpu_endpoint_uses_observed_telemetry_even_in_demo(monkeypatch):
    from server import main
    monkeypatch.setattr(main.config, "MOCK_PIPELINE", True)
    monkeypatch.setattr(main, "probe_gpus", lambda: {"gpus": [], "available": False, "error": "No local GPU"})
    response = asyncio.run(main.gpu_status())
    assert response["gpus"] == []
    assert response["available"] is False
    assert response["mode"] == "demo"


def test_system_endpoint_exposes_worker_task_and_dependency_scope(monkeypatch):
    from server import main
    def status(config, active_task):
        return {"mode": "real", "ready": False, "verification_scope": "dependency_preflight",
                "gpu": {"active_task": active_task}}
    monkeypatch.setattr(main.worker, "_current_task_id", "running-task")
    monkeypatch.setattr(main, "get_system_status", status)
    response = asyncio.run(main.system_status())
    assert response["gpu"]["active_task"] == "running-task"
    assert response["verification_scope"] == "dependency_preflight"


@pytest.mark.parametrize("task_type", ["worldgen", "worldgen_from_pano", "worldrecon"])
def test_native_pipeline_commands_preserve_interpreter_imports_and_absolute_paths(tmp_path, monkeypatch, task_type):
    from server.pipeline.worldgen import get_worldgen_commands
    from server.pipeline.worldrecon import get_worldrecon_commands
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch-models"))
    arguments = {"task_dir": Path("data/task"), "input_dir": Path("data/task/input"),
                 "hyworld_repo": Path("custom-hyworld"), "config": {"recon_gpus": 2, "hf_home": "model-cache"}}
    if task_type == "worldrecon":
        commands = get_worldrecon_commands(**arguments)
    else:
        commands = get_worldgen_commands(**arguments, scene_dir=Path("data/task/scene"),
                                        vllm_host="localhost", vllm_port=8100, vllm_model="qwen", task_type=task_type)
    for name, command, env, cwd in commands:
        assert cwd.is_absolute()
        roots = env["PYTHONPATH"].split(os.pathsep)
        assert str(Path(__file__).resolve().parents[1]) in roots
        assert str(tmp_path / "custom-hyworld") in roots
        assert env["HF_HOME"] == str(tmp_path / "model-cache")
        assert env["TORCH_HOME"] == str(tmp_path / "torch-models")
        if name not in {"source_generate", "panogen"}:
            assert command[0] == sys.executable
        if "--nproc_per_node" in command:
            assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
        for flag in ("--task-dir", "--target_path", "--input_path", "--output_path", "--result_dir"):
            if flag in command:
                assert Path(command[command.index(flag) + 1]).is_absolute()
