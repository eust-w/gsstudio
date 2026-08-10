"""Backend tests for GSStudio."""
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Set mock mode before importing server modules
os.environ["GSSTUDIO_MOCK_PIPELINE"] = "true"
os.environ["GSSTUDIO_DATA_DIR"] = tempfile.mkdtemp()
os.environ["GSSTUDIO_BASIC_AUTH_USER"] = "poc-user"
os.environ["GSSTUDIO_BASIC_AUTH_PASSWORD"] = "poc-password"

from server.config import Config
from server.db import create_task, get_task, init_db, list_tasks, set_task_stages, update_stage, update_task
from server.models import StageStatus, TaskStatus
from server.pipeline.stage_runner import run_stage
from server.pipeline.worldgen import get_worldgen_commands
from server.pipeline.worker import PipelineWorker, VLLM_FREE_STAGES


@pytest.fixture
def cfg():
    c = Config()
    c.DATA_DIR.mkdir(parents=True, exist_ok=True)
    c.tasks_dir.mkdir(parents=True, exist_ok=True)
    init_db(c.DB_PATH)
    return c


@pytest.fixture
def task_id(cfg):
    return create_task(cfg.DB_PATH, name="test-task", task_type="worldgen", config={"prompt": "test"})


class TestDB:
    def test_create_task(self, cfg):
        tid = create_task(cfg.DB_PATH, name="t1", task_type="worldrecon")
        assert tid is not None
        task = get_task(cfg.DB_PATH, tid)
        assert task["name"] == "t1"
        assert task["type"] == "worldrecon"
        assert task["status"] == "pending"

    def test_get_task_not_found(self, cfg):
        assert get_task(cfg.DB_PATH, "nonexistent") is None

    def test_list_tasks(self, cfg):
        create_task(cfg.DB_PATH, name="t1", task_type="worldgen")
        create_task(cfg.DB_PATH, name="t2", task_type="worldrecon")
        tasks, total = list_tasks(cfg.DB_PATH)
        assert total >= 2

    def test_list_tasks_filter(self, cfg):
        create_task(cfg.DB_PATH, name="wg1", task_type="worldgen")
        create_task(cfg.DB_PATH, name="wr1", task_type="worldrecon")
        tasks, total = list_tasks(cfg.DB_PATH, task_type="worldrecon")
        assert all(t["type"] == "worldrecon" for t in tasks)

    def test_update_task(self, cfg, task_id):
        update_task(cfg.DB_PATH, task_id, status="running", progress=0.5)
        task = get_task(cfg.DB_PATH, task_id)
        assert task["status"] == "running"
        assert task["progress"] == 0.5

    def test_set_and_update_stages(self, cfg, task_id):
        stage_names = ["panogen", "traj_generate", "video_gen"]
        set_task_stages(cfg.DB_PATH, task_id, stage_names)

        task = get_task(cfg.DB_PATH, task_id)
        assert len(task["stages"]) == 3
        assert all(s["status"] == "pending" for s in task["stages"])

        update_stage(cfg.DB_PATH, task_id, "panogen", StageStatus.RUNNING)
        task = get_task(cfg.DB_PATH, task_id)
        assert task["stages"][0]["status"] == "running"
        assert task["current_stage"] == "panogen"

        update_stage(cfg.DB_PATH, task_id, "panogen", StageStatus.COMPLETED)
        task = get_task(cfg.DB_PATH, task_id)
        assert task["stages"][0]["status"] == "completed"
        assert task["progress"] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_stage_failed_with_error(self, cfg, task_id):
        set_task_stages(cfg.DB_PATH, task_id, ["s1"])
        update_stage(cfg.DB_PATH, task_id, "s1", StageStatus.RUNNING)
        update_stage(cfg.DB_PATH, task_id, "s1", StageStatus.FAILED, error="OOM")
        task = get_task(cfg.DB_PATH, task_id)
        assert task["stages"][0]["status"] == "failed"
        assert task["stages"][0]["error"] == "OOM"


class TestTaskStateMachine:
    VALID_TRANSITIONS = {
        "pending": ["running", "cancelled"],
        "running": ["completed", "failed", "cancelled"],
        "completed": [],
        "failed": ["pending"],  # retry
        "cancelled": ["pending"],  # retry
    }

    def test_valid_transitions(self):
        for from_status, to_list in self.VALID_TRANSITIONS.items():
            for to_status in to_list:
                assert to_status in self.VALID_TRANSITIONS.get(from_status, [])


class TestCheckpointResume:
    def test_checkpoint_marker(self, cfg, task_id):
        task_dir = cfg.task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)

        # No marker initially
        assert not (task_dir / ".stage_panogen_done").exists()

        # Create marker
        (task_dir / ".stage_panogen_done").touch()
        assert (task_dir / ".stage_panogen_done").exists()

    def test_worker_recovers_interrupted_task_after_container_restart(self, cfg):
        tid = create_task(cfg.DB_PATH, name="interrupted", task_type="worldgen_from_pano")
        update_task(cfg.DB_PATH, tid, status=TaskStatus.RUNNING.value)

        PipelineWorker(cfg)._recover_interrupted_tasks()

        task = get_task(cfg.DB_PATH, tid)
        assert task["status"] == TaskStatus.PENDING.value
        assert task["error"] is None

    def test_world_expansion_releases_vllm_resources(self):
        assert "panogen" in VLLM_FREE_STAGES
        assert "video_gen" in VLLM_FREE_STAGES

    def test_worker_marks_silent_navmesh_error_as_failed_stage(self, cfg, monkeypatch):
        import server.pipeline.worker as worker_module

        cfg.MOCK_PIPELINE = False

        task_id = create_task(
            cfg.DB_PATH,
            name="silent-navmesh-error",
            task_type="worldgen_from_pano",
            config={"quality_preset": "high"},
        )

        monkeypatch.setattr(
            worker_module,
            "get_worldgen_commands",
            lambda **_kwargs: [("traj_generate", ["fake"], {}, Path("/tmp"))],
        )

        async def successful_process_with_bad_log(*, log_file, **_kwargs):
            log_file.write_text("Navmesh Error: No module named 'rtree'\n")
            return True, None

        monkeypatch.setattr(worker_module, "run_stage", successful_process_with_bad_log)

        asyncio.run(PipelineWorker(cfg).execute_task(task_id))

        task = get_task(cfg.DB_PATH, task_id)
        assert task["status"] == TaskStatus.FAILED.value
        stage = next(stage for stage in task["stages"] if stage["name"] == "traj_generate")
        assert stage["status"] == StageStatus.FAILED.value
        assert "Navmesh Error" in stage["error"]
        assert any(artifact["name"] == "quality_report.json" for artifact in task["artifacts"])

    def test_worker_preserves_cancelled_terminal_status_when_a_stage_stops(self, cfg, monkeypatch):
        import server.pipeline.worker as worker_module

        cfg.MOCK_PIPELINE = False
        task_id = create_task(
            cfg.DB_PATH,
            name="cancelled-worldgen",
            task_type="worldgen_from_pano",
            config={"quality_preset": "high"},
        )
        monkeypatch.setattr(
            worker_module,
            "get_worldgen_commands",
            lambda **_kwargs: [("traj_generate", ["fake"], {}, Path("/tmp"))],
        )

        async def cancelled_stage(*, cancel_event, **_kwargs):
            cancel_event.set()
            return False, "Cancelled by user"

        monkeypatch.setattr(worker_module, "run_stage", cancelled_stage)

        asyncio.run(PipelineWorker(cfg).execute_task(task_id))

        assert get_task(cfg.DB_PATH, task_id)["status"] == TaskStatus.CANCELLED.value


class TestWorldgenCommands:
    def test_high_quality_uses_full_model_whitelist_and_fixed_training_schedule(self, tmp_path):
        commands = get_worldgen_commands(
            task_dir=tmp_path / "task",
            scene_dir=tmp_path / "scene",
            input_dir=tmp_path / "input",
            config={"quality_preset": "high", "num_gpus": 5},
            hyworld_repo=Path("/opt/hyworld2"),
            vllm_host="localhost",
            vllm_port=8100,
            vllm_model="Qwen/Qwen3-VL-8B-Instruct",
            task_type="worldgen_from_pano",
        )
        by_stage = {name: command for name, command, _, _ in commands}

        assert by_stage["video_gen"][by_stage["video_gen"].index("--model_type") + 1] == "worldstereo-memory"
        assert by_stage["video_gen"][by_stage["video_gen"].index("--trajectory_whitelist") + 1] == str(
            tmp_path / "task" / "video_trajectory_whitelist.json"
        )
        assert "--skip_exist" in by_stage["video_gen"]
        assert by_stage["gen_gs_data"][by_stage["gen_gs_data"].index("--result_name") + 1] == "worldstereo-memory"
        assert by_stage["gen_gs_data"][by_stage["gen_gs_data"].index("--trajectory_whitelist") + 1] == str(
            tmp_path / "task" / "trajectory_whitelist.json"
        )
        assert by_stage["gs_training"][by_stage["gs_training"].index("--max_steps") + 1] == "2000"
        assert by_stage["gs_training"][by_stage["gs_training"].index("--strategy.refine-start-iter") + 1] == "200"
        assert by_stage["gs_training"][by_stage["gs_training"].index("--strategy.refine-stop-iter") + 1] == "1000"
        assert by_stage["gs_training"][by_stage["gs_training"].index("--strategy.refine-every") + 1] == "133"

    def test_fast_quality_keeps_dmd_model_type_in_every_downstream_stage(self, tmp_path):
        commands = get_worldgen_commands(
            task_dir=tmp_path / "task",
            scene_dir=tmp_path / "scene",
            input_dir=tmp_path / "input",
            config={"quality_preset": "fast"},
            hyworld_repo=Path("/opt/hyworld2"),
            vllm_host="localhost",
            vllm_port=8100,
            vllm_model="Qwen/Qwen3-VL-8B-Instruct",
            task_type="worldgen_from_pano",
        )
        by_stage = {name: command for name, command, _, _ in commands}

        assert by_stage["video_gen"][by_stage["video_gen"].index("--model_type") + 1] == "worldstereo-memory-dmd"
        assert by_stage["gen_gs_data"][by_stage["gen_gs_data"].index("--result_name") + 1] == "worldstereo-memory-dmd"

    def test_panogen_uses_isolated_runtime_and_supported_attention(self, tmp_path):
        commands = get_worldgen_commands(
            task_dir=tmp_path / "task",
            scene_dir=tmp_path / "scene",
            input_dir=tmp_path / "input",
            config={"panogen_python": "/opt/panogen/bin/python"},
            hyworld_repo=Path("/opt/hyworld2"),
            vllm_host="localhost",
            vllm_port=8100,
            vllm_model="Qwen/Qwen3-VL-8B-Instruct",
        )

        _, command, _, _ = commands[0]
        assert command[0] == "/opt/panogen/bin/python"
        assert command[command.index("--attn-impl") + 1] == "sdpa"

    def test_trajectory_generation_uses_only_persisted_huggingface_cache(self, tmp_path):
        commands = get_worldgen_commands(
            task_dir=tmp_path / "task",
            scene_dir=tmp_path / "scene",
            input_dir=tmp_path / "input",
            config={"hf_home": "/models/hf-cache"},
            hyworld_repo=Path("/opt/hyworld2"),
            vllm_host="localhost",
            vllm_port=8100,
            vllm_model="Qwen/Qwen3-VL-8B-Instruct",
            task_type="worldgen_from_pano",
        )

        _, _, trajectory_env, _ = next(
            command for command in commands if command[0] == "traj_generate"
        )
        assert trajectory_env["HF_HUB_OFFLINE"] == "1"
        assert trajectory_env["TRANSFORMERS_OFFLINE"] == "1"

    def test_world_expansion_uses_only_persisted_huggingface_cache(self, tmp_path):
        commands = get_worldgen_commands(
            task_dir=tmp_path / "task",
            scene_dir=tmp_path / "scene",
            input_dir=tmp_path / "input",
            config={"hf_home": "/models/hf-cache"},
            hyworld_repo=Path("/opt/hyworld2"),
            vllm_host="localhost",
            vllm_port=8100,
            vllm_model="Qwen/Qwen3-VL-8B-Instruct",
            task_type="worldgen_from_pano",
        )

        _, command, video_env, _ = next(
            item for item in commands if item[0] == "video_gen"
        )
        assert "--local_files_only" in command
        assert command[command.index("--nproc_per_node") + 1] == "4"
        assert video_env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert video_env["HF_HUB_OFFLINE"] == "1"
        assert video_env["TRANSFORMERS_OFFLINE"] == "1"


class TestStageRunner:
    def test_stage_accepts_log_lines_larger_than_asyncio_default_limit(self, tmp_path):
        log_file = tmp_path / "long-line.log"
        payload_size = 128 * 1024

        success, error = asyncio.run(
            run_stage(
                stage_name="long-line",
                cmd=[
                    sys.executable,
                    "-c",
                    f"import sys; sys.stdout.write('x' * {payload_size}); sys.stdout.flush()",
                ],
                log_file=log_file,
            )
        )

        assert success, error
        assert log_file.stat().st_size == payload_size + 1


class TestFrontendViewerContract:
    def test_splat_viewer_uses_spark_streaming_lod_loader(self):
        source = (Path(__file__).parents[1] / "web/src/components/SplatViewer.tsx").read_text()

        assert 'from "@sparkjsdev/spark"' in source
        assert "new SplatMesh" in source
        assert "lod: true" in source
        assert "onProgress" in source


class TestDeploymentContract:
    def test_quality_filter_patch_adds_a_trajectory_whitelist_to_upstream_gs_data(self, tmp_path):
        upstream = tmp_path / "gen_gs_data.py"
        upstream.write_text(
            "import argparse\nimport json\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--result_name', '-n', type=str, default='worldstereo-memory-dmd')\n"
            "video_paths = sorted(video_paths, key=lambda x: (x.split(\"/\")[-3], x.split(\"/\")[-2]))\n"
            "video_paths = video_paths[rank::world_size]\n"
        )
        patch_script = Path(__file__).parents[1] / "deploy/scripts/patch_hyworld_quality_filter.py"

        subprocess.run([sys.executable, str(patch_script), str(upstream)], check=True)
        patched = upstream.read_text()

        assert "--trajectory_whitelist" in patched
        assert "accepted_trajectory_ids" in patched
        assert "Skipping all videos is not allowed" in patched

    def test_video_filter_patch_limits_worldstereo_to_planned_candidates(self, tmp_path):
        upstream = tmp_path / "video_gen.py"
        upstream.write_text(
            "import argparse\nimport json\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument(\"--target_path\", default=None, type=str, help=\"target path\")\n"
            "render_list = sort_trajs(f\"{scene}/render_results\")\n"
            "if os.path.exists(f\"{scene}/render_results/generation_bank_{args.model_type}/aligned_pcd.ply\") and args.skip_exist:\n"
            "    continue\n"
            "memory_bank.apply_worldmirror(skip_exist=True)\n"
        )
        patch_script = Path(__file__).parents[1] / "deploy/scripts/patch_hyworld_video_filter.py"

        subprocess.run([sys.executable, str(patch_script), str(upstream)], check=True)
        patched = upstream.read_text()

        assert "--trajectory_whitelist" in patched
        assert "candidate_trajectory_ids" in patched
        assert "No trajectory renderings matched" in patched
        assert "and not args.trajectory_whitelist" in patched
        assert "skip_exist=not bool(args.trajectory_whitelist)" in patched

        subprocess.run([sys.executable, str(patch_script), str(upstream)], check=True)
        assert upstream.read_text() == patched

    def test_image_and_startup_preflight_require_rtree(self):
        dockerfile = (Path(__file__).parents[1] / "Dockerfile.a800").read_text()
        entrypoint = (Path(__file__).parents[1] / "deploy/scripts/entrypoint.sh").read_text()

        assert "libspatialindex-dev" in dockerfile
        assert "pip install --no-cache-dir rtree" in dockerfile
        assert "import rtree" in dockerfile
        assert "import rtree" in entrypoint

    def test_image_patches_upstream_gs_data_for_quality_whitelists(self):
        dockerfile = (Path(__file__).parents[1] / "Dockerfile.a800").read_text()

        assert "patch_hyworld_quality_filter.py" in dockerfile
        assert "/opt/hyworld2/hyworld2/worldgen/gen_gs_data.py" in dockerfile

    def test_image_patches_upstream_video_generation_for_candidate_whitelists(self):
        dockerfile = (Path(__file__).parents[1] / "Dockerfile.a800").read_text()

        assert "patch_hyworld_video_filter.py" in dockerfile
        assert "/opt/hyworld2/hyworld2/worldgen/video_gen.py" in dockerfile

    def test_gpu_rollout_manifests_target_the_immutable_v30_image(self):
        root = Path(__file__).parents[1]
        deployment = (root / "deploy/k8s/deployment-gpu.yaml").read_text()
        build = (root / "deploy/k8s/kaniko-build.yaml").read_text()

        assert "gsstudio:a800-v30" in deployment
        assert "gsstudio:a800-v30" in build
        assert "gsstudio:a800-v22" not in deployment
        assert "gsstudio:a800-v22" not in build

    def test_entrypoint_maps_upstream_hf_cache_to_persistent_hub(self):
        source = (Path(__file__).parents[1] / "deploy/scripts/entrypoint.sh").read_text()

        assert 'UPSTREAM_HF_HOME="/root/.cache/huggingface"' in source
        assert 'ln -sfn "$GSSTUDIO_HF_HOME/hub" "$UPSTREAM_HF_HOME/hub"' in source

    def test_prefetch_includes_all_trajectory_huggingface_dependencies(self):
        source = (Path(__file__).parents[1] / "deploy/scripts/prefetch_models.sh").read_text()

        assert "naver-iv/zim-anything-vitl" in source
        assert "IDEA-Research/grounding-dino-tiny" in source

    def test_prefetch_includes_worldstereo_base_model_components(self):
        source = (Path(__file__).parents[1] / "deploy/scripts/prefetch_models.sh").read_text()

        assert "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers" in source
        assert "facebook/dinov2-base" in source
        for component in (
            "transformer/*",
            "text_encoder/*",
            "image_encoder/*",
            "image_processor/*",
            "tokenizer/*",
            "vae/*",
            "scheduler/*",
        ):
            assert component in source

    def test_startup_preflight_rejects_an_incomplete_full_worldstereo_cache(self):
        source = (Path(__file__).parents[1] / "deploy/scripts/entrypoint.sh").read_text()

        assert "WorldStereo offline preflight" in source
        assert "scheduler/scheduler_config.json" in source
        assert "worldstereo-memory/model.safetensors" in source

    def test_kubernetes_prefetch_includes_full_worldstereo_weights_for_high_quality(self):
        source = (Path(__file__).parents[1] / "deploy/scripts/prefetch_models.sh").read_text()

        assert "('hanshanxue/WorldStereo', None)" in source

    def test_quality_overlay_inherits_verified_cuda_runtime(self):
        source = (Path(__file__).parents[1] / "Dockerfile.a800").read_text()

        assert "FROM ccr-29eug8s3-pub.cnc.bj.baidubce.com/dataflow-poc/gsstudio:a800-v22" in source
        assert "GSStudio a800-v30" in source

    def test_image_patches_worldstereo_transformer_for_local_only_loading(self):
        source = (Path(__file__).parents[1] / "Dockerfile.a800").read_text()
        patch_source = (
            Path(__file__).parents[1] / "deploy/scripts/patch_hyworld_local_cache.py"
        ).read_text()

        assert "patch_hyworld_local_cache.py" in source
        assert "/opt/hyworld2/hyworld2/worldgen/models/worldstereo_wrapper.py" in source
        assert "T5Tokenizer.from_pretrained" in patch_source

    def test_gpu_deployment_allows_worldstereo_cpu_shard_loading(self):
        source = (Path(__file__).parents[1] / "deploy/k8s/deployment-gpu.yaml").read_text()

        assert 'memory: 128Gi' in source
        assert 'memory: 512Gi' in source


class TestAPI:
    @pytest.fixture
    def client(self, cfg):
        from fastapi.testclient import TestClient
        from server.main import app
        token = base64.b64encode(b"poc-user:poc-password").decode()
        with TestClient(app, headers={"Authorization": f"Basic {token}"}) as c:
            yield c

    def test_auth_rejects_missing_credentials(self, client):
        resp = client.get("/api/tasks", headers={"Authorization": ""})
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == 'Basic realm="GSStudio"'

    def test_auth_rejects_wrong_credentials(self, client):
        token = base64.b64encode(b"poc-user:wrong").decode()
        resp = client.get("/api/tasks", headers={"Authorization": f"Basic {token}"})
        assert resp.status_code == 401

    def test_health_does_not_require_auth(self, client):
        resp = client.get("/api/health", headers={"Authorization": ""})
        assert resp.status_code == 200

    def test_spa_deep_link_serves_frontend(self, client):
        resp = client.get("/tasks/example-task")
        assert resp.status_code == 200
        assert "<title>GSStudio</title>" in resp.text

    def test_unknown_api_route_remains_not_found(self, client):
        resp = client.get("/api/does-not-exist")
        assert resp.status_code == 404

    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_gpu_status_mock(self, client):
        resp = client.get("/api/gpu/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "gpus" in data

    def test_create_and_get_task(self, client):
        resp = client.post("/api/tasks", data={
            "type": "worldrecon",
            "name": "test-api-task",
            "config": "{}",
        })
        assert resp.status_code == 201
        task_id = resp.json()["task_id"]

        resp = client.get(f"/api/tasks/{task_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "test-api-task"

    def test_worldgen_form_prompt_overrides_advanced_config_and_persists_high_defaults(self, client, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

        resp = client.post(
            "/api/tasks",
            data={
                "type": "worldgen",
                "name": "quality-config",
                "prompt": "form prompt wins",
                "config": json.dumps({"prompt": "advanced prompt", "seed": "17"}),
            },
            files={"image": ("input.png", b"image-bytes", "image/png")},
        )

        assert resp.status_code == 201
        task = client.get(f"/api/tasks/{resp.json()['task_id']}").json()
        assert task["config"] == {
            "prompt": "form prompt wins",
            "quality_preset": "high",
            "seed": 17,
            "worldstereo_model_type": "worldstereo-memory",
        }

    def test_worldgen_rejects_invalid_quality_preset(self, client, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

        resp = client.post(
            "/api/tasks",
            data={
                "type": "worldgen",
                "name": "bad-quality-config",
                "config": json.dumps({"quality_preset": "ultra"}),
            },
            files={"image": ("input.png", b"image-bytes", "image/png")},
        )

        assert resp.status_code == 400
        assert "quality_preset" in resp.json()["detail"]

    @pytest.mark.parametrize(
        ("task_type", "upload_field"),
        [("worldgen", "image"), ("worldgen_from_pano", "panorama")],
    )
    def test_world_generation_fails_fast_without_sam3_access_token(
        self, client, monkeypatch, task_type, upload_field
    ):
        monkeypatch.delenv("HF_TOKEN", raising=False)

        resp = client.post(
            "/api/tasks",
            data={"type": task_type, "name": "missing-sam3-token", "config": "{}"},
            files={upload_field: ("input.png", b"image-bytes", "image/png")},
        )

        assert resp.status_code == 503
        assert resp.json()["detail"] == (
            "World generation requires a one-time download of facebook/sam3. "
            "Approve access and configure the gsstudio-huggingface Kubernetes "
            "Secret; after the model is cached, the token is no longer required."
        )

    def test_world_generation_uses_persisted_sam3_cache_without_token(
        self, client, monkeypatch, tmp_path
    ):
        from server.routes import tasks as task_routes

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(task_routes._config, "HF_HOME", str(tmp_path))
        cached_snapshot = (
            tmp_path
            / "hub"
            / "models--facebook--sam3"
            / "snapshots"
            / "approved-revision"
        )
        cached_snapshot.mkdir(parents=True)
        (cached_snapshot / "config.json").write_text("{}")
        (cached_snapshot / "model.safetensors").write_bytes(b"cached-weights")

        resp = client.post(
            "/api/tasks",
            data={"type": "worldgen", "name": "cached-sam3", "config": "{}"},
            files={"image": ("input.png", b"image-bytes", "image/png")},
        )

        assert resp.status_code == 201

    def test_get_task_not_found(self, client):
        resp = client.get("/api/tasks/nonexistent")
        assert resp.status_code == 404

    def test_list_tasks(self, client):
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert "total" in data

    def test_cancel_task(self, client):
        resp = client.post("/api/tasks", data={
            "type": "worldrecon",
            "name": "cancel-test",
            "config": "{}",
        })
        task_id = resp.json()["task_id"]

        resp = client.post(f"/api/tasks/{task_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_cancel_completed_task_fails(self, client):
        resp = client.post("/api/tasks", data={
            "type": "worldrecon",
            "name": "cancel-completed",
            "config": "{}",
        })
        task_id = resp.json()["task_id"]

        # Directly set task to completed in DB instead of waiting for pipeline
        from server.config import Config
        from server.db import update_task
        from server.models import TaskStatus
        cfg = Config()
        update_task(cfg.DB_PATH, task_id, status=TaskStatus.COMPLETED.value)

        resp = client.post(f"/api/tasks/{task_id}/cancel")
        assert resp.status_code == 400

    def test_update_task_name(self, client):
        resp = client.post("/api/tasks", data={
            "type": "worldrecon",
            "name": "before-edit",
            "config": "{}",
        })
        task_id = resp.json()["task_id"]

        resp = client.patch(f"/api/tasks/{task_id}", json={"name": "after-edit"})

        assert resp.status_code == 200
        assert resp.json()["name"] == "after-edit"
        assert client.get(f"/api/tasks/{task_id}").json()["name"] == "after-edit"

    def test_update_task_rejects_blank_name(self, client):
        task_id = client.post("/api/tasks", data={
            "type": "worldrecon",
            "name": "keep-me",
            "config": "{}",
        }).json()["task_id"]

        resp = client.patch(f"/api/tasks/{task_id}", json={"name": "   "})

        assert resp.status_code == 400
        assert client.get(f"/api/tasks/{task_id}").json()["name"] == "keep-me"

    def test_delete_completed_task_removes_record_and_files(self, client):
        from server.config import Config
        from server.db import update_task

        cfg = Config()
        task_id = client.post("/api/tasks", data={
            "type": "worldrecon",
            "name": "delete-me",
            "config": "{}",
        }).json()["task_id"]
        task_dir = cfg.task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "artifact.txt").write_text("generated")
        update_task(cfg.DB_PATH, task_id, status=TaskStatus.COMPLETED.value)

        resp = client.delete(f"/api/tasks/{task_id}")

        assert resp.status_code == 204
        assert client.get(f"/api/tasks/{task_id}").status_code == 404
        assert not task_dir.exists()

    def test_delete_active_task_is_rejected(self, client):
        from server.config import Config
        from server.db import update_task

        cfg = Config()
        task_id = client.post("/api/tasks", data={
            "type": "worldrecon",
            "name": "active-task",
            "config": "{}",
        }).json()["task_id"]
        update_task(cfg.DB_PATH, task_id, status=TaskStatus.RUNNING.value)

        resp = client.delete(f"/api/tasks/{task_id}")

        assert resp.status_code == 409
        assert client.get(f"/api/tasks/{task_id}").status_code == 200

    @pytest.mark.parametrize("retryable_status", [TaskStatus.FAILED, TaskStatus.CANCELLED])
    def test_retry_task_resets_transient_state_and_preserves_checkpoints(
        self, client, retryable_status
    ):
        from server.config import Config
        from server.db import update_task

        cfg = Config()
        task_id = client.post("/api/tasks", data={
            "type": "worldrecon",
            "name": "retry-me",
            "config": "{}",
        }).json()["task_id"]
        checkpoint = cfg.task_dir(task_id) / ".stage_traj_render_done"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()
        update_task(
            cfg.DB_PATH,
            task_id,
            status=retryable_status.value,
            progress=0.75,
            current_stage="gs_training",
            error="interrupted",
        )

        resp = client.post(f"/api/tasks/{task_id}/retry")

        assert resp.status_code == 200
        task = resp.json()
        assert task["status"] == TaskStatus.PENDING.value
        assert task["progress"] == 0.0
        assert task["current_stage"] is None
        assert task["error"] is None
        assert checkpoint.exists()

    @pytest.mark.parametrize(
        "blocked_status",
        [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED],
    )
    def test_retry_task_rejects_non_retryable_status(self, client, blocked_status):
        from server.config import Config
        from server.db import update_task

        cfg = Config()
        task_id = client.post("/api/tasks", data={
            "type": "worldrecon",
            "name": "do-not-retry",
            "config": "{}",
        }).json()["task_id"]
        update_task(cfg.DB_PATH, task_id, status=blocked_status.value)

        resp = client.post(f"/api/tasks/{task_id}/retry")

        assert resp.status_code == 409

    def test_retry_missing_task_returns_not_found(self, client):
        assert client.post("/api/tasks/missing/retry").status_code == 404
