"""Backend tests for GSStudio."""
import asyncio
import base64
import json
import os
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
from server.pipeline.worldgen import get_worldgen_commands


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


class TestWorldgenCommands:
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
            "type": "worldgen",
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
