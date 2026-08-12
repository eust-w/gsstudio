"""Backend tests for GSStudio."""
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

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
from server.pipeline.worker import PipelineWorker, VLLM_FREE_STAGES, _next_recovery_seed


def png_bytes(color=(120, 80, 40)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return buffer.getvalue()


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

    def test_pending_tasks_are_fifo_even_when_ids_sort_differently(self, cfg):
        first = create_task(cfg.DB_PATH, name="first", task_type="worldrecon", task_id="ffff-first")
        second = create_task(cfg.DB_PATH, name="second", task_type="worldrecon", task_id="aaaa-second")

        tasks, _ = list_tasks(cfg.DB_PATH, status="pending", limit=100)

        ids = [task["id"] for task in tasks]
        assert ids.index(first) < ids.index(second)

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
    def test_training_oom_retries_with_frozen_means_and_no_densification(self, cfg, monkeypatch, tmp_path):
        import server.pipeline.worker as worker_module

        calls: list[list[str]] = []

        async def fake_stage(**kwargs):
            calls.append(kwargs["cmd"])
            return (False, "CUDA out of memory") if len(calls) == 1 else (True, "")

        monkeypatch.setattr(worker_module, "run_stage", fake_stage)
        monkeypatch.setattr(worker_module, "prepare_safe_training_retry", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(worker_module, "validate_completed_delivery", lambda *_args, **_kwargs: {"valid": True})
        worker = PipelineWorker(cfg)

        success, error = asyncio.run(worker._run_gs_training_with_safe_fallback(
            task_dir=tmp_path,
            scene_dir=tmp_path / "scene",
            task_type="worldgen",
            stage_name="gs_training",
            cmd=["train", "--max_steps", "2000"],
            env={},
            log_file=tmp_path / "training.log",
            cwd=tmp_path,
            cancel_event=asyncio.Event(),
        ))

        assert success, error
        assert calls[1][calls[1].index("--means-lr") + 1] == "0"
        assert calls[1][calls[1].index("--strategy.refine-start-iter") + 1] == "1999"

    def test_recovery_seeds_are_based_on_the_original_seed(self):
        assert _next_recovery_seed({"seed": 1024}, 0) == 1025
        assert _next_recovery_seed({"seed": 1025, "_recovery_base_seed": 1024}, 1) == 1026

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

    def test_worker_requeues_only_prompt_tasks_failed_by_missing_input(self, cfg):
        recoverable = create_task(
            cfg.DB_PATH,
            name="prompt missing input",
            task_type="worldgen",
            config={"prompt": "robotics office"},
        )
        ignored = create_task(
            cfg.DB_PATH,
            name="other failure",
            task_type="worldgen",
            config={"prompt": "robotics office"},
        )
        recoverable_dir = cfg.task_dir(recoverable)
        (recoverable_dir / "logs").mkdir(parents=True)
        (recoverable_dir / "logs" / "panogen.log").write_text("old failure")
        update_task(
            cfg.DB_PATH,
            recoverable,
            status="failed",
            error="ValueError: Input image does not exist: /task/input/input.png",
        )
        update_task(cfg.DB_PATH, ignored, status="failed", error="CUDA out of memory")

        worker = PipelineWorker(cfg)
        worker._recover_prompt_only_failures()
        worker._recover_prompt_only_failures()

        recovered = get_task(cfg.DB_PATH, recoverable)
        assert recovered["status"] == "pending"
        assert recovered["config"]["input_mode"] == "prompt"
        assert recovered["config"]["seed"] == 1024
        assert (recoverable_dir / "logs" / "panogen_missing_input_attempt_000.log").read_text() == "old failure"
        assert get_task(cfg.DB_PATH, ignored)["status"] == "failed"

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

    def test_worker_runs_auto_recovery_initial_batch_and_stops_after_six_accepted(self, cfg, monkeypatch):
        import server.pipeline.worker as worker_module

        task_dir = cfg.task_dir("auto-recovery")
        scene_dir = task_dir / "scene"
        logs_dir = task_dir / "logs"
        logs_dir.mkdir(parents=True)
        scene_dir.mkdir(parents=True)
        (task_dir / "quality_report.json").write_text(json.dumps({
            "planning": {
                "trajectories": [
                    {"id": f"view{index}/traj0", "kind": "regular"}
                    for index in range(8)
                ]
            }
        }))

        async def successful_worldstereo(*, log_file, **_kwargs):
            manifest = json.loads((task_dir / "video_trajectory_whitelist.json").read_text())
            lines = []
            for trajectory_id in manifest["candidate_trajectory_ids"]:
                lines.extend([
                    f"{trajectory_id} (2 frames): k=[0.9,1.0], b=[0.0,0.1], max_abs_dev=[0.01,0.02], max_rel_dev=[1.0%,10.0%]",
                    f"[Alignment Rank0] [Frame {trajectory_id}/000000] [Success] Guidance mask ratio: 0.9200, depth percentile error ratio: 0.02000",
                    f"[Alignment Rank0] [Frame {trajectory_id}/000020] [Success] Guidance mask ratio: 0.9100, depth percentile error ratio: 0.03000",
                ])
            log_file.write_text("\n".join(lines))
            return True, ""

        monkeypatch.setattr(worker_module, "run_stage", successful_worldstereo)
        async def run_auto_recovery():
            return await PipelineWorker(cfg)._run_auto_high_quality_video_recovery(
                task_dir=task_dir,
                scene_dir=scene_dir,
                log_file=logs_dir / "video_gen.log",
                cmd=["worldstereo"],
                env={},
                cwd=task_dir,
                cancel_event=asyncio.Event(),
                task_type="worldgen_from_pano",
                recovery_policy="auto",
                render_spec=None,
            )
        success, error = asyncio.run(run_auto_recovery())

        assert success is True
        assert error == ""
        recovery = json.loads((task_dir / "auto_recovery.json").read_text())
        assert recovery["state"] == "completed"
        assert recovery["delivery_tier"] == "strict_3d"
        assert recovery["generated_video_trajectory_count"] == 8
        assert len(recovery["accepted_trajectory_ids"]) == 8

    def test_execute_task_routes_high_auto_policy_to_auto_recovery_worker(self, cfg, monkeypatch):
        import server.pipeline.worker as worker_module

        cfg.MOCK_PIPELINE = False
        task_id = create_task(
            cfg.DB_PATH,
            name="automatic-recovery",
            task_type="worldgen_from_pano",
            config={"quality_preset": "high", "recovery_policy": "auto"},
        )
        monkeypatch.setattr(
            worker_module,
            "get_worldgen_commands",
            lambda **_kwargs: [("video_gen", ["fake"], {}, Path("/tmp"))],
        )
        called = {"auto": False}

        async def auto_recovery(**_kwargs):
            called["auto"] = True
            return True, ""

        async def legacy_recovery(**_kwargs):
            raise AssertionError("automatic policy must not run the legacy replenishment loop")

        worker = PipelineWorker(cfg)
        monkeypatch.setattr(worker_module, "validate_completed_delivery", lambda *_args, **_kwargs: {"valid": True})
        monkeypatch.setattr(worker, "_run_auto_high_quality_video_recovery", auto_recovery)
        monkeypatch.setattr(worker, "_run_high_quality_video_recovery", legacy_recovery)

        asyncio.run(worker.execute_task(task_id))

        assert called["auto"] is True
        assert get_task(cfg.DB_PATH, task_id)["status"] == TaskStatus.COMPLETED.value

    def test_auto_worker_uses_rescue_before_normal_backlog_after_two_acceptances(self, cfg, monkeypatch):
        import server.pipeline.worker as worker_module

        cfg.MANAGE_VLLM = False
        task_dir = cfg.task_dir("auto-rescue")
        logs_dir = task_dir / "logs"
        scene_dir = task_dir / "scene"
        logs_dir.mkdir(parents=True)
        scene_dir.mkdir(parents=True)
        (task_dir / "quality_report.json").write_text(json.dumps({
            "planning": {"trajectories": [
                {"id": f"view{index}/traj0", "kind": "regular"} for index in range(12)
            ]}
        }))
        calls: list[tuple[str, list[str]]] = []

        async def fake_stage(*, stage_name, log_file, **_kwargs):
            manifest = json.loads((task_dir / "video_trajectory_whitelist.json").read_text())
            candidate_ids = manifest["candidate_trajectory_ids"]
            calls.append((stage_name, candidate_ids))
            if stage_name == "traj_render":
                for candidate_id in candidate_ids:
                    trajectory_dir = scene_dir / "render_results" / candidate_id
                    trajectory_dir.mkdir(parents=True, exist_ok=True)
                    for name in ("render.mp4", "render_mask.mp4", "traj_caption.json"):
                        (trajectory_dir / name).write_bytes(b"complete")
                return True, ""
            accepted_count = 2 if len([item for item in calls if item[0] == "video_gen"]) == 1 else 6
            with open(log_file, "a") as log:
                for index, trajectory_id in enumerate(candidate_ids):
                    guidance = 0.92 if index < accepted_count else 0.10
                    log.write(
                        f"{trajectory_id} (2 frames): k=[0.9,1.0], b=[0.0,0.1], max_abs_dev=[0.01,0.02], max_rel_dev=[1.0%,10.0%]\n"
                        f"[Alignment Rank0] [Frame {trajectory_id}/000000] [Success] "
                        f"Guidance mask ratio: {guidance:.4f}, depth percentile error ratio: 0.02000\n"
                        f"[Alignment Rank0] [Frame {trajectory_id}/000020] [Success] "
                        f"Guidance mask ratio: {guidance:.4f}, depth percentile error ratio: 0.02000\n"
                    )
            return True, ""

        monkeypatch.setattr(worker_module, "run_stage", fake_stage)
        monkeypatch.setattr(
            worker_module,
            "create_conservative_rescue_trajectories",
            lambda _scene_dir, *, round_number, **_kwargs: [
                {"id": f"view_recovery_r{round_number}_{index}/traj0", "kind": "regular"}
                for index in range(6)
            ],
        )

        async def run_recovery():
            return await PipelineWorker(cfg)._run_auto_high_quality_video_recovery(
                task_dir=task_dir,
                scene_dir=scene_dir,
                log_file=logs_dir / "video_gen.log",
                cmd=["worldstereo"],
                env={},
                cwd=task_dir,
                cancel_event=asyncio.Event(),
                task_type="worldgen_from_pano",
                recovery_policy="auto",
                render_spec=(["renderer"], {}, task_dir),
            )
        success, error = asyncio.run(run_recovery())

        assert success, error
        assert [(stage, len(batch)) for stage, batch in calls] == [
            ("video_gen", 8),
            ("traj_render", 6),
            ("video_gen", 6),
        ]
        assert calls[1][1] == [f"view_recovery_r1_{index}/traj0" for index in range(6)]

    def test_auto_worker_rerenders_uncheckpointed_pending_rescue_before_video(self, cfg, monkeypatch):
        """A crash after MP4 output but before captions must resume rendering."""
        import server.pipeline.worker as worker_module
        from server.pipeline.recovery import (
            complete_auto_recovery_batch,
            prepare_auto_recovery,
            schedule_auto_recovery_batch,
        )

        cfg.MANAGE_VLLM = False
        task_dir = cfg.task_dir("resume-pending-rescue")
        scene_dir = task_dir / "scene"
        logs_dir = task_dir / "logs"
        scene_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        planning = {
            "trajectories": [
                {"id": f"view{index}/traj0", "kind": "regular"}
                for index in range(8)
            ]
        }
        (task_dir / "quality_report.json").write_text(json.dumps({"planning": planning}))
        initial = prepare_auto_recovery(
            task_dir,
            planning=planning,
            task_type="worldgen_from_pano",
            recovery_policy="auto",
        )
        complete_auto_recovery_batch(
            task_dir,
            alignment={"accepted_trajectory_ids": initial["pending_candidate_trajectory_ids"][:2]},
            duration_seconds=1,
        )
        needs_rescue = prepare_auto_recovery(
            task_dir,
            planning=planning,
            task_type="worldgen_from_pano",
            recovery_policy="auto",
        )
        assert needs_rescue["state"] == "needs_rescue"
        rescue_ids = [f"view_recovery_r1_{index}/traj0" for index in range(4)]
        schedule_auto_recovery_batch(task_dir, phase="rescue_1", candidate_ids=rescue_ids)

        calls: list[str] = []

        async def fake_stage(*, stage_name, log_file, **_kwargs):
            calls.append(stage_name)
            if stage_name == "traj_render":
                for candidate_id in rescue_ids:
                    trajectory_dir = scene_dir / "render_results" / candidate_id
                    trajectory_dir.mkdir(parents=True, exist_ok=True)
                    for name in ("render.mp4", "render_mask.mp4", "traj_caption.json"):
                        (trajectory_dir / name).write_bytes(b"complete")
            else:
                with open(log_file, "a") as log:
                    for trajectory_id in rescue_ids:
                        log.write(
                            f"{trajectory_id} (2 frames): k=[0.9,1.0], b=[0.0,0.1], "
                            "max_abs_dev=[0.01,0.02], max_rel_dev=[1.0%,10.0%]\n"
                            f"[Alignment Rank0] [Frame {trajectory_id}/000000] [Success] "
                            "Guidance mask ratio: 0.9200, depth percentile error ratio: 0.02000\n"
                            f"[Alignment Rank0] [Frame {trajectory_id}/000020] [Success] "
                            "Guidance mask ratio: 0.9100, depth percentile error ratio: 0.03000\n"
                        )
            return True, ""

        monkeypatch.setattr(worker_module, "run_stage", fake_stage)
        worker = PipelineWorker(cfg)
        monkeypatch.setattr(worker_module, "validate_completed_delivery", lambda *_args, **_kwargs: {"valid": True})
        async def run_recovery():
            return await worker._run_auto_high_quality_video_recovery(
                task_dir=task_dir,
                scene_dir=scene_dir,
                log_file=logs_dir / "video_gen.log",
                cmd=["worldstereo"],
                env={},
                cwd=task_dir,
                cancel_event=asyncio.Event(),
                task_type="worldgen_from_pano",
                recovery_policy="auto",
                render_spec=(["renderer"], {}, task_dir),
            )

        success, error = asyncio.run(run_recovery())

        assert success, error
        assert calls == ["traj_render", "video_gen"]
        report = json.loads((task_dir / "auto_recovery.json").read_text())
        assert set(rescue_ids).issubset(report["rendered_candidate_trajectory_ids"])
        assert report["delivery_tier"] == "strict_3d"

    def test_recovery_render_retries_success_without_required_caption(self, cfg, monkeypatch):
        import server.pipeline.worker as worker_module
        from server.pipeline.recovery import prepare_auto_recovery

        task_dir = cfg.task_dir("retry-missing-render-assets")
        scene_dir = task_dir / "scene"
        task_dir.mkdir(parents=True)
        scene_dir.mkdir()
        planning = {"trajectories": [{"id": "view0/traj0", "kind": "regular"}]}
        prepare_auto_recovery(
            task_dir,
            planning=planning,
            task_type="worldgen_from_pano",
            recovery_policy="auto",
        )
        candidate_ids = ["view_recovery_r1_0/traj0"]
        attempts = 0

        async def fake_stage(**_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                trajectory_dir = scene_dir / "render_results" / candidate_ids[0]
                trajectory_dir.mkdir(parents=True)
                for name in ("render.mp4", "render_mask.mp4", "traj_caption.json"):
                    (trajectory_dir / name).write_bytes(b"complete")
            return True, ""

        async def no_sleep(_seconds):
            return None

        monkeypatch.setattr(worker_module, "run_stage", fake_stage)
        monkeypatch.setattr(worker_module.asyncio, "sleep", no_sleep)
        worker = PipelineWorker(cfg)

        async def render_batch():
            return await worker._render_recovery_batch(
                task_dir=task_dir,
                scene_dir=scene_dir,
                candidate_ids=candidate_ids,
                render_spec=(["renderer"], {}, task_dir),
                render_log=task_dir / "render.log",
                cancel_event=asyncio.Event(),
            )

        success, error = asyncio.run(render_batch())

        assert success, error
        assert attempts == 2
        report = json.loads((task_dir / "auto_recovery.json").read_text())
        assert report["transient_retry_count"] == 1
        assert report["rendered_candidate_trajectory_ids"] == candidate_ids

    def test_auto_worker_uses_next_seed_for_generated_source_after_two_failed_rescue_rounds(self, cfg, monkeypatch):
        from server.pipeline.recovery import (
            complete_auto_recovery_batch,
            prepare_auto_recovery,
            schedule_auto_recovery_batch,
        )

        cfg.MANAGE_VLLM = False
        task_dir = cfg.task_dir("auto-regenerate")
        task_dir.mkdir(parents=True)
        planning = {"trajectories": [{"id": f"view{index}/traj0", "kind": "regular"} for index in range(8)]}
        (task_dir / "quality_report.json").write_text(json.dumps({"planning": planning}))
        first = prepare_auto_recovery(task_dir, planning=planning, task_type="worldgen", recovery_policy="auto")
        complete_auto_recovery_batch(task_dir, alignment={"accepted_trajectory_ids": []}, duration_seconds=1)
        prepare_auto_recovery(task_dir, planning=planning, task_type="worldgen", recovery_policy="auto")
        schedule_auto_recovery_batch(task_dir, phase="rescue_1", candidate_ids=[f"view_recovery_r1_{i}/traj0" for i in range(6)])
        complete_auto_recovery_batch(task_dir, alignment={"accepted_trajectory_ids": []}, duration_seconds=1)
        prepare_auto_recovery(task_dir, planning=planning, task_type="worldgen", recovery_policy="auto")
        schedule_auto_recovery_batch(task_dir, phase="rescue_2", candidate_ids=[f"view_recovery_r2_{i}/traj0" for i in range(6)])
        complete_auto_recovery_batch(task_dir, alignment={"accepted_trajectory_ids": []}, duration_seconds=1)
        assert prepare_auto_recovery(task_dir, planning=planning, task_type="worldgen", recovery_policy="auto")["state"] == "needs_regeneration"

        called: dict[str, object] = {}

        async def regenerated_attempt(**kwargs):
            called.update(kwargs)
            return True, ""

        worker = PipelineWorker(cfg)
        monkeypatch.setattr(worker, "_run_regenerated_source_attempt", regenerated_attempt)

        async def run_auto_recovery():
            return await worker._run_auto_high_quality_video_recovery(
                task_dir=task_dir,
                scene_dir=task_dir / "scene",
                log_file=task_dir / "logs" / "video_gen.log",
                cmd=["worldstereo"],
                env={},
                cwd=task_dir,
                cancel_event=asyncio.Event(),
                task_type="worldgen",
                recovery_policy="auto",
                render_spec=None,
                input_dir=task_dir / "input",
                task_config={"seed": 1024},
            )
        success, error = asyncio.run(run_auto_recovery())

        assert success, error
        assert called["seed"] == 1025

    def test_resumed_source_regeneration_continues_with_gs_data_and_training(self, cfg, monkeypatch):
        import server.pipeline.worker as worker_module

        cfg.MOCK_PIPELINE = False
        task_id = create_task(
            cfg.DB_PATH,
            name="resumed-source-regeneration",
            task_type="worldgen",
            config={"quality_preset": "high", "recovery_policy": "auto", "seed": 1024},
        )
        task_dir = cfg.task_dir(task_id)
        task_dir.mkdir(parents=True)
        (task_dir / "auto_recovery.json").write_text(json.dumps({
            "version": 1,
            "state": "regenerating_panorama",
            "source_attempt": 1,
            "source_seed": 1025,
            "active_scene_relpath": "recovery_attempts/attempt-001/scene",
        }))
        monkeypatch.setattr(
            worker_module,
            "get_worldgen_commands",
            lambda **_kwargs: [
                ("panogen", ["panogen"], {}, Path("/tmp")),
                ("traj_generate", ["traj"], {}, Path("/tmp")),
                ("traj_render", ["render"], {}, Path("/tmp")),
                ("video_gen", ["video"], {}, Path("/tmp")),
                ("gen_gs_data", ["data"], {}, Path("/tmp")),
                ("gs_training", ["train"], {}, Path("/tmp")),
            ],
        )
        worker = PipelineWorker(cfg)
        resumed: dict[str, object] = {}

        async def regenerate(**kwargs):
            resumed.update(kwargs)
            return True, ""

        called_stages: list[str] = []

        async def fake_stage(*, stage_name, **_kwargs):
            called_stages.append(stage_name)
            return True, ""

        monkeypatch.setattr(worker, "_run_regenerated_source_attempt", regenerate)
        monkeypatch.setattr(worker_module, "run_stage", fake_stage)
        monkeypatch.setattr(worker_module, "validate_completed_delivery", lambda *_args, **_kwargs: {"valid": True})

        asyncio.run(worker.execute_task(task_id))

        assert resumed["resume"] is True
        assert resumed["seed"] == 1025
        assert called_stages == ["gen_gs_data", "gs_training"]
        assert get_task(cfg.DB_PATH, task_id)["status"] == TaskStatus.COMPLETED.value

    def test_replacement_panorama_resume_never_runs_panogen_and_continues_training(self, cfg, monkeypatch):
        import server.pipeline.worker as worker_module

        cfg.MOCK_PIPELINE = False
        task_id = create_task(
            cfg.DB_PATH,
            name="replacement-panorama",
            task_type="worldgen_from_pano",
            config={"quality_preset": "high", "recovery_policy": "auto"},
        )
        task_dir = cfg.task_dir(task_id)
        task_dir.mkdir(parents=True)
        (task_dir / "auto_recovery.json").write_text(json.dumps({
            "version": 1,
            "state": "replacing_panorama_source",
            "source_attempt": 1,
            "active_scene_relpath": "recovery_attempts/attempt-001/scene",
        }))
        monkeypatch.setattr(
            worker_module,
            "get_worldgen_commands",
            lambda **_kwargs: [
                ("traj_generate", ["traj"], {}, Path("/tmp")),
                ("traj_render", ["render"], {}, Path("/tmp")),
                ("video_gen", ["video"], {}, Path("/tmp")),
                ("gen_gs_data", ["data"], {}, Path("/tmp")),
                ("gs_training", ["train"], {}, Path("/tmp")),
            ],
        )
        worker = PipelineWorker(cfg)
        resumed: dict[str, object] = {}

        async def replacement(**kwargs):
            resumed.update(kwargs)
            return True, ""

        called_stages: list[str] = []

        async def fake_stage(*, stage_name, **_kwargs):
            called_stages.append(stage_name)
            return True, ""

        monkeypatch.setattr(worker, "_run_replacement_panorama_attempt", replacement)
        monkeypatch.setattr(worker_module, "run_stage", fake_stage)
        monkeypatch.setattr(worker_module, "validate_completed_delivery", lambda *_args, **_kwargs: {"valid": True})

        asyncio.run(worker.execute_task(task_id))

        assert resumed["resume"] is True
        assert called_stages == ["gen_gs_data", "gs_training"]
        assert get_task(cfg.DB_PATH, task_id)["status"] == TaskStatus.COMPLETED.value

    def test_legacy_panorama_delivery_is_migrated_and_runs_both_3d_stages(self, cfg, monkeypatch):
        import server.pipeline.worker as worker_module

        cfg.MOCK_PIPELINE = False
        task_id = create_task(
            cfg.DB_PATH,
            name="panorama-delivery",
            task_type="worldgen_from_pano",
            config={"quality_preset": "high", "recovery_policy": "auto"},
        )
        task_dir = cfg.task_dir(task_id)
        task_dir.mkdir(parents=True)
        warning = "Only 3 trajectories passed; 3D training was safely skipped."
        (task_dir / "auto_recovery.json").write_text(json.dumps({
            "version": 2,
            "state": "completed",
            "delivery_tier": "panorama_only",
            "quality_warning": warning,
            "limits": {},
        }))
        monkeypatch.setattr(
            worker_module,
            "get_worldgen_commands",
            lambda **_kwargs: [
                ("gen_gs_data", ["data"], {}, Path("/tmp")),
                ("gs_training", ["train"], {}, Path("/tmp")),
            ],
        )
        monkeypatch.setattr(worker_module, "get_stage_names", lambda _task_type: ["gen_gs_data", "gs_training"])
        called_stages: list[str] = []

        async def fake_stage(*, stage_name, **_kwargs):
            called_stages.append(stage_name)
            return True, ""

        monkeypatch.setattr(worker_module, "run_stage", fake_stage)
        monkeypatch.setattr(worker_module, "validate_completed_delivery", lambda *_args, **_kwargs: {"valid": True})

        asyncio.run(PipelineWorker(cfg).execute_task(task_id))

        task = get_task(cfg.DB_PATH, task_id)
        assert task["status"] == TaskStatus.COMPLETED.value
        assert called_stages == ["gen_gs_data", "gs_training"]
        migrated = json.loads((task_dir / "auto_recovery.json").read_text())
        assert migrated["delivery_tier"] == "anchored_3d"
        assert migrated["training_mode"] == "panorama_anchored"

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

    def test_worker_replenishes_high_quality_video_candidates_until_six_pass(self, cfg, monkeypatch):
        import server.pipeline.worker as worker_module

        cfg.MOCK_PIPELINE = False
        task_dir = cfg.task_dir("video-recovery")
        logs_dir = task_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        trajectories = [
            *({"id": f"view{index}/traj0", "kind": "regular"} for index in range(9)),
            *({"id": f"target_{index}/traj0", "kind": "surround"} for index in range(8)),
            *({"id": f"wonder_{index}/traj0", "kind": "exploration"} for index in range(6)),
            *({"id": f"reconstruct_{index}/traj0", "kind": "reconstruction"} for index in range(8)),
        ]
        (task_dir / "quality_report.json").write_text(json.dumps({"planning": {"trajectories": trajectories}}))
        calls: list[list[str]] = []

        async def fake_video_stage(*, log_file, **_kwargs):
            manifest = json.loads((task_dir / "video_trajectory_whitelist.json").read_text())
            candidate_ids = manifest["candidate_trajectory_ids"]
            calls.append(candidate_ids)
            accepted_count = 2 if len(calls) == 1 else 4
            with open(log_file, "a") as log:
                for index, trajectory_id in enumerate(candidate_ids):
                    guidance = 0.92 if index < accepted_count else 0.10
                    log.write(
                        f"[Alignment Rank0] [Frame {trajectory_id}/000000] [Success] "
                        f"Guidance mask ratio: {guidance:.4f}, depth percentile error ratio: 0.02000\\n"
                    )
            return True, ""

        monkeypatch.setattr(worker_module, "run_stage", fake_video_stage)

        async def run_recovery():
            return await PipelineWorker(cfg)._run_high_quality_video_recovery(
                task_dir=task_dir,
                scene_dir=task_dir / "scene",
                log_file=logs_dir / "video_gen.log",
                cmd=["fake"],
                env={},
                cwd=Path("/tmp"),
                cancel_event=asyncio.Event(),
            )

        success, error = asyncio.run(run_recovery())

        assert success, error
        assert [len(batch) for batch in calls] == [8, 8]
        recovery = json.loads((task_dir / "video_recovery.json").read_text())
        assert recovery["status"] == "accepted"
        assert len(recovery["accepted_trajectory_ids"]) == 6


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
        env_by_stage = {name: env for name, _, env, _ in commands}

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
        assert env_by_stage["gs_training"]["TORCH_HOME"] == "/models/torch-cache"

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
        assert command[1].endswith("server/pipeline/panogen_adapter.py")
        assert command[command.index("--input-mode") + 1] == "image"
        assert command[command.index("--source-attempt") + 1] == "0"

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
        assert "Empty trajectory whitelist" in patched
        assert "Anchor output resolution" in patched

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

    def test_render_filter_patch_limits_rendering_to_new_recovery_candidates(self, tmp_path):
        upstream = tmp_path / "traj_render.py"
        upstream.write_text(
            "import argparse\nimport json\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument(\"--target_path\", default=None, type=str, help=\"target path\")\n"
            "args = parser.parse_args()\n"
            "traj_list = (glob(f\"{scene_path}/render_results/view*/traj*\") +\n"
            "             glob(f\"{scene_path}/render_results/target*/traj*\"))\n"
            "rank0_log(f\"Get {len(traj_list)} trajectories at all for {scene_path}\")\n"
            "for traj_path in traj_list:\n"
            "    if not os.path.exists(f\"{traj_path}/camera.json\"):\n"
            "        continue\n"
            "    pass\n"
            "total_render_list = glob(f\"{scene_path}/render_results/*/traj*/render.mp4\")\n"
        )
        patch_script = Path(__file__).parents[1] / "deploy/scripts/patch_hyworld_render_filter.py"

        subprocess.run([sys.executable, str(patch_script), str(upstream)], check=True)
        patched = upstream.read_text()

        assert "--trajectory_whitelist" in patched
        assert "candidate_trajectory_ids" in patched
        assert "No trajectory paths matched" in patched
        assert "--skip_exist" in patched

        subprocess.run([sys.executable, str(patch_script), str(upstream)], check=True)
        assert upstream.read_text() == patched

    def test_render_filter_patch_repairs_partial_parent(self, tmp_path):
        """An already-patched parent image must still receive the missing list filter.

        a800-v34 already carried the CLI/caption markers.  Treating the CLI marker
        as proof that every patch section existed left ``candidate_trajectory_ids``
        undefined at caption time in a800-v37.
        """
        upstream = tmp_path / "traj_render.py"
        upstream.write_text(
            "import argparse\nimport json\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument(\"--target_path\", default=None, type=str)\n"
            "# GSStudio trajectory render whitelist patch\n"
            "parser.add_argument(\"--trajectory_whitelist\", type=str, default=None)\n"
            "parser.add_argument(\"--skip_exist\", action=\"store_true\")\n"
            "args = parser.parse_args()\n"
            "traj_list = (glob(f\"{scene_path}/render_results/view*/traj*\") +\n"
            "             glob(f\"{scene_path}/render_results/target*/traj*\"))\n"
            "rank0_log(f\"Get {len(traj_list)} trajectories at all for {scene_path}\")\n"
            "for traj_path in traj_list:\n"
            "    if not os.path.exists(f\"{traj_path}/camera.json\"):\n"
            "        continue\n"
            "    pass\n"
            "total_render_list = glob(f\"{scene_path}/render_results/*/traj*/render.mp4\")\n"
            "# GSStudio trajectory caption whitelist patch\n"
            "if args.trajectory_whitelist:\n"
            "    total_render_list = [path for path in total_render_list "
            "if path in candidate_trajectory_ids]\n"
            "if args.trajectory_whitelist:\n"
            "    with open(args.trajectory_whitelist) as whitelist_file:\n"
            "        whitelist = json.load(whitelist_file)\n"
            "    candidate_trajectory_ids = set(whitelist.get('candidate_trajectory_ids', []))\n"
        )
        patch_script = Path(__file__).parents[1] / "deploy/scripts/patch_hyworld_render_filter.py"

        subprocess.run([sys.executable, str(patch_script), str(upstream)], check=True)
        patched = upstream.read_text()

        assert "# GSStudio trajectory list whitelist patch" in patched
        assert "# GSStudio trajectory whitelist load patch" in patched
        assert "candidate_trajectory_ids = set" in patched
        assert "No trajectory paths matched" in patched
        load_index = patched.index("# GSStudio trajectory whitelist load patch")
        render_index = patched.index("for traj_path in traj_list")
        caption_index = patched.index("total_render_list = [path")
        assert load_index < render_index < caption_index

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

    def test_image_patches_upstream_trajectory_rendering_for_recovery_whitelists(self):
        dockerfile = (Path(__file__).parents[1] / "Dockerfile.a800").read_text()

        assert "patch_hyworld_render_filter.py" in dockerfile
        assert "/opt/hyworld2/hyworld2/worldgen/traj_render.py" in dockerfile

    def test_gpu_rollout_manifests_target_the_immutable_v46_image(self):
        root = Path(__file__).parents[1]
        deployment = (root / "deploy/k8s/deployment-gpu.yaml").read_text()
        build = (root / "deploy/k8s/kaniko-build.yaml").read_text()

        assert "gsstudio:a800-v47" in deployment
        assert "gsstudio:a800-v47" in build
        assert "gsstudio:a800-v35" not in deployment
        assert "gsstudio:a800-v35" not in build
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

    def test_prefetch_and_startup_preflight_pin_the_vgg16_loss_checkpoint(self):
        root = Path(__file__).parents[1]
        prefetch = (root / "deploy/scripts/prefetch_models.sh").read_text()
        entrypoint = (root / "deploy/scripts/entrypoint.sh").read_text()
        deployment = (root / "deploy/k8s/deployment-gpu.yaml").read_text()

        assert "TORCH_HOME" in prefetch
        assert "download_url_to_file" in prefetch
        assert "vgg16-397923af.pth" in prefetch
        assert 'TORCH_HOME="${TORCH_HOME:-/models/torch-cache}"' in entrypoint
        assert "VGG16 offline preflight OK" in entrypoint
        assert "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0" in entrypoint
        assert 'value: "/models/torch-cache"' in deployment

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
        assert "GSStudio a800-v43" in source

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
            files={"image": ("input.png", png_bytes(), "image/png")},
        )

        assert resp.status_code == 201
        task = client.get(f"/api/tasks/{resp.json()['task_id']}").json()
        assert task["config"] == {
            "prompt": "form prompt wins",
            "quality_preset": "high",
            "recovery_policy": "auto",
            "seed": 17,
            "worldstereo_model_type": "worldstereo-memory",
            "input_mode": "image",
        }

    def test_prompt_only_worldgen_is_queued_without_a_fake_input_image(self, client, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

        response = client.post("/api/tasks", data={
            "type": "worldgen",
            "name": "prompt-only",
            "prompt": "  a robotics office  ",
            "config": "{}",
        })

        assert response.status_code == 201
        task_id = response.json()["task_id"]
        task = client.get(f"/api/tasks/{task_id}").json()
        assert task["config"]["input_mode"] == "prompt"
        assert task["config"]["prompt"] == "a robotics office"
        assert not Config().task_input_dir(task_id).joinpath("input.png").exists()

    def test_worldgen_rejects_missing_prompt_and_image_without_creating_a_task(self, client, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")
        before = client.get("/api/tasks", params={"limit": 100}).json()["total"]

        response = client.post("/api/tasks", data={"type": "worldgen", "name": "empty", "config": "{}"})

        assert response.status_code == 400
        assert client.get("/api/tasks", params={"limit": 100}).json()["total"] == before

    def test_worldgen_rejects_corrupt_image_without_creating_a_task(self, client, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")
        before = client.get("/api/tasks", params={"limit": 100}).json()["total"]

        response = client.post(
            "/api/tasks",
            data={"type": "worldgen", "name": "corrupt", "config": "{}"},
            files={"image": ("broken.png", b"not-an-image", "image/png")},
        )

        assert response.status_code == 400
        assert client.get("/api/tasks", params={"limit": 100}).json()["total"] == before

    def test_worldgen_normalizes_upload_and_preserves_original(self, client, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

        response = client.post(
            "/api/tasks",
            data={"type": "worldgen", "name": "uploaded", "config": "{}"},
            files={"image": ("camera.jpg", png_bytes(), "image/jpeg")},
        )

        assert response.status_code == 201
        input_dir = Config().task_input_dir(response.json()["task_id"])
        assert Image.open(input_dir / "input.png").mode == "RGB"
        assert (input_dir / "source_upload.jpg").is_file()

    def test_task_quality_endpoint_returns_compact_auto_recovery_state(self, client):
        from server.config import Config

        task_id = client.post("/api/tasks", data={
            "type": "worldrecon",
            "name": "quality-status",
            "config": "{}",
        }).json()["task_id"]
        task_dir = Config().task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "auto_recovery.json").write_text(json.dumps({
            "version": 1,
            "state": "generating_rescue",
            "phase": "rescue_1",
            "rescue_round": 1,
            "source_attempt": 0,
            "minimum_accepted_trajectories": 6,
            "accepted_trajectory_ids": ["view1/traj0", "view2/traj1"],
            "attempted_trajectory_ids": ["view0/traj0"],
            "pending_candidate_trajectory_ids": ["view_recovery_r1_0/traj0"],
            "generated_video_trajectory_count": 8,
            "video_seconds": 9.0,
            "limits": {"maximum_video_trajectories": 40, "maximum_video_seconds": 7200},
            "terminal_reason": None,
            "updated_at": "2026-08-11T00:00:00Z",
        }))

        response = client.get(f"/api/tasks/{task_id}/quality")

        assert response.status_code == 200
        payload = response.json()
        assert payload | {} == {
            "available": True,
            "state": "generating_rescue",
            "phase": "rescue_1",
            "rescue_round": 1,
            "source_attempt": 0,
            "accepted_trajectories": 2,
            "required_trajectories": 6,
            "attempted_trajectories": 1,
            "pending_trajectories": 1,
            "generated_video_trajectories": 8,
            "video_seconds": 9.0,
            "limits": {
                "maximum_video_trajectories": 32,
                "maximum_video_seconds": 5400,
                "maximum_source_regenerations": 1,
            },
            "terminal_reason": None,
            "delivery_tier": None,
            "best_source_attempt": 0,
            "quality_warning": None,
            "recovery_exhausted": False,
            "transient_retry_count": 0,
            "updated_at": "2026-08-11T00:00:00Z",
            "source_generation": None,
            "guaranteed_3d": True,
            "training_mode": None,
            "strict_accepted_count": 2,
            "anchor_view_count": 0,
            "training_input_manifest": None,
            "artifact_validation": None,
            "fallback_reason": None,
        }

    def test_retry_source_advances_generated_seed_without_relaxing_quality_policy(self, client, cfg):
        task_id = create_task(
            cfg.DB_PATH,
            name="source-retry",
            task_type="worldgen",
            config={"quality_preset": "high", "recovery_policy": "auto", "seed": 1024},
        )
        update_task(cfg.DB_PATH, task_id, status=TaskStatus.FAILED.value, error="source_quality_blocked")
        task_dir = cfg.task_dir(task_id)
        task_dir.mkdir(parents=True)
        (task_dir / "auto_recovery.json").write_text(json.dumps({
            "version": 1,
            "task_type": "worldgen",
            "state": "quality_blocked",
            "source_attempt": 0,
            "attempt_history": [],
            "accepted_trajectory_ids": [],
            "attempted_trajectory_ids": [],
            "batches": [],
            "minimum_accepted_trajectories": 6,
            "limits": {},
        }))

        response = client.post(f"/api/tasks/{task_id}/retry-source")

        assert response.status_code == 200
        assert response.json()["status"] == "pending"
        report = json.loads((task_dir / "auto_recovery.json").read_text())
        assert report["state"] == "regenerating_panorama"
        assert report["source_seed"] == 1025
        assert get_task(cfg.DB_PATH, task_id)["config"]["recovery_policy"] == "auto"

    def test_image_worldgen_retry_accepts_a_replacement_source(self, client, cfg):
        task_id = create_task(
            cfg.DB_PATH,
            name="image-source-retry",
            task_type="worldgen",
            config={"quality_preset": "high", "recovery_policy": "auto", "seed": 1024, "input_mode": "image"},
        )
        update_task(cfg.DB_PATH, task_id, status="failed", error="source_quality_blocked")
        task_dir = cfg.task_dir(task_id)
        task_dir.mkdir(parents=True)
        (task_dir / "auto_recovery.json").write_text(json.dumps({
            "version": 2,
            "task_type": "worldgen",
            "state": "quality_blocked",
            "source_attempt": 0,
            "attempt_history": [],
            "source_attempts": [],
            "accepted_trajectory_ids": [],
            "attempted_trajectory_ids": [],
            "pending_candidate_trajectory_ids": [],
            "batches": [],
            "minimum_accepted_trajectories": 6,
            "limits": {},
        }))

        response = client.post(
            f"/api/tasks/{task_id}/retry-source",
            files={"image": ("replacement.jpg", png_bytes((5, 6, 7)), "image/jpeg")},
        )

        assert response.status_code == 200
        replacement = task_dir / "recovery_attempts" / "attempt-001" / "input" / "input.png"
        assert Image.open(replacement).mode == "RGB"
        assert get_task(cfg.DB_PATH, task_id)["config"]["input_mode"] == "image"

    def test_corrupt_replacement_image_does_not_advance_recovery_journal(self, client, cfg):
        task_id = create_task(
            cfg.DB_PATH,
            name="corrupt-image-source-retry",
            task_type="worldgen",
            config={"quality_preset": "high", "recovery_policy": "auto", "seed": 1024, "input_mode": "image"},
        )
        update_task(cfg.DB_PATH, task_id, status="failed", error="source_quality_blocked")
        task_dir = cfg.task_dir(task_id)
        task_dir.mkdir(parents=True)
        journal = {
            "version": 2,
            "task_type": "worldgen",
            "state": "quality_blocked",
            "source_attempt": 0,
            "attempt_history": [],
            "source_attempts": [],
            "accepted_trajectory_ids": [],
            "attempted_trajectory_ids": [],
            "pending_candidate_trajectory_ids": [],
            "batches": [],
            "minimum_accepted_trajectories": 6,
            "limits": {},
        }
        journal_path = task_dir / "auto_recovery.json"
        journal_path.write_text(json.dumps(journal))
        before = journal_path.read_bytes()

        response = client.post(
            f"/api/tasks/{task_id}/retry-source",
            files={"image": ("broken.png", b"not-an-image", "image/png")},
        )

        assert response.status_code == 400
        assert journal_path.read_bytes() == before
        assert get_task(cfg.DB_PATH, task_id)["status"] == "failed"
        assert not (task_dir / "recovery_attempts" / "attempt-001").exists()

    def test_completed_degraded_delivery_can_request_a_higher_quality_source(self, client, cfg):
        task_id = create_task(
            cfg.DB_PATH,
            name="degraded-source-retry",
            task_type="worldgen",
            config={"quality_preset": "high", "recovery_policy": "auto", "seed": 1024},
        )
        update_task(cfg.DB_PATH, task_id, status=TaskStatus.COMPLETED.value)
        task_dir = cfg.task_dir(task_id)
        task_dir.mkdir(parents=True)
        (task_dir / "auto_recovery.json").write_text(json.dumps({
            "version": 2,
            "task_type": "worldgen",
            "state": "completed",
            "delivery_tier": "degraded_3d",
            "recovery_exhausted": True,
            "source_attempt": 2,
            "attempt_history": [],
            "source_attempts": [],
            "accepted_trajectory_ids": [f"view{i}/traj0" for i in range(5)],
            "attempted_trajectory_ids": [],
            "pending_candidate_trajectory_ids": [],
            "batches": [],
            "generated_video_trajectory_count": 40,
            "video_seconds": 7000,
            "minimum_accepted_trajectories": 6,
            "limits": {},
        }))

        response = client.post(f"/api/tasks/{task_id}/retry-source")

        assert response.status_code == 200
        report = json.loads((task_dir / "auto_recovery.json").read_text())
        assert report["state"] == "regenerating_panorama"
        assert report["source_attempt"] == 3
        assert report["source_seed"] == 1027
        assert report["delivery_tier"] is None
        assert report["recovery_exhausted"] is False

    def test_retry_source_requires_and_preserves_a_replacement_panorama(self, client, cfg):
        task_id = create_task(
            cfg.DB_PATH,
            name="panorama-source-retry",
            task_type="worldgen_from_pano",
            config={"quality_preset": "high", "recovery_policy": "auto"},
        )
        update_task(cfg.DB_PATH, task_id, status=TaskStatus.FAILED.value, error="source_quality_blocked")
        task_dir = cfg.task_dir(task_id)
        task_dir.mkdir(parents=True)
        (task_dir / "auto_recovery.json").write_text(json.dumps({
            "version": 1,
            "task_type": "worldgen_from_pano",
            "state": "source_quality_blocked",
            "source_attempt": 0,
            "attempt_history": [],
            "accepted_trajectory_ids": [],
            "attempted_trajectory_ids": [],
            "batches": [],
            "minimum_accepted_trajectories": 6,
            "limits": {},
        }))

        missing = client.post(f"/api/tasks/{task_id}/retry-source")
        response = client.post(
            f"/api/tasks/{task_id}/retry-source",
            files={"panorama": ("replacement.png", png_bytes((5, 6, 7)), "image/png")},
        )

        assert missing.status_code == 400
        assert response.status_code == 200
        report = json.loads((task_dir / "auto_recovery.json").read_text())
        assert report["state"] == "replacing_panorama_source"
        replacement = task_dir / "recovery_attempts" / "attempt-001" / "scene" / "panorama.png"
        with Image.open(replacement) as decoded:
            assert decoded.mode == "RGB"

    def test_worldgen_rejects_invalid_quality_preset(self, client, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")

        resp = client.post(
            "/api/tasks",
            data={
                "type": "worldgen",
                "name": "bad-quality-config",
                "config": json.dumps({"quality_preset": "ultra"}),
            },
            files={"image": ("input.png", png_bytes(), "image/png")},
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
            files={"image": ("input.png", png_bytes(), "image/png")},
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
