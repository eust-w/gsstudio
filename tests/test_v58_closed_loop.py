"""Contract-level tests for the v58 strict free-view closed loop."""
from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from server.pipeline.navigation_stress import (
    MICRO_TRANSLATION_PAIRS,
    NAVIGATION_STEPS,
    REPRESENTATIVE_VIEWS,
    _count_giant_splats,
    _load_gaussians,
    _render_projection,
    validate_stress_measurements,
)
from server.pipeline.panogen_adapter import generate_source_pool
from server.pipeline.pilot import PilotValidationError, validate_pilot_output
from server.pipeline.preflight import (
    InputPreflightError,
    MAXIMUM_TASK_SECONDS,
    MAXIMUM_WORLDSTEREO_TRAJECTORIES,
    PANORAMA_VLM_SYSTEM_PROMPT,
    normalize_task_contract,
    run_post_panorama_preflight,
    run_multiview_preflight,
    score_source_candidates,
)
import server.pipeline.preflight as preflight_module
import server.pipeline.worker as worker_module
from server.pipeline.recovery import (
    AUTO_RECOVERY_REPORT_NAME,
    prepare_auto_recovery,
    request_pilot_content_recovery,
)
from server.pipeline.gaussian_repair import assess_repair_eligibility
from server.pipeline.viewer_metadata import _camera_matrices
from server.pipeline.worldgen import get_stage_names, get_worldgen_commands
from server.pipeline.worker import (
    ISOLATED_SOURCE_COMPLETED_STAGES,
    PipelineWorker,
    _archive_current_source_reports,
    _archive_incomplete_final_training,
    _should_resume_isolated_source_attempt,
)


def _passing_vlm(_path: Path) -> dict:
    return {"vlm_score": 0.95, "hard_flags": {}, "reasons": []}


def test_panorama_vlm_prompt_distinguishes_projection_curves_from_physical_walls():
    assert "normal equirectangular projection across the entire panorama" in PANORAMA_VLM_SYSTEM_PROMPT
    assert "Do not set curved_walls merely because ceiling or floor lines" in PANORAMA_VLM_SYSTEM_PROMPT
    assert "physically" in PANORAMA_VLM_SYSTEM_PROMPT
    assert "bowed, cylindrical" in PANORAMA_VLM_SYSTEM_PROMPT


def test_server_contract_ignores_advanced_budget_and_threshold_overrides():
    config = normalize_task_contract({
        "success_policy": "anchored",
        "maximum_worldstereo_trajectories": 999,
        "maximum_task_seconds": 1,
        "pilot_steps": 1,
        "navigation_steps": 1,
    }, "worldgen")
    assert config["contract_version"] == 4
    assert config["success_policy"] == "strict_free_view"
    assert config["delivery_tier"] == "strict_3d"
    assert config["maximum_worldstereo_trajectories"] == MAXIMUM_WORLDSTEREO_TRAJECTORIES == 24
    assert config["maximum_task_seconds"] == MAXIMUM_TASK_SECONDS == 4 * 60 * 60
    assert config["pilot_steps"] == 500
    assert config["navigation_steps"] == 1000


def test_navigation_stress_decodes_flat_spz_rgb_per_gaussian(monkeypatch, tmp_path):
    cloud = SimpleNamespace(
        positions=np.asarray([0, 0, 0, 1, 0, 0], dtype=np.float32),
        scales=np.asarray([0, 0, 0, 0, 0, 0], dtype=np.float32),
        rotations=np.asarray([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
        alphas=np.asarray([0, 1], dtype=np.float32),
        colors=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
    )
    fake_spz = SimpleNamespace(
        UnpackOptions=lambda: object(),
        load_spz=lambda _path, _options: cloud,
    )
    monkeypatch.setitem(sys.modules, "spz", fake_spz)

    decoded = _load_gaussians(tmp_path / "model.spz")

    assert decoded["positions"].shape == (2, 3)
    assert decoded["colors"].shape == (2, 3)
    assert decoded["rotations"].shape == (2, 4)
    assert np.allclose(decoded["rotations"][:, 0], 1.0)
    assert not np.allclose(decoded["colors"][0], decoded["colors"][1])


def test_navigation_stress_counts_only_gsplat_projected_giants():
    radii = np.asarray([[20, 30], [151, 140], [170, 180]], dtype=np.float32)
    opacities = np.asarray([0.9, 0.06, 0.04], dtype=np.float32)

    assert _count_giant_splats(radii, opacities, width=256, height=160) == 1


def test_navigation_stress_uses_gsplat_projection_metadata():
    source = inspect.getsource(_render_projection)

    assert 'metadata.get("radii")' in source
    assert 'metadata.get("gaussian_ids")' in source
    assert "cv2.dilate" not in source


def test_worldmirror_camera_params_unwrap_matrix_records():
    matrices = _camera_matrices({
        "extrinsics": [
            {"camera_id": 0, "matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
            {"camera_id": 1, "matrix": [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        ],
    })
    assert len(matrices) == 2
    assert matrices[1][0][3] == 1


def test_v57_regenerating_source_is_adopted_and_preflighted_before_resume():
    source = inspect.getsource(PipelineWorker._run_regenerated_source_attempt)
    assert "adopted_v57_source" in source
    assert 'by_stage["input_preflight"]' in source


def test_cold_start_waits_for_supervisor_vllm_instead_of_running_preflight(monkeypatch):
    worker = PipelineWorker(SimpleNamespace(
        MANAGE_VLLM=True,
        VLLM_HOST="127.0.0.1",
        VLLM_PORT=8001,
    ))
    health = iter((False, False, True))
    supervisor_calls = []

    async def healthy():
        return next(health)

    async def supervisor(*args):
        supervisor_calls.append(args)
        return True, "started"

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(worker, "_vllm_healthy", healthy)
    monkeypatch.setattr(worker, "_supervisorctl", supervisor)
    monkeypatch.setattr(worker_module.asyncio, "sleep", no_sleep)

    asyncio.run(worker._start_vllm())

    assert supervisor_calls == []


def test_v57_resume_rebuilds_and_archives_current_source_alignment(monkeypatch, tmp_path):
    task_dir = tmp_path / "task"
    attempt_dir = task_dir / "recovery_attempts" / "attempt-001"
    attempt_scene_dir = attempt_dir / "scene"
    attempt_scene_dir.mkdir(parents=True)
    (task_dir / "logs").mkdir()
    (task_dir / "logs" / "recovery_attempt_001_video_gen.log").write_text("immutable evidence")
    (task_dir / "alignment_quality.json").write_text(json.dumps({
        "accepted_trajectory_ids": ["old/0", "old/1", "old/2"],
    }))
    expected = [f"view{i}/traj0" for i in range(7)]
    completed = {
        "source_attempts": [{
            "source_attempt": 1,
            "accepted_trajectory_ids": expected,
            "attempted_trajectory_ids": [*expected, "view7/traj0"],
        }],
    }

    def rebuild(_stage_name, **kwargs):
        alignment = {
            "accepted_trajectory_ids": expected,
            "trajectories": {
                **{item: {"accepted": True} for item in expected},
                "view7/traj0": {"accepted": False},
            },
        }
        (task_dir / "alignment_quality.json").write_text(json.dumps(alignment))
        (task_dir / "trajectory_whitelist.json").write_text(json.dumps({
            "accepted_trajectory_ids": expected,
        }))
        (task_dir / "video_trajectory_whitelist.json").write_text("{}")
        return alignment

    monkeypatch.setattr(worker_module, "postvalidate_worldgen_stage", rebuild)

    _archive_current_source_reports(
        task_dir=task_dir,
        attempt_dir=attempt_dir,
        attempt_scene_dir=attempt_scene_dir,
        source_attempt=1,
        completed_report=completed,
    )

    assert json.loads((task_dir / "trajectory_whitelist.json").read_text())["accepted_trajectory_ids"] == expected
    assert json.loads((attempt_dir / "reports" / "alignment_quality.json").read_text())[
        "accepted_trajectory_ids"
    ] == expected

    (task_dir / "alignment_quality.json").write_text('{"accepted_trajectory_ids": ["stale"]}')
    monkeypatch.setattr(
        worker_module,
        "postvalidate_worldgen_stage",
        lambda *_args, **_kwargs: pytest.fail("archived evidence should be restored directly"),
    )
    _archive_current_source_reports(
        task_dir=task_dir,
        attempt_dir=attempt_dir,
        attempt_scene_dir=attempt_scene_dir,
        source_attempt=1,
        completed_report=completed,
    )
    assert json.loads((task_dir / "alignment_quality.json").read_text())[
        "accepted_trajectory_ids"
    ] == expected


def test_promoted_completed_source_reconciles_evidence_before_gs_manifest():
    source = inspect.getsource(PipelineWorker._execute_real)
    reconciliation = source.index("_archive_current_source_reports(")
    isolated_stage_guard = source.index(
        "regenerated_source_stages = set(ISOLATED_SOURCE_COMPLETED_STAGES)",
        reconciliation,
    )
    command_construction = source.index("get_worldgen_commands(")
    assert reconciliation < isolated_stage_guard < command_construction


@pytest.mark.parametrize("state", [
    "regenerating_panorama", "generating_normal", "generating_rescue",
    "needs_rescue", "ready", "accepted", "completed",
])
def test_v57_isolated_source_resumes_from_every_durable_batch_boundary(state):
    assert _should_resume_isolated_source_attempt({
        "version": 3,
        "state": state,
        "source_attempt": 1,
        "active_scene_relpath": "recovery_attempts/attempt-001/scene",
    })


def test_terminal_or_root_scene_does_not_resume_as_isolated_source():
    assert not _should_resume_isolated_source_attempt({
        "state": "quality_blocked",
        "source_attempt": 1,
        "active_scene_relpath": "recovery_attempts/attempt-001/scene",
    })
    assert not _should_resume_isolated_source_attempt({
        "state": "ready",
        "source_attempt": 1,
        "active_scene_relpath": "scene",
    })


def test_isolated_source_resume_does_not_regenerate_v4_source_or_preflight():
    assert ISOLATED_SOURCE_COMPLETED_STAGES == {
        "source_generate", "input_preflight", "panogen", "post_panorama_preflight",
        "traj_generate", "traj_render", "video_gen",
    }


def test_unvalidated_formal_training_is_archived_before_fresh_2000_steps(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "ckpt_0123.pt").write_bytes(b"partial")
    (tmp_path / "artifact_validation.json").write_text('{"valid": false}')

    archive = _archive_incomplete_final_training(tmp_path)

    assert archive is not None
    assert (archive / "output" / "ckpt_0123.pt").read_bytes() == b"partial"
    assert (archive / "artifact_validation.json").is_file()
    assert not output.exists()
    assert json.loads((archive / "archive_report.json").read_text())["version"] == 4


def test_prompt_pool_uses_base_seed_plus_zero_through_three(tmp_path):
    class Model:
        def __init__(self):
            self.seeds = []

        def generate_image(self, *, seed, image_size, **_kwargs):
            self.seeds.append(seed)
            height, width = image_size
            return None, [Image.new("RGB", (width, height), (seed % 255, 10, 20))]

    model = Model()
    pipeline = SimpleNamespace(model=model)
    report = generate_source_pool(
        pipeline,
        input_mode="prompt",
        input_image=tmp_path / "input.png",
        candidate_dir=tmp_path / "input" / "source_candidates",
        report_path=tmp_path / "source_pool.json",
        prompt="robotics office",
        base_seed=1024,
        source_attempt=0,
    )
    assert model.seeds == [1024, 1025, 1026, 1027]
    assert [item["seed"] for item in report["candidates"]] == model.seeds
    assert all((tmp_path / item["path"]).is_file() for item in report["candidates"])


def test_source_scoring_is_55_percent_cv_and_45_percent_qwen(tmp_path, monkeypatch):
    image = tmp_path / "candidate.png"
    image.write_bytes(b"evidence")
    monkeypatch.setattr(preflight_module, "analyze_image_geometry", lambda _path: {
        "cv_score": 0.8,
        "hard_reject_reasons": [],
    })
    result = score_source_candidates(
        tmp_path,
        [{"id": "source-00", "path": "candidate.png", "seed": 7}],
        vlm_scorer=lambda _path: {"vlm_score": 0.6, "hard_flags": {}, "reasons": []},
    )
    assert result["candidates"][0]["composite_score"] == pytest.approx(0.71)
    assert result["selected_candidate_id"] == "source-00"


def test_generated_panorama_is_rejected_before_trajectory_work_when_people_appear(tmp_path, monkeypatch):
    scene = tmp_path / "scene"
    scene.mkdir()
    panorama = scene / "panorama.png"
    panorama.write_bytes(b"generated panorama evidence")
    monkeypatch.setattr(preflight_module, "analyze_image_geometry", lambda _path: {
        "cv_score": 0.95,
        "hard_reject_reasons": [],
    })

    with pytest.raises(InputPreflightError, match="post_panorama_preflight_failed"):
        run_post_panorama_preflight(
            tmp_path,
            panorama,
            host="unused",
            port=0,
            model="unused",
            vlm_scorer=lambda _path: {
                "vlm_score": 0.95,
                "hard_flags": {"person": True},
                "reasons": ["person introduced during panorama generation"],
            },
        )

    report = json.loads((tmp_path / "post_panorama_preflight.json").read_text())
    assert report["state"] == "failed"
    assert "person" in report["failure_reasons"]


def test_legacy_giant_only_stress_can_queue_renderer_reanalysis(tmp_path):
    (tmp_path / "navigation_stress.json").write_text(json.dumps({
        "state": "failed",
        "free_view_eligible": False,
        "failure_reasons": ["view_01_giant_gaussian", "view_12_giant_gaussian"],
    }))

    assessment = assess_repair_eligibility(tmp_path)

    assert assessment["eligible"] is False
    assert assessment["requires_reanalysis"] is True
    assert assessment["reason"] == "renderer_reanalysis_required"


def test_multiview_twelve_frame_boundary_and_atomic_selection(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(12):
        (input_dir / f"image_{index:04d}.png").write_bytes(b"frame")
    monkeypatch.setattr(preflight_module, "analyze_image_geometry", lambda _path: {
        "cv_score": 0.95,
        "hard_reject_reasons": [],
    })
    monkeypatch.setattr(preflight_module, "analyze_pair_geometry", lambda _left, _right: {
        "usable": True,
        "pure_rotation": False,
        "match_inliers": 100,
    })
    report = run_multiview_preflight(
        tmp_path,
        input_dir,
        host="unused",
        port=0,
        model="unused",
        vlm_scorer=_passing_vlm,
    )
    assert report["state"] == "passed"
    assert report["selected_frame_count"] == 12
    assert len(list((input_dir / "preflight_selected").iterdir())) == 12
    assert json.loads((tmp_path / "input_preflight.json").read_text())["state"] == "passed"


def test_eleven_multiview_frames_fail_before_gpu(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(11):
        (input_dir / f"image_{index:04d}.png").write_bytes(b"frame")
    with pytest.raises(InputPreflightError, match="at least 12"):
        run_multiview_preflight(tmp_path, input_dir, host="unused", port=0, model="unused")
    assert json.loads((tmp_path / "input_preflight.json").read_text())["state"] == "failed"


def test_pure_rotation_pairs_are_rejected(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for index in range(12):
        (input_dir / f"image_{index:04d}.png").write_bytes(b"frame")
    monkeypatch.setattr(preflight_module, "analyze_image_geometry", lambda _path: {
        "cv_score": 0.95,
        "hard_reject_reasons": [],
    })
    monkeypatch.setattr(preflight_module, "analyze_pair_geometry", lambda _left, _right: {
        "usable": False,
        "pure_rotation": True,
        "reason": "non_translational_or_insufficient_parallax",
    })
    with pytest.raises(InputPreflightError, match="pure_rotation"):
        run_multiview_preflight(
            tmp_path, input_dir, host="unused", port=0, model="unused", vlm_scorer=_passing_vlm,
        )


def _planning(count: int = 24) -> dict:
    return {"trajectories": [{"id": f"view{index}/traj0", "kind": "regular"} for index in range(count)]}


def test_v3_journal_migrates_to_v4_without_resetting_global_budget(tmp_path):
    (tmp_path / AUTO_RECOVERY_REPORT_NAME).write_text(json.dumps({
        "version": 3,
        "task_type": "worldgen",
        "state": "ready",
        "source_attempt": 1,
        "normal_candidate_trajectory_ids": [f"view{i}/traj0" for i in range(24)],
        "attempted_trajectory_ids": [f"view{i}/traj0" for i in range(8)],
        "accepted_trajectory_ids": [],
        "pending_candidate_trajectory_ids": [],
        "generated_video_trajectory_count": 8,
        "video_seconds": 1234.5,
        "batches": [{"phase": "normal_initial"}],
        "source_attempts": [],
        "attempt_history": [],
        "created_at": "2099-01-01T00:00:00+00:00",
    }))
    report = prepare_auto_recovery(
        tmp_path, planning=_planning(), task_type="worldgen", recovery_policy="auto",
    )
    assert report["version"] == 4
    assert report["source_attempt"] == 1
    assert report["generated_video_trajectory_count"] == 8
    assert report["video_seconds"] == 1234.5
    assert report["limits"]["maximum_video_trajectories"] == 24


def test_pilot_failure_supplements_four_before_switching_source(tmp_path):
    initial = prepare_auto_recovery(
        tmp_path, planning=_planning(), task_type="worldgen", recovery_policy="auto",
    )
    report_path = tmp_path / AUTO_RECOVERY_REPORT_NAME
    report = json.loads(report_path.read_text())
    report["pending_candidate_trajectory_ids"] = []
    report["attempted_trajectory_ids"] = initial["pending_candidate_trajectory_ids"]
    report["accepted_trajectory_ids"] = initial["pending_candidate_trajectory_ids"][:6]
    report["batches"] = [{"phase": "normal_initial"}]
    report["state"] = "completed"
    report["delivery_tier"] = "strict_3d"
    report_path.write_text(json.dumps(report))

    refill = request_pilot_content_recovery(tmp_path, planning=_planning(), reason="pilot_content_failed:first")
    assert refill["phase"] == "pilot_content_refill"
    assert len(refill["pending_candidate_trajectory_ids"]) == 4

    refill["pending_candidate_trajectory_ids"] = []
    refill["attempted_trajectory_ids"] += [f"view{i}/traj0" for i in range(8, 12)]
    report_path.write_text(json.dumps(refill))
    switch = request_pilot_content_recovery(tmp_path, planning=_planning(), reason="pilot_content_failed:again")
    assert switch["state"] == "needs_regeneration"
    assert switch["phase"] == "pilot_source_replacement"


def test_pilot_thresholds_pass_and_checkpoint_is_evaluation_only(tmp_path, monkeypatch):
    output = tmp_path / "pilot_output"
    (output / "stats").mkdir(parents=True)
    (output / "renders").mkdir()
    (output / "ply").mkdir()
    (output / "ckpts").mkdir()
    (output / "stats" / "val_step499.json").write_text(json.dumps({"psnr": 22, "ssim": 0.75, "lpips": 0.25}))
    for index in range(4):
        Image.new("RGB", (8, 8)).save(output / "renders" / f"val_step499_{index}.png")
    (output / "ply" / "point_cloud_499.ply").write_text("ply\nformat ascii 1.0\nelement vertex 50000\nend_header\n")
    (output / "ckpts" / "ckpt_499_rank0.pt").write_bytes(b"checkpoint")
    monkeypatch.setitem(sys.modules, "trimesh", SimpleNamespace(load=lambda *_args, **_kwargs: SimpleNamespace(
        vertices=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=float),
    )))
    report = validate_pilot_output(tmp_path)
    assert report["state"] == "passed"
    assert report["evaluation_only_checkpoint"] is True
    assert report["full_training_starts_from_scratch"] is True


def test_pilot_rejects_non_finite_metrics(tmp_path):
    output = tmp_path / "pilot_output" / "stats"
    output.mkdir(parents=True)
    (output / "val_step499.json").write_text(json.dumps({"psnr": float("nan"), "ssim": 1, "lpips": 0}))
    with pytest.raises(PilotValidationError, match="pilot_psnr_below_22"):
        validate_pilot_output(tmp_path)


def test_navigation_measurements_reject_giant_gaussian_and_bad_micro_pair():
    walk = {
        "completed_steps": NAVIGATION_STEPS,
        "accepted_steps": NAVIGATION_STEPS,
        "all_camera_positions_finite": True,
        "height_preserved": True,
        "final_position_inside_bounds": True,
        "collision_mesh_checked": True,
    }
    views = [{
        "finite_pixels": True,
        "giant_gaussian_count": 0,
        "transparent_ratio": 0.1,
        "black_white_clip_ratio": 0.1,
    } for _ in range(REPRESENTATIVE_VIEWS)]
    pairs = [{"lpips": 0.1, "brightness_delta": 0.05} for _ in range(MICRO_TRANSLATION_PAIRS)]
    assert validate_stress_measurements(walk, views, pairs) == []
    views[3]["giant_gaussian_count"] = 1
    pairs[2]["lpips"] = 0.16
    reasons = validate_stress_measurements(walk, views, pairs)
    assert "view_03_giant_gaussian" in reasons
    assert "micro_pair_02_lpips_above_0.15" in reasons


def test_worldgen_command_graph_has_independent_pilot_and_navigation(tmp_path):
    config = normalize_task_contract({"quality_preset": "high", "recovery_policy": "auto"}, "worldgen")
    commands = get_worldgen_commands(
        task_dir=tmp_path,
        scene_dir=tmp_path / "scene",
        input_dir=tmp_path / "input",
        config=config,
        hyworld_repo=tmp_path / "hyworld",
        vllm_host="localhost",
        vllm_port=8100,
        vllm_model="qwen",
    )
    by_stage = {name: command for name, command, _env, _cwd in commands}
    env_by_stage = {name: env for name, _command, env, _cwd in commands}
    assert get_stage_names("worldgen") == [
        "source_generate", "input_preflight", "panogen", "post_panorama_preflight",
        "traj_generate", "traj_render", "video_gen", "gen_gs_data", "gs_pilot",
        "gs_training", "navigation_stress",
    ]
    assert by_stage["gs_pilot"][by_stage["gs_pilot"].index("--max_steps") + 1] == "500"
    assert by_stage["gs_pilot"][by_stage["gs_pilot"].index("--result_dir") + 1].endswith("pilot_output")
    assert "--ckpt" not in by_stage["gs_training"]
    assert by_stage["gs_training"][by_stage["gs_training"].index("--max_steps") + 1] == "2000"
    assert by_stage["gs_training"][by_stage["gs_training"].index("--test-every") + 1] == "16"
    assert by_stage["navigation_stress"][:3] == [
        "python", "-m", "server.pipeline.gaussian_repair",
    ]
    assert env_by_stage["navigation_stress"]["PYTHONPATH"].startswith("/app")
