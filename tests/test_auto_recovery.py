"""Test-first coverage for durable high-quality trajectory recovery."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from server.pipeline.recovery import (
    AUTO_RECOVERY_REPORT_NAME,
    AutoRecoveryError,
    archive_downstream_checkpoints,
    begin_source_regeneration,
    complete_auto_recovery_batch,
    promote_recovered_scene,
    request_source_retry,
    create_conservative_rescue_trajectories,
    prepare_auto_recovery,
    schedule_auto_recovery_batch,
)


def _planning(*, regular: int = 10, other: int = 3) -> dict:
    trajectories = [
        {"id": f"view{index}/traj{side}", "kind": "regular"}
        for index in range(regular)
        for side in (0, 1)
    ]
    trajectories.extend(
        {"id": f"target_{index}/traj0", "kind": "surround"}
        for index in range(other)
    )
    trajectories.extend(
        {"id": f"reconstruct_{index}/traj0", "kind": "reconstruction"}
        for index in range(other)
    )
    return {"trajectories": trajectories}


def _alignment(accepted: list[str]) -> dict:
    return {
        "accepted_trajectory_ids": accepted,
        "trajectories": {trajectory_id: {"accepted": trajectory_id in accepted} for trajectory_id in accepted},
    }


class TestAutoRecoveryStateMachine:
    def test_starts_with_eight_regular_first_candidates(self, tmp_path):
        recovery = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )

        assert recovery["state"] == "generating_normal"
        assert recovery["phase"] == "normal_initial"
        assert len(recovery["pending_candidate_trajectory_ids"]) == 8
        assert all(item.startswith("view") for item in recovery["pending_candidate_trajectory_ids"])
        assert (tmp_path / AUTO_RECOVERY_REPORT_NAME).is_file()
        assert json.loads((tmp_path / "quality_report.json").read_text())["auto_recovery"]["phase"] == "normal_initial"

    def test_low_acceptance_skips_normal_backlog_and_requests_first_rescue_round(self, tmp_path):
        first = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )
        complete_auto_recovery_batch(
            tmp_path,
            alignment=_alignment(first["pending_candidate_trajectory_ids"][:2]),
            duration_seconds=10,
        )

        next_step = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )

        assert next_step["state"] == "needs_rescue"
        assert next_step["rescue_round"] == 1
        assert next_step["pending_candidate_trajectory_ids"] == []
        assert next_step["attempted_trajectory_ids"] == first["pending_candidate_trajectory_ids"]

    def test_four_to_five_accepted_adds_only_four_normal_candidates(self, tmp_path):
        first = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )
        complete_auto_recovery_batch(
            tmp_path,
            alignment=_alignment(first["pending_candidate_trajectory_ids"][:4]),
            duration_seconds=10,
        )

        refill = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )

        assert refill["state"] == "generating_normal"
        assert refill["phase"] == "normal_refill"
        assert len(refill["pending_candidate_trajectory_ids"]) == 4

    def test_resume_keeps_pending_batch_without_double_counting(self, tmp_path):
        first = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )

        resumed = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )

        assert resumed["pending_candidate_trajectory_ids"] == first["pending_candidate_trajectory_ids"]
        assert resumed["attempted_trajectory_ids"] == []

    def test_uploaded_panorama_stops_with_source_quality_diagnosis_after_two_rescue_rounds(self, tmp_path):
        initial = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen_from_pano",
            recovery_policy="auto",
        )
        complete_auto_recovery_batch(tmp_path, alignment=_alignment([]), duration_seconds=1)
        rescue_one = prepare_auto_recovery(tmp_path, planning=_planning(), task_type="worldgen_from_pano", recovery_policy="auto")
        schedule_auto_recovery_batch(tmp_path, phase="rescue_1", candidate_ids=[f"view_recovery_r1_{i}/traj0" for i in range(6)])
        complete_auto_recovery_batch(tmp_path, alignment=_alignment([]), duration_seconds=1)
        rescue_two = prepare_auto_recovery(tmp_path, planning=_planning(), task_type="worldgen_from_pano", recovery_policy="auto")
        schedule_auto_recovery_batch(tmp_path, phase="rescue_2", candidate_ids=[f"view_recovery_r2_{i}/traj0" for i in range(6)])
        complete_auto_recovery_batch(tmp_path, alignment=_alignment([]), duration_seconds=1)
        blocked = prepare_auto_recovery(tmp_path, planning=_planning(), task_type="worldgen_from_pano", recovery_policy="auto")

        assert initial["phase"] == "normal_initial"
        assert rescue_one["rescue_round"] == 1
        assert rescue_two["rescue_round"] == 2
        assert blocked["state"] == "source_quality_blocked"
        assert "replacement panorama" in blocked["terminal_reason"]

    def test_regenerated_source_keeps_the_task_wide_video_budget(self, tmp_path):
        planning = _planning()
        first = prepare_auto_recovery(tmp_path, planning=planning, task_type="worldgen", recovery_policy="auto")
        complete_auto_recovery_batch(tmp_path, alignment=_alignment([]), duration_seconds=40)
        prepare_auto_recovery(tmp_path, planning=planning, task_type="worldgen", recovery_policy="auto")
        schedule_auto_recovery_batch(tmp_path, phase="rescue_1", candidate_ids=[f"view_recovery_r1_{i}/traj0" for i in range(6)])
        complete_auto_recovery_batch(tmp_path, alignment=_alignment([]), duration_seconds=30)
        prepare_auto_recovery(tmp_path, planning=planning, task_type="worldgen", recovery_policy="auto")
        schedule_auto_recovery_batch(tmp_path, phase="rescue_2", candidate_ids=[f"view_recovery_r2_{i}/traj0" for i in range(6)])
        complete_auto_recovery_batch(tmp_path, alignment=_alignment([]), duration_seconds=20)
        assert prepare_auto_recovery(tmp_path, planning=planning, task_type="worldgen", recovery_policy="auto")["state"] == "needs_regeneration"

        report = begin_source_regeneration(
            tmp_path,
            seed=1025,
            scene_relpath="recovery_attempts/attempt-001/scene",
        )

        assert report["generated_video_trajectory_count"] == 20
        assert report["video_seconds"] == 90


def _matrix(angle_degrees: float) -> list[list[float]]:
    angle = math.radians(angle_degrees)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return [
        [cos_angle, 0.0, sin_angle, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sin_angle, 0.0, cos_angle, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _write_regular_source(scene_dir: Path, view: str, traj: str, final_angle: float) -> None:
    source_dir = scene_dir / "render_results" / view / traj
    source_dir.mkdir(parents=True)
    frames = [_matrix(final_angle * index / 20) for index in range(21)]
    (source_dir / "camera.json").write_text(
        json.dumps(
            {
                "extrinsic": frames,
                "intrinsic": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] * 21,
                "width": 832,
                "height": 480,
                "type": "left-rotation",
            }
        )
    )
    parent = source_dir.parent
    (parent / "start_frame.png").write_bytes(b"frame")


class TestConservativeTrajectoryConstruction:
    def test_creates_six_capped_twenty_five_degree_safe_prefixes(self, tmp_path):
        scene_dir = tmp_path / "scene"
        for view in range(3):
            for traj in range(2):
                _write_regular_source(scene_dir, f"view{view}", f"traj{traj}", final_angle=40)

        candidates = create_conservative_rescue_trajectories(scene_dir, round_number=1)

        assert [candidate["id"] for candidate in candidates] == [
            f"view_recovery_r1_{index}/traj0" for index in range(6)
        ]
        for candidate in candidates:
            camera = json.loads((scene_dir / "render_results" / candidate["id"] / "camera.json").read_text())
            assert len(camera["extrinsic"]) == 21
            assert camera["recovery_source_trajectory"].startswith("view")
            assert camera["recovery_target_rotation_degrees"] == 25.0
            assert (scene_dir / "render_results" / candidate["id"].split("/")[0] / "start_frame.png").is_file()

    def test_second_round_uses_twelve_degree_prefix_and_is_idempotent(self, tmp_path):
        scene_dir = tmp_path / "scene"
        for view in range(3):
            for traj in range(2):
                _write_regular_source(scene_dir, f"view{view}", f"traj{traj}", final_angle=40)

        first = create_conservative_rescue_trajectories(scene_dir, round_number=2)
        second = create_conservative_rescue_trajectories(scene_dir, round_number=2)

        assert first == second
        camera = json.loads((scene_dir / "render_results" / first[0]["id"] / "camera.json").read_text())
        assert camera["recovery_target_rotation_degrees"] == 12.0
        assert len(camera["extrinsic"]) == 21

    def test_rejects_scene_without_six_safe_regular_sources(self, tmp_path):
        scene_dir = tmp_path / "scene"
        _write_regular_source(scene_dir, "view0", "traj0", final_angle=40)

        with pytest.raises(AutoRecoveryError, match="six"):
            create_conservative_rescue_trajectories(scene_dir, round_number=1)


class TestRecoveredSourcePromotion:
    def test_promotes_new_scene_and_archives_original_with_recoverable_journal(self, tmp_path):
        task_dir = tmp_path / "task"
        canonical_scene = task_dir / "scene"
        canonical_scene.mkdir(parents=True)
        (canonical_scene / "old.txt").write_text("original")
        regenerated_scene = task_dir / "recovery_attempts" / "attempt-001" / "scene"
        regenerated_scene.mkdir(parents=True)
        (regenerated_scene / "new.txt").write_text("regenerated")

        result = promote_recovered_scene(
            task_dir,
            source_attempt=1,
            regenerated_scene_dir=regenerated_scene,
            canonical_scene_dir=canonical_scene,
        )

        assert result["state"] == "completed"
        assert (canonical_scene / "new.txt").read_text() == "regenerated"
        assert (task_dir / "recovery_attempts" / "attempt-000" / "scene" / "old.txt").read_text() == "original"
        assert json.loads((task_dir / "source_promotion.json").read_text())["state"] == "completed"

    def test_resumes_after_original_is_archived_before_new_scene_is_promoted(self, tmp_path):
        task_dir = tmp_path / "task"
        canonical_scene = task_dir / "scene"
        archive_scene = task_dir / "recovery_attempts" / "attempt-000" / "scene"
        archive_scene.mkdir(parents=True)
        (archive_scene / "old.txt").write_text("original")
        regenerated_scene = task_dir / "recovery_attempts" / "attempt-001" / "scene"
        regenerated_scene.mkdir(parents=True)
        (regenerated_scene / "new.txt").write_text("regenerated")
        (task_dir / "source_promotion.json").write_text(json.dumps({
            "source_attempt": 1,
            "from": "recovery_attempts/attempt-001/scene",
            "to": "scene",
            "archive": "recovery_attempts/attempt-000/scene",
            "state": "original_archived",
        }))

        result = promote_recovered_scene(
            task_dir,
            source_attempt=1,
            regenerated_scene_dir=regenerated_scene,
            canonical_scene_dir=canonical_scene,
        )

        assert result["state"] == "completed"
        assert (canonical_scene / "new.txt").is_file()

    def test_preserves_a_failed_prior_attempt_when_manually_promoting_a_later_source(self, tmp_path):
        task_dir = tmp_path / "task"
        canonical_scene = task_dir / "scene"
        canonical_scene.mkdir(parents=True)
        (canonical_scene / "old.txt").write_text("original")
        failed_attempt = task_dir / "recovery_attempts" / "attempt-001" / "scene"
        failed_attempt.mkdir(parents=True)
        (failed_attempt / "failed.txt").write_text("keep this diagnosis")
        regenerated_scene = task_dir / "recovery_attempts" / "attempt-002" / "scene"
        regenerated_scene.mkdir(parents=True)
        (regenerated_scene / "new.txt").write_text("manual replacement")

        result = promote_recovered_scene(
            task_dir,
            source_attempt=2,
            regenerated_scene_dir=regenerated_scene,
            canonical_scene_dir=canonical_scene,
        )

        assert result["archive"] == "recovery_attempts/attempt-000/scene"
        assert (failed_attempt / "failed.txt").read_text() == "keep this diagnosis"
        assert (task_dir / "recovery_attempts" / "attempt-000" / "scene" / "old.txt").is_file()


class TestManualSourceRetry:
    def test_generated_source_retry_uses_the_next_deterministic_attempt_and_seed(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "auto_recovery.json").write_text(json.dumps({
            "version": 1,
            "task_type": "worldgen",
            "source_attempt": 0,
            "attempt_history": [],
            "accepted_trajectory_ids": ["view0/traj0"],
            "attempted_trajectory_ids": ["view0/traj0"],
            "batches": [{"phase": "rescue_2"}],
            "generated_video_trajectory_count": 20,
            "video_seconds": 600.0,
            "minimum_accepted_trajectories": 6,
            "limits": {},
        }))

        report = request_source_retry(task_dir, task_type="worldgen", seed=1025)

        assert report["state"] == "regenerating_panorama"
        assert report["source_attempt"] == 1
        assert report["source_seed"] == 1025
        assert report["active_scene_relpath"] == "recovery_attempts/attempt-001/scene"
        assert report["accepted_trajectory_ids"] == []
        assert report["generated_video_trajectory_count"] == 20
        assert report["video_seconds"] == 600.0
        assert report["attempt_history"][0]["accepted_trajectory_ids"] == ["view0/traj0"]

    def test_uploaded_panorama_retry_never_requests_panogen(self, tmp_path):
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "auto_recovery.json").write_text(json.dumps({
            "version": 1,
            "task_type": "worldgen_from_pano",
            "source_attempt": 0,
            "attempt_history": [],
            "accepted_trajectory_ids": [],
            "attempted_trajectory_ids": [],
            "batches": [],
            "minimum_accepted_trajectories": 6,
            "limits": {},
        }))

        report = request_source_retry(task_dir, task_type="worldgen_from_pano")

        assert report["state"] == "replacing_panorama_source"
        assert report["source_attempt"] == 1
        assert "source_seed" not in report

    def test_archives_partial_downstream_output_instead_of_overwriting_it(self, tmp_path):
        task_dir = tmp_path / "task"
        output_dir = task_dir / "output"
        output_dir.mkdir(parents=True)
        (output_dir / "partial.spz").write_bytes(b"partial")
        (task_dir / ".stage_gen_gs_data_done").write_text("done")
        (task_dir / ".stage_gs_training_done").write_text("done")
        (task_dir / "source_promotion.json").write_text(json.dumps({"state": "completed"}))

        archive_downstream_checkpoints(task_dir, source_attempt=1)

        archive = task_dir / "recovery_attempts" / "attempt-001" / "previous_downstream"
        assert (archive / "output" / "partial.spz").read_bytes() == b"partial"
        assert (archive / ".stage_gen_gs_data_done").is_file()
        assert (archive / ".stage_gs_training_done").is_file()
        assert json.loads((archive / "source_promotion.json").read_text())["state"] == "completed"
        assert not output_dir.exists()
        assert not (task_dir / "source_promotion.json").exists()
