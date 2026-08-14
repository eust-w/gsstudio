"""Test-first coverage for durable high-quality trajectory recovery."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from server.pipeline.recovery import (
    AUTO_RECOVERY_REPORT_NAME,
    GUIDANCE_COLLAPSE_CUTOFF,
    auto_recovery_summary,
    archive_downstream_checkpoints,
    begin_source_regeneration,
    complete_auto_recovery_batch,
    desired_rescue_candidate_count,
    migrate_non_strict_delivery,
    promote_recovered_scene,
    record_recovery_render_complete,
    record_transient_retry,
    request_source_retry,
    restore_best_attempt_reports,
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


def _alignment_with_guidance(
    trajectory_ids: list[str],
    *,
    minimum_guidance_mask: float,
) -> dict:
    return {
        "accepted_trajectory_ids": [],
        "trajectories": {
            trajectory_id: {
                "accepted": False,
                "minimum_guidance_mask": minimum_guidance_mask,
                "max_relative_deviation": 0.1,
                "maximum_keyframe_depth_error_ratio": 0.01,
            }
            for trajectory_id in trajectory_ids
        },
    }


class TestAutoRecoveryStateMachine:
    def test_completed_summary_reports_the_best_source_attempt_not_the_last_attempt(self, tmp_path):
        (tmp_path / "source_generation.json").write_text(json.dumps({
            "state": "completed",
            "source_attempt": 0,
            "seed": 1024,
        }))
        latest = tmp_path / "recovery_attempts" / "attempt-001"
        latest.mkdir(parents=True)
        (latest / "source_generation.json").write_text(json.dumps({
            "state": "completed",
            "source_attempt": 1,
            "seed": 1025,
        }))
        (tmp_path / AUTO_RECOVERY_REPORT_NAME).write_text(json.dumps({
            "version": 2,
            "task_type": "worldgen",
            "state": "completed",
            "source_attempt": 1,
            "best_source_attempt": 0,
            "active_scene_relpath": "scene",
            "accepted_trajectory_ids": [],
            "attempted_trajectory_ids": [],
            "pending_candidate_trajectory_ids": [],
            "minimum_accepted_trajectories": 6,
            "delivery_tier": "anchored_3d",
            "training_mode": "panorama_anchored",
            "strict_accepted_count": 0,
            "guaranteed_3d": True,
            "limits": {},
        }))

        summary = auto_recovery_summary(tmp_path)

        assert summary["best_source_attempt"] == 0
        assert summary["source_generation"]["source_attempt"] == 0
        assert summary["source_generation"]["seed"] == 1024
        assert summary["state"] == "quality_blocked"
        assert summary["guaranteed_3d"] is False
        assert summary["free_view_eligible"] is False
        assert summary["training_mode"] is None
        assert summary["strict_accepted_count"] == 0

    def test_legacy_strict_worldgen_remains_eligible_without_a_v4_stress_report(self, tmp_path):
        (tmp_path / AUTO_RECOVERY_REPORT_NAME).write_text(json.dumps({
            "version": 3,
            "contract_version": 3,
            "task_type": "worldgen",
            "state": "completed",
            "delivery_tier": "strict_3d",
            "training_mode": "accepted_trajectories",
            "accepted_trajectory_ids": [f"view{i}/traj0" for i in range(6)],
            "attempted_trajectory_ids": [f"view{i}/traj0" for i in range(6)],
            "pending_candidate_trajectory_ids": [],
            "minimum_accepted_trajectories": 6,
            "limits": {},
        }))
        (tmp_path / "artifact_validation.json").write_text(json.dumps({"free_view_eligible": True}))
        (tmp_path / "navigation_validation.json").write_text(json.dumps({"free_view_eligible": True}))

        summary = auto_recovery_summary(tmp_path)

        assert summary["contract_version"] == 3
        assert summary["legacy_strict_validation"] is True
        assert summary["free_view_eligible"] is True

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

    def test_uniformly_weak_initial_guidance_regenerates_source_before_rescue(self, tmp_path):
        first = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )
        complete_auto_recovery_batch(
            tmp_path,
            alignment=_alignment_with_guidance(
                first["pending_candidate_trajectory_ids"],
                minimum_guidance_mask=GUIDANCE_COLLAPSE_CUTOFF - 0.1,
            ),
            duration_seconds=10,
        )

        next_step = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )

        assert next_step["state"] == "needs_regeneration"
        assert next_step["phase"] == "source_guidance_collapse"
        assert next_step["source_guidance_collapse"] is True
        assert next_step["latest_guidance_diagnostics"] == {
            "detected": True,
            "diagnostic_cutoff": GUIDANCE_COLLAPSE_CUTOFF,
            "candidate_count": 8,
            "observed_count": 8,
            "missing_trajectory_ids": [],
            "best_minimum_guidance_mask": GUIDANCE_COLLAPSE_CUTOFF - 0.1,
        }
        assert [batch["phase"] for batch in next_step["batches"]] == ["normal_initial"]

    def test_near_threshold_initial_guidance_still_uses_rescue_paths(self, tmp_path):
        first = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )
        complete_auto_recovery_batch(
            tmp_path,
            alignment=_alignment_with_guidance(
                first["pending_candidate_trajectory_ids"],
                minimum_guidance_mask=GUIDANCE_COLLAPSE_CUTOFF,
            ),
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
        assert next_step["source_guidance_collapse"] is False

    def test_missing_guidance_is_not_misclassified_as_source_collapse(self, tmp_path):
        first = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )
        alignment = _alignment_with_guidance(
            first["pending_candidate_trajectory_ids"],
            minimum_guidance_mask=0.2,
        )
        alignment["trajectories"].pop(first["pending_candidate_trajectory_ids"][0])
        complete_auto_recovery_batch(tmp_path, alignment=alignment, duration_seconds=10)

        next_step = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )

        assert next_step["state"] == "needs_rescue"
        assert next_step["source_guidance_collapse"] is False
        assert next_step["latest_guidance_diagnostics"]["missing_trajectory_ids"] == [
            first["pending_candidate_trajectory_ids"][0]
        ]

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

    def test_uploaded_panorama_is_blocked_after_two_failed_rescue_rounds(self, tmp_path):
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
        completed = prepare_auto_recovery(tmp_path, planning=_planning(), task_type="worldgen_from_pano", recovery_policy="auto")

        assert initial["phase"] == "normal_initial"
        assert rescue_one["rescue_round"] == 1
        assert rescue_two["rescue_round"] == 2
        assert completed["state"] == "quality_blocked"
        assert completed["delivery_tier"] is None
        assert completed["training_mode"] is None
        assert completed["guaranteed_3d"] is False
        assert completed["free_view_eligible"] is False
        assert completed["recovery_exhausted"] is True
        assert "source_replacement_required" in completed["terminal_reason"]

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
        record_recovery_render_complete(
            tmp_path,
            candidate_ids=["view_recovery_r1_0/traj0", "view_recovery_r2_0/traj0"],
        )

        report = begin_source_regeneration(
            tmp_path,
            seed=1025,
            scene_relpath="recovery_attempts/attempt-001/scene",
        )

        assert report["generated_video_trajectory_count"] == 20
        assert report["video_seconds"] == 90
        assert report["rendered_candidate_trajectory_ids"] == []


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

        candidates = create_conservative_rescue_trajectories(
            scene_dir,
            round_number=1,
            planning=_planning(regular=3, other=0),
            target_count=6,
        )

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

        planning = _planning(regular=3, other=0)
        first = create_conservative_rescue_trajectories(
            scene_dir, round_number=2, planning=planning, target_count=6
        )
        second = create_conservative_rescue_trajectories(
            scene_dir, round_number=2, planning=planning, target_count=6
        )

        assert first == second
        camera = json.loads((scene_dir / "render_results" / first[0]["id"] / "camera.json").read_text())
        assert camera["recovery_target_rotation_degrees"] == 12.0
        assert len(camera["extrinsic"]) == 21

    def test_uses_all_available_sources_when_fewer_than_target_exist(self, tmp_path):
        scene_dir = tmp_path / "scene"
        _write_regular_source(scene_dir, "view0", "traj0", final_angle=40)

        candidates = create_conservative_rescue_trajectories(
            scene_dir,
            round_number=1,
            planning={"trajectories": [{"id": "view0/traj0", "kind": "regular"}]},
            target_count=6,
        )

        assert [candidate["source_trajectory_id"] for candidate in candidates] == ["view0/traj0"]

    def test_accepts_traj2_and_any_regular_path_from_planning_report(self, tmp_path):
        scene_dir = tmp_path / "scene"
        _write_regular_source(scene_dir, "view0", "traj2", final_angle=40)
        _write_regular_source(scene_dir, "view_custom", "path7", final_angle=40)
        planning = {
            "trajectories": [
                {"id": "view0/traj2", "kind": "regular"},
                {"id": "view_custom/path7", "kind": "regular"},
                {"id": "view9/traj0", "kind": "surround"},
            ]
        }

        candidates = create_conservative_rescue_trajectories(
            scene_dir, round_number=1, planning=planning, target_count=6
        )

        assert {item["source_trajectory_id"] for item in candidates} == {
            "view0/traj2",
            "view_custom/path7",
        }

    def test_zero_safe_regular_sources_is_a_nonfatal_empty_batch(self, tmp_path):
        candidates = create_conservative_rescue_trajectories(
            tmp_path / "scene",
            round_number=1,
            planning={"trajectories": [{"id": "view0/traj2", "kind": "regular"}]},
            target_count=6,
        )

        assert candidates == []

    def test_current_5_of_12_fixture_reuses_four_partial_rescues_without_index_drift(self, tmp_path):
        scene_dir = tmp_path / "scene"
        source_ids = [
            "view0/traj1", "view0/traj2", "view1/traj0", "view1/traj1",
            "view1/traj2", "view2/traj0", "view2/traj2",
        ]
        for source_id in source_ids:
            view, trajectory = source_id.split("/")
            _write_regular_source(scene_dir, view, trajectory, final_angle=40)
        planning = {"trajectories": [{"id": item, "kind": "regular"} for item in source_ids]}
        for index, source_id in enumerate((
            "view0/traj1", "view1/traj0", "view1/traj1", "view2/traj0",
        )):
            source_path = scene_dir / "render_results" / source_id / "camera.json"
            recovery_dir = scene_dir / "render_results" / f"view_recovery_r1_{index}" / "traj0"
            recovery_dir.mkdir(parents=True)
            camera = json.loads(source_path.read_text())
            camera["recovery_source_trajectory"] = source_id
            camera["recovery_round"] = 1
            camera["recovery_target_rotation_degrees"] = 25.0
            (recovery_dir / "camera.json").write_text(json.dumps(camera))
            (recovery_dir.parent / "start_frame.png").write_bytes(b"frame")

        first = create_conservative_rescue_trajectories(
            scene_dir,
            round_number=1,
            planning=planning,
            target_count=desired_rescue_candidate_count(5),
        )
        second = create_conservative_rescue_trajectories(
            scene_dir,
            round_number=1,
            planning=planning,
            target_count=desired_rescue_candidate_count(5),
        )

        assert desired_rescue_candidate_count(5) == 2
        assert [item["id"] for item in first] == [
            f"view_recovery_r1_{index}/traj0" for index in range(4)
        ]
        assert second == first


class TestDeliveryTiersAndMigration:
    def _exhausted_report(self, tmp_path: Path, accepted: list[str]) -> dict:
        first = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )
        complete_auto_recovery_batch(tmp_path, alignment=_alignment(accepted), duration_seconds=1)
        report_path = tmp_path / AUTO_RECOVERY_REPORT_NAME
        report = json.loads(report_path.read_text())
        report["video_seconds"] = 4 * 60 * 60
        report_path.write_text(json.dumps(report))
        return prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )

    def test_six_passed_delivers_strict_3d(self, tmp_path):
        result = self._exhausted_report(tmp_path, [f"view{i}/traj0" for i in range(6)])
        assert result["state"] == "completed"
        assert result["delivery_tier"] == "strict_3d"

    @pytest.mark.parametrize("accepted_count", range(6))
    def test_zero_through_five_passed_are_all_quality_blocked(self, tmp_path, accepted_count):
        task_dir = tmp_path / str(accepted_count)
        task_dir.mkdir()
        result = self._exhausted_report(
            task_dir,
            [f"view{i}/traj0" for i in range(accepted_count)],
        )

        assert result["state"] == "quality_blocked"
        assert result["delivery_tier"] is None
        assert result["free_view_eligible"] is False
        assert "source_replacement_required" in result["terminal_reason"]

    def test_five_passed_is_quality_blocked(self, tmp_path):
        result = self._exhausted_report(tmp_path, [f"view{i}/traj0" for i in range(5)])
        assert result["state"] == "quality_blocked"
        assert result["delivery_tier"] is None
        assert "only 5 passed" in result["terminal_reason"]

    def test_three_passed_is_quality_blocked(self, tmp_path):
        result = self._exhausted_report(tmp_path, [f"view{i}/traj0" for i in range(3)])
        assert result["state"] == "quality_blocked"
        assert result["delivery_tier"] is None
        assert result["training_mode"] is None

    def test_zero_passed_is_quality_blocked(self, tmp_path):
        result = self._exhausted_report(tmp_path, [])
        assert result["state"] == "quality_blocked"
        assert result["delivery_tier"] is None
        assert result["training_mode"] is None
        assert result["strict_accepted_count"] == 0
        assert result["guaranteed_3d"] is False

    def test_migrates_v1_journal_without_repeating_completed_work(self, tmp_path):
        attempted = [f"view{i}/traj0" for i in range(12)]
        accepted = attempted[:5]
        (tmp_path / AUTO_RECOVERY_REPORT_NAME).write_text(json.dumps({
            "version": 1,
            "task_type": "worldgen",
            "state": "needs_rescue",
            "phase": "rescue_1",
            "rescue_round": 1,
            "source_attempt": 0,
            "active_scene_relpath": "scene",
            "minimum_accepted_trajectories": 6,
            "normal_candidate_trajectory_ids": attempted,
            "attempted_trajectory_ids": attempted,
            "accepted_trajectory_ids": accepted,
            "pending_candidate_trajectory_ids": [],
            "generated_video_trajectory_count": 12,
            "video_seconds": 3813.4,
            "batches": [
                {"phase": "normal_initial", "candidate_trajectory_ids": attempted[:8]},
                {"phase": "normal_refill", "candidate_trajectory_ids": attempted[8:]},
            ],
            "attempt_history": [],
            "limits": {
                "maximum_video_trajectories": 40,
                "maximum_video_seconds": 7200,
                "maximum_source_regenerations": 1,
            },
        }))

        result = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )

        assert result["version"] == 4
        assert result["state"] == "needs_rescue"
        assert result["rescue_round"] == 1
        assert result["attempted_trajectory_ids"] == attempted
        assert result["accepted_trajectory_ids"] == accepted
        assert result["limits"] == {
            "maximum_video_trajectories": 24,
            "maximum_video_seconds": 4 * 60 * 60,
            "maximum_source_regenerations": 2,
            "maximum_source_attempts": 3,
            "maximum_trajectories_per_source": 16,
        }

    def test_non_strict_delivery_migration_is_idempotent_and_archives_evidence(self, tmp_path):
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "gs_training.log").write_text("historical training log")
        (tmp_path / "artifact.spz").write_bytes(b"preserved")
        (tmp_path / AUTO_RECOVERY_REPORT_NAME).write_text(json.dumps({
            "version": 2,
            "task_type": "worldgen",
            "state": "completed",
            "delivery_tier": "anchored_3d",
            "accepted_trajectory_ids": [],
            "limits": {},
        }))

        first = migrate_non_strict_delivery(tmp_path)
        second = migrate_non_strict_delivery(tmp_path)

        assert second == first
        assert first["gpu_work_scheduled"] is False
        assert (tmp_path / "artifact.spz").read_bytes() == b"preserved"
        archive = tmp_path / first["archive_relpath"]
        assert (archive / "auto_recovery.json").is_file()
        assert (archive / "logs" / "gs_training.log").read_text() == "historical training log"
        migrated = json.loads((tmp_path / AUTO_RECOVERY_REPORT_NAME).read_text())
        assert migrated["state"] == "quality_blocked"
        assert migrated["delivery_tier"] is None

    def test_transient_retry_is_persisted_without_resetting_recovery_progress(self, tmp_path):
        first = prepare_auto_recovery(
            tmp_path,
            planning=_planning(),
            task_type="worldgen",
            recovery_policy="auto",
        )

        updated = record_transient_retry(
            tmp_path,
            stage="video_gen",
            error="NCCL connection reset",
            attempt=1,
        )

        assert updated["transient_retry_count"] == 1
        assert updated["pending_candidate_trajectory_ids"] == first["pending_candidate_trajectory_ids"]
        assert updated["transient_retries"][0]["stage"] == "video_gen"

    def test_budget_finalization_uses_best_attempt_not_last_attempt(self, tmp_path):
        planning = _planning()
        report = prepare_auto_recovery(
            tmp_path,
            planning=planning,
            task_type="worldgen",
            recovery_policy="auto",
        )
        report["source_attempt"] = 1
        report["active_scene_relpath"] = "recovery_attempts/attempt-001/scene"
        report["accepted_trajectory_ids"] = ["view9/traj0", "view10/traj0"]
        report["pending_candidate_trajectory_ids"] = []
        report["source_attempts"] = [
            {
                "source_attempt": 0,
                "scene_relpath": "scene",
                "accepted_trajectory_ids": [f"view{i}/traj0" for i in range(5)],
                "accepted_count": 5,
                "quality_margin": 0.20,
            },
            {
                "source_attempt": 1,
                "scene_relpath": "recovery_attempts/attempt-001/scene",
                "accepted_trajectory_ids": ["view9/traj0", "view10/traj0"],
                "accepted_count": 2,
                "quality_margin": 0.30,
            },
        ]
        report["video_seconds"] = 4 * 60 * 60
        (tmp_path / AUTO_RECOVERY_REPORT_NAME).write_text(json.dumps(report))

        completed = prepare_auto_recovery(
            tmp_path,
            planning=planning,
            task_type="worldgen",
            recovery_policy="auto",
        )

        assert completed["state"] == "quality_blocked"
        assert completed["delivery_tier"] is None
        assert completed["best_source_attempt"] == 0
        assert completed["accepted_trajectory_ids"] == [f"view{i}/traj0" for i in range(5)]

    def test_manual_replacement_wins_a_blocked_zero_coverage_tie(self, tmp_path):
        planning = _planning()
        report = prepare_auto_recovery(
            tmp_path,
            planning=planning,
            task_type="worldgen",
            recovery_policy="auto",
        )
        report["source_attempt"] = 7
        report["active_scene_relpath"] = "recovery_attempts/attempt-007/scene"
        report["manual_source_retry"] = True
        report["accepted_trajectory_ids"] = []
        report["pending_candidate_trajectory_ids"] = []
        report["source_attempts"] = [
            {
                "source_attempt": 0,
                "scene_relpath": "scene",
                "accepted_trajectory_ids": [],
                "accepted_count": 0,
                "quality_margin": 0.0,
            }
        ]
        report["video_seconds"] = 4 * 60 * 60
        (tmp_path / AUTO_RECOVERY_REPORT_NAME).write_text(json.dumps(report))

        completed = prepare_auto_recovery(
            tmp_path,
            planning=planning,
            task_type="worldgen",
            recovery_policy="auto",
        )

        assert completed["state"] == "quality_blocked"
        assert completed["delivery_tier"] is None
        assert completed["best_source_attempt"] == 7
        assert completed["best_scene_relpath"] == "recovery_attempts/attempt-007/scene"
        assert completed["accepted_trajectory_ids"] == []

    def test_restores_best_attempt_whitelist_without_copying_stale_recovery_journal(self, tmp_path):
        reports = tmp_path / "recovery_attempts" / "attempt-000" / "reports"
        reports.mkdir(parents=True)
        (reports / "trajectory_whitelist.json").write_text(json.dumps({
            "accepted_trajectory_ids": [f"view{i}/traj0" for i in range(5)]
        }))
        (reports / "alignment_quality.json").write_text(json.dumps({"source": "best"}))
        (reports / AUTO_RECOVERY_REPORT_NAME).write_text(json.dumps({"version": 1, "state": "stale"}))
        active = {
            "version": 2,
            "task_type": "worldgen",
            "state": "completed",
            "best_source_attempt": 0,
            "accepted_trajectory_ids": [f"view{i}/traj0" for i in range(5)],
            "limits": {},
        }
        (tmp_path / AUTO_RECOVERY_REPORT_NAME).write_text(json.dumps(active))

        restore_best_attempt_reports(tmp_path, source_attempt=0)

        assert json.loads((tmp_path / "alignment_quality.json").read_text())["source"] == "best"
        assert json.loads((tmp_path / AUTO_RECOVERY_REPORT_NAME).read_text())["state"] == "completed"


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
            "rendered_candidate_trajectory_ids": ["view_recovery_r1_0/traj0"],
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
        assert report["rendered_candidate_trajectory_ids"] == []
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
