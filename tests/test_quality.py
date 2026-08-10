"""Quality gates for real HY-World generation tasks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.pipeline.quality import (
    HIGH_VIDEO_CANDIDATE_BUDGET,
    HIGH_VIDEO_RECOVERY_BATCH_SIZE,
    QualityGateError,
    build_alignment_quality,
    complete_high_quality_video_recovery,
    enforce_alignment_quality,
    normalize_worldgen_config,
    postvalidate_worldgen_stage,
    prepare_high_quality_video_recovery,
    rank_high_quality_video_candidates,
    select_high_quality_video_candidates,
    validate_trajectory_plan,
    write_video_trajectory_candidates,
    write_alignment_quality,
)


def _write_camera(scene_dir: Path, view: str, traj: str) -> None:
    camera = scene_dir / "render_results" / view / traj / "camera.json"
    camera.parent.mkdir(parents=True, exist_ok=True)
    camera.write_text('{"extrinsic": [], "intrinsic": []}')


def _healthy_plan(scene_dir: Path) -> None:
    navmesh = scene_dir / "navmesh"
    navmesh.mkdir(parents=True)
    (navmesh / "reconstruct_pairs.json").write_text("[]")
    for index in range(7):
        _write_camera(scene_dir, f"view{index}", "traj0")
    _write_camera(scene_dir, "reconstruct_room", "traj0")


class TestTaskQualityConfig:
    def test_defaults_to_full_worldstereo_and_form_prompt_wins(self):
        config = normalize_worldgen_config(
            {"prompt": "advanced prompt", "quality_preset": "high", "seed": 17},
            form_prompt="form prompt",
        )

        assert config["prompt"] == "form prompt"
        assert config["quality_preset"] == "high"
        assert config["worldstereo_model_type"] == "worldstereo-memory"
        assert config["seed"] == 17

    def test_fast_preset_maps_to_dmd(self):
        config = normalize_worldgen_config({"quality_preset": "fast"}, form_prompt=None)

        assert config["quality_preset"] == "fast"
        assert config["worldstereo_model_type"] == "worldstereo-memory-dmd"

    def test_rejects_unknown_quality_preset(self):
        with pytest.raises(ValueError, match="quality_preset"):
            normalize_worldgen_config({"quality_preset": "ultra"}, form_prompt=None)


class TestTrajectoryPlanValidation:
    def test_high_quality_requires_navmesh_paths_and_non_regular_trajectory(self, tmp_path):
        scene_dir = tmp_path / "scene"
        _healthy_plan(scene_dir)
        log_file = tmp_path / "traj_generate.log"
        log_file.write_text("planned trajectories\n")

        report = validate_trajectory_plan(scene_dir, log_file, quality_preset="high")

        assert report["planning"]["valid_trajectory_count"] == 8
        assert report["planning"]["non_regular_trajectory_count"] == 1
        assert report["planning"]["by_kind"]["reconstruction"] == 1

    def test_navmesh_log_error_is_fatal_even_when_process_exited_zero(self, tmp_path):
        scene_dir = tmp_path / "scene"
        _healthy_plan(scene_dir)
        log_file = tmp_path / "traj_generate.log"
        log_file.write_text("Navmesh Error: No module named 'rtree'\n")

        with pytest.raises(QualityGateError, match="Navmesh Error"):
            validate_trajectory_plan(scene_dir, log_file, quality_preset="high")

    def test_quality_report_counts_worldnav_collision_ignore_messages(self, tmp_path):
        scene_dir = tmp_path / "scene"
        _healthy_plan(scene_dir)
        log_file = tmp_path / "traj_generate.log"
        log_file.write_text(
            "candidate collision discarded\\n"
            "Too many collisions in view0/traj0, ignore...\\n"
        )

        report = validate_trajectory_plan(scene_dir, log_file, quality_preset="high")

        assert report["planning"]["collision_dropped_count"] == 2

    def test_high_quality_video_candidates_include_alignment_recovery_capacity(self, tmp_path):
        scene_dir = tmp_path / "scene"
        navmesh = scene_dir / "navmesh"
        navmesh.mkdir(parents=True)
        (navmesh / "reconstruct_pairs.json").write_text("[]")
        for index in range(8):
            _write_camera(scene_dir, f"view{index}", "traj0")
        for index in range(4):
            _write_camera(scene_dir, f"target_object_{index}", "traj0")
            _write_camera(scene_dir, f"wonder_{index}", "traj0")
            _write_camera(scene_dir, f"reconstruct_{index}", "traj0")
        log_file = tmp_path / "traj_generate.log"
        log_file.write_text("planned trajectories\\n")

        report = validate_trajectory_plan(scene_dir, log_file, quality_preset="high")
        candidates = select_high_quality_video_candidates(report["planning"])
        manifest = write_video_trajectory_candidates(tmp_path, candidates)

        assert len(candidates) == 16
        assert {candidate["kind"] for candidate in candidates} == {
            "regular", "surround", "exploration", "reconstruction",
        }
        assert {
            kind: sum(candidate["kind"] == kind for candidate in candidates)
            for kind in {candidate["kind"] for candidate in candidates}
        } == {
            "regular": 8,
            "surround": 2,
            "exploration": 2,
            "reconstruction": 4,
        }
        assert json.loads(manifest.read_text())["candidate_trajectory_ids"] == [
            candidate["id"] for candidate in candidates
        ]

    def test_missing_reconstruct_pairs_is_fatal(self, tmp_path):
        scene_dir = tmp_path / "scene"
        _healthy_plan(scene_dir)
        (scene_dir / "navmesh" / "reconstruct_pairs.json").unlink()
        log_file = tmp_path / "traj_generate.log"
        log_file.write_text("planned trajectories\n")

        with pytest.raises(QualityGateError, match="reconstruct_pairs.json"):
            validate_trajectory_plan(scene_dir, log_file, quality_preset="high")


class TestAlignmentQuality:
    def test_high_quality_rejects_bad_alignment_and_writes_whitelist(self, tmp_path):
        log_text = "\n".join(
            [
                "view0/traj0 (2 frames): k=[0.9,1.0], b=[0.0,0.1], max_abs_dev=[0.01,0.02], max_rel_dev=[1.0%,10.0%]",
                "[Alignment Rank0] [Frame view0/traj0/000000] [Success] Guidance mask ratio: 0.9200, depth percentile error ratio: 0.02000",
                "[Alignment Rank0] [Frame view0/traj0/000020] [Success] Guidance mask ratio: 0.9100, depth percentile error ratio: 0.03000",
                "view1/traj0 (2 frames): k=[0.9,1.0], b=[0.0,0.1], max_abs_dev=[0.01,0.02], max_rel_dev=[1.0%,60.0%]",
                "[Alignment Rank0] [Frame view1/traj0/000000] [Success] Guidance mask ratio: 0.7900, depth percentile error ratio: 0.01000",
                "[Alignment Rank0] [Frame view1/traj0/000020] [Success] Guidance mask ratio: 0.9000, depth percentile error ratio: 0.15000",
            ]
        )

        report = build_alignment_quality(log_text, quality_preset="high")

        assert report["accepted_trajectory_ids"] == ["view0/traj0"]
        rejected = report["trajectories"]["view1/traj0"]
        assert rejected["accepted"] is False
        assert set(rejected["rejection_reasons"]) == {
            "guidance_mask_below_0.80",
            "depth_relative_deviation_above_0.50",
            "keyframe_depth_error_ratio_above_0.10",
        }

        alignment_path, manifest_path = write_alignment_quality(tmp_path, report)
        assert json.loads(alignment_path.read_text())["accepted_trajectory_ids"] == ["view0/traj0"]
        assert json.loads(manifest_path.read_text())["accepted_trajectory_ids"] == ["view0/traj0"]

    def test_high_quality_requires_six_accepted_trajectories(self):
        log_text = "\n".join(
            f"[Alignment Rank0] [Frame view{index}/traj0/000000] [Success] Guidance mask ratio: 0.9000, depth percentile error ratio: 0.02000"
            for index in range(5)
        )

        report = build_alignment_quality(log_text, quality_preset="high")
        with pytest.raises(QualityGateError, match="at least 6"):
            enforce_alignment_quality(report, quality_preset="high")

    def test_stage_postvalidation_persists_reports_before_failing(self, tmp_path):
        scene_dir = tmp_path / "scene"
        _healthy_plan(scene_dir)
        log_file = tmp_path / "traj_generate.log"
        log_file.write_text("Navmesh Error: No module named 'rtree'\n")

        with pytest.raises(QualityGateError):
            postvalidate_worldgen_stage(
                "traj_generate",
                task_dir=tmp_path,
                scene_dir=scene_dir,
                log_file=log_file,
                quality_preset="high",
            )

        report = json.loads((tmp_path / "quality_report.json").read_text())
        assert report["planning"]["log_errors"] == ["Navmesh Error: No module named 'rtree'"]

    def test_video_postvalidation_writes_accepted_trajectory_whitelist(self, tmp_path):
        lines = []
        for index in range(6):
            lines.extend(
                [
                    f"view{index}/traj0 (2 frames): k=[0.9,1.0], b=[0.0,0.1], max_abs_dev=[0.01,0.02], max_rel_dev=[1.0%,10.0%]",
                    f"[Alignment Rank0] [Frame view{index}/traj0/000000] [Success] Guidance mask ratio: 0.9200, depth percentile error ratio: 0.02000",
                    f"[Alignment Rank0] [Frame view{index}/traj0/000020] [Success] Guidance mask ratio: 0.9100, depth percentile error ratio: 0.03000",
                ]
            )
        log_file = tmp_path / "video_gen.log"
        log_file.write_text("\n".join(lines))

        postvalidate_worldgen_stage(
            "video_gen",
            task_dir=tmp_path,
            scene_dir=tmp_path / "scene",
            log_file=log_file,
            quality_preset="high",
        )

        manifest = json.loads((tmp_path / "trajectory_whitelist.json").read_text())
        assert manifest["accepted_trajectory_ids"] == [f"view{index}/traj0" for index in range(6)]
        report = json.loads((tmp_path / "quality_report.json").read_text())
        assert report["alignment"]["accepted_trajectory_ids"] == manifest["accepted_trajectory_ids"]


class TestHighQualityVideoRecovery:
    @staticmethod
    def _planning() -> dict:
        trajectories = []
        for index in range(9):
            trajectories.append({"id": f"view{index}/traj0", "kind": "regular"})
        for index in range(8):
            trajectories.append({"id": f"target_{index}/traj0", "kind": "surround"})
        for index in range(6):
            trajectories.append({"id": f"wonder_{index}/traj0", "kind": "exploration"})
        for index in range(8):
            trajectories.append({"id": f"reconstruct_{index}/traj0", "kind": "reconstruction"})
        return {"trajectories": trajectories}

    def test_replenishes_untried_candidates_after_legacy_alignment_failure(self, tmp_path):
        planning = self._planning()
        ranked = rank_high_quality_video_candidates(planning)
        initial_ids = [candidate["id"] for candidate in ranked[:16]]
        accepted_ids = ["view0/traj0", "view1/traj0"]

        (tmp_path / "quality_report.json").write_text(
            json.dumps({"planning": planning}, indent=2) + "\n"
        )
        (tmp_path / "alignment_quality.json").write_text(
            json.dumps(
                {
                    "accepted_trajectory_ids": accepted_ids,
                    "trajectories": {trajectory_id: {} for trajectory_id in initial_ids},
                },
                indent=2,
            )
            + "\n"
        )
        write_video_trajectory_candidates(
            tmp_path,
            [candidate for candidate in ranked if candidate["id"] in initial_ids],
        )

        recovery = prepare_high_quality_video_recovery(tmp_path)

        assert recovery["status"] == "recovering"
        assert recovery["candidate_budget"] == min(HIGH_VIDEO_CANDIDATE_BUDGET, len(ranked))
        assert recovery["accepted_trajectory_ids"] == accepted_ids
        assert len(recovery["next_candidate_trajectory_ids"]) == HIGH_VIDEO_RECOVERY_BATCH_SIZE
        assert not set(recovery["next_candidate_trajectory_ids"]).intersection(initial_ids)
        assert recovery["next_candidate_trajectory_ids"][:2] == [
            "view8/traj0",
            "target_2/traj0",
        ]
        manifest = json.loads((tmp_path / "video_trajectory_whitelist.json").read_text())
        assert manifest["candidate_trajectory_ids"] == recovery["next_candidate_trajectory_ids"]

    def test_recovery_keeps_a_pending_batch_on_retry_then_records_its_result(self, tmp_path):
        planning = self._planning()
        (tmp_path / "quality_report.json").write_text(
            json.dumps({"planning": planning}, indent=2) + "\n"
        )

        first = prepare_high_quality_video_recovery(tmp_path)
        retry = prepare_high_quality_video_recovery(tmp_path)

        assert first["status"] == "initial"
        assert len(first["next_candidate_trajectory_ids"]) == 16
        assert retry["next_candidate_trajectory_ids"] == first["next_candidate_trajectory_ids"]

        alignment = {
            "accepted_trajectory_ids": first["next_candidate_trajectory_ids"][:6],
            "trajectories": {
                trajectory_id: {"accepted": trajectory_id in first["next_candidate_trajectory_ids"][:6]}
                for trajectory_id in first["next_candidate_trajectory_ids"]
            },
        }
        completed = complete_high_quality_video_recovery(tmp_path, alignment)

        assert completed["status"] == "accepted"
        assert completed["pending_candidate_trajectory_ids"] == []
        assert completed["attempted_trajectory_ids"] == first["next_candidate_trajectory_ids"]
