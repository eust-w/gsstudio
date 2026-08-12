from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.pipeline.delivery import (
    DeliveryValidationError,
    build_training_input_manifest,
    validate_completed_delivery,
)


def test_training_manifest_excludes_rejected_trajectories_and_counts_anchors(tmp_path: Path):
    scene = tmp_path / "scene"
    for bank, count in (("pano_bank", 3), ("polar_bank", 2)):
        images = scene / "render_results" / bank / "images"
        depths = scene / "render_results" / bank / "depths"
        images.mkdir(parents=True)
        depths.mkdir(parents=True)
        for index in range(count):
            (images / f"{index:04d}.png").write_bytes(b"image")
            (depths / f"{index:04d}.png").write_bytes(b"depth")
    (tmp_path / "alignment_quality.json").write_text(json.dumps({
        "accepted_trajectory_ids": ["view0/traj0"],
        "trajectories": {
            "view0/traj0": {"accepted": True},
            "view1/traj0": {"accepted": False},
        },
    }))

    manifest = build_training_input_manifest(tmp_path, scene, "degraded_3d")

    assert manifest["training_mode"] == "accepted_plus_anchors"
    assert manifest["accepted_trajectory_ids"] == ["view0/traj0"]
    assert manifest["excluded_trajectory_ids"] == ["view1/traj0"]
    assert manifest["anchor_view_count"] == 5
    assert json.loads((tmp_path / "training_input_manifest.json").read_text()) == manifest


def test_completed_worldgen_requires_validated_spz(tmp_path: Path):
    with pytest.raises(DeliveryValidationError, match="SPZ"):
        validate_completed_delivery(tmp_path, task_type="worldgen")


def test_non_worldgen_delivery_is_not_forced_through_worldgen_validator(tmp_path: Path):
    assert validate_completed_delivery(tmp_path, task_type="worldrecon")["required"] is False
