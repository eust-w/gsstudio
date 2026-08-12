from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
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


def _write_delivery_shell(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "cloud.spz").write_bytes(b"spz")
    (output / "cloud.ply").write_bytes(b"ply")
    (output / "ckpt.pt").write_bytes(b"checkpoint")
    (tmp_path / "position_meta_info.json").write_text(json.dumps({
        "center_point": [0, 0, 0],
        "facing_direction": [0, 0, 1],
        "up_direction": [0, 1, 0],
    }))


def _fake_cloud(*, alphas: np.ndarray) -> SimpleNamespace:
    count = 50_000
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0, 1, count)
    return SimpleNamespace(
        num_points=count,
        positions=positions,
        colors=np.zeros((count, 3), dtype=np.float32),
        alphas=alphas,
        scales=np.zeros((count, 3), dtype=np.float32),
        rotations=np.zeros((count, 4), dtype=np.float32),
    )


def test_spz_logit_alpha_endpoints_are_valid_finite_opacities(tmp_path: Path, monkeypatch):
    _write_delivery_shell(tmp_path)
    alphas = np.zeros(50_000, dtype=np.float32)
    alphas[:2] = [-np.inf, np.inf]
    cloud = _fake_cloud(alphas=alphas)
    monkeypatch.setitem(sys.modules, "spz", SimpleNamespace(
        UnpackOptions=lambda: object(),
        load_spz=lambda *_args: cloud,
    ))

    result = validate_completed_delivery(tmp_path, task_type="worldgen")

    assert result["valid"] is True
    assert result["opacity_logit_endpoint_count"] == 2


def test_spz_nan_alpha_is_rejected(tmp_path: Path, monkeypatch):
    _write_delivery_shell(tmp_path)
    alphas = np.zeros(50_000, dtype=np.float32)
    alphas[0] = np.nan
    cloud = _fake_cloud(alphas=alphas)
    monkeypatch.setitem(sys.modules, "spz", SimpleNamespace(
        UnpackOptions=lambda: object(),
        load_spz=lambda *_args: cloud,
    ))

    with pytest.raises(DeliveryValidationError, match="opacity"):
        validate_completed_delivery(tmp_path, task_type="worldgen")
