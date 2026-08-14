from __future__ import annotations

import json
import math
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
        "accepted_trajectory_ids": [f"view{index}/traj0" for index in range(6)],
        "trajectories": {
            **{f"view{index}/traj0": {"accepted": True} for index in range(6)},
            "view6/traj0": {"accepted": False},
        },
    }))

    manifest = build_training_input_manifest(tmp_path, scene, "strict_3d")

    assert manifest["training_mode"] == "accepted_trajectories"
    assert manifest["accepted_trajectory_ids"] == [f"view{index}/traj0" for index in range(6)]
    assert manifest["excluded_trajectory_ids"] == ["view6/traj0"]
    assert manifest["anchor_view_count"] == 5
    assert json.loads((tmp_path / "training_input_manifest.json").read_text()) == manifest


def test_completed_worldgen_requires_validated_spz(tmp_path: Path):
    with pytest.raises(DeliveryValidationError, match="SPZ"):
        validate_completed_delivery(tmp_path, task_type="worldgen")


def test_delivery_rejects_checkpoint_from_a_different_training_step(tmp_path: Path, monkeypatch):
    _write_delivery_shell(tmp_path)
    checkpoint = tmp_path / "output" / "ckpts" / "ckpt_1999_rank0.pt"
    checkpoint.rename(checkpoint.with_name("ckpt_1500_rank0.pt"))
    monkeypatch.setitem(sys.modules, "spz", SimpleNamespace(
        UnpackOptions=lambda: object(),
        load_spz=lambda *_args: _fake_cloud(alphas=np.zeros(50_000, dtype=np.float32)),
    ))

    with pytest.raises(DeliveryValidationError, match="same training step"):
        validate_completed_delivery(tmp_path, task_type="worldgen")


def test_non_worldgen_delivery_is_not_forced_through_worldgen_validator(tmp_path: Path):
    assert validate_completed_delivery(tmp_path, task_type="worldrecon")["required"] is False


def _write_delivery_shell(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "point_cloud_1999.spz").write_bytes(b"spz")
    (output / "point_cloud_1999.ply").write_bytes(b"ply")
    (output / "ckpts").mkdir()
    (output / "ckpts" / "ckpt_1999_rank0.pt").write_bytes(b"checkpoint")
    accepted = [f"view{index}/traj0" for index in range(6)]
    (tmp_path / "alignment_quality.json").write_text(json.dumps({
        "accepted_trajectory_ids": accepted,
        "trajectories": {trajectory_id: {"accepted": True} for trajectory_id in accepted},
    }))
    (tmp_path / "training_input_manifest.json").write_text(json.dumps({
        "delivery_tier": "strict_3d",
        "training_mode": "accepted_trajectories",
        "accepted_trajectory_ids": accepted,
        "rejected_trajectories_consumed": False,
    }))
    (tmp_path / "position_meta_info.json").write_text(json.dumps({
        "center_point": [0, 0, 0],
        "facing_direction": [0, 0, 1],
        "up_direction": [0, 1, 0],
        "human_scale": 1.0,
        "kwargs": {"camera_obb": [0, 0, 0, 2, 2, 0]},
    }))
    cameras = {}
    for index in range(12):
        angle = ((index % 6) + 0.25) * math.pi / 4
        cosine = math.cos(angle)
        sine = math.sin(angle)
        camera_to_world = np.asarray([
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ])
        rotation = camera_to_world.T
        center = np.asarray([index * 0.2, 0.0, (index % 3) * 0.2])
        translation = -rotation @ center
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation
        cameras[f"view{index % 6}-traj0_{index:06d}"] = {"extrinsic": extrinsic.tolist()}
    gs_data = tmp_path / "scene" / "gs_data"
    gs_data.mkdir(parents=True)
    (gs_data / "cameras.json").write_text(json.dumps(cameras))
    stats = output / "stats"
    stats.mkdir()
    (stats / "val_step1999.json").write_text(json.dumps({"psnr": 28.0, "ssim": 0.90, "lpips": 0.10}))
    renders = output / "renders"
    renders.mkdir()
    for index in range(4):
        (renders / f"val_step1999_{index}.png").write_bytes(b"render")


def _fake_cloud(*, alphas: np.ndarray) -> SimpleNamespace:
    count = 50_000
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(0, 1, count)
    positions[:, 2] = np.linspace(0, 0.5, count) ** 2
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


def test_delivery_prefers_current_output_metadata_over_recovery_archive(tmp_path: Path, monkeypatch):
    _write_delivery_shell(tmp_path)
    archived = tmp_path / "recovery_attempts" / "attempt-002" / "previous_downstream" / "output" / "ply"
    archived.mkdir(parents=True)
    (archived / "position_meta_info.json").write_text(json.dumps({
        "center_point": [9, 9, 9],
        "facing_direction": [0, 0, 1],
        "up_direction": [0, 1, 0],
    }))
    current = tmp_path / "output" / "ply" / "position_meta_info.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps({
        "center_point": [1, 2, 3],
        "facing_direction": [0, 0, 1],
        "up_direction": [0, 1, 0],
        "human_scale": 1.0,
        "kwargs": {"camera_obb": [1, 2, 3, 2, 2, 0]},
    }))
    monkeypatch.setitem(sys.modules, "spz", SimpleNamespace(
        UnpackOptions=lambda: object(),
        load_spz=lambda *_args: _fake_cloud(alphas=np.zeros(50_000, dtype=np.float32)),
    ))

    result = validate_completed_delivery(tmp_path, task_type="worldgen")

    assert result["metadata_path"] == "output/ply/position_meta_info.json"


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


def test_camera_obb_half_extent_must_be_strictly_greater_than_threshold(tmp_path: Path, monkeypatch):
    _write_delivery_shell(tmp_path)
    metadata_path = tmp_path / "position_meta_info.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["kwargs"]["camera_obb"][3:5] = [0.1, 0.1]
    metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setitem(sys.modules, "spz", SimpleNamespace(
        UnpackOptions=lambda: object(),
        load_spz=lambda *_args: _fake_cloud(alphas=np.zeros(50_000, dtype=np.float32)),
    ))

    with pytest.raises(DeliveryValidationError, match="camera_obb_is_not_navigable"):
        validate_completed_delivery(tmp_path, task_type="worldgen")


def test_delivery_rejects_unapproved_cameras_in_gs_training_data(tmp_path: Path, monkeypatch):
    _write_delivery_shell(tmp_path)
    cameras_path = tmp_path / "scene" / "gs_data" / "cameras.json"
    cameras = json.loads(cameras_path.read_text())
    cameras["view99-traj0_000000"] = next(iter(cameras.values()))
    cameras_path.write_text(json.dumps(cameras))
    monkeypatch.setitem(sys.modules, "spz", SimpleNamespace(
        UnpackOptions=lambda: object(),
        load_spz=lambda *_args: _fake_cloud(alphas=np.zeros(50_000, dtype=np.float32)),
    ))

    with pytest.raises(DeliveryValidationError, match="unapproved_trajectory_cameras"):
        validate_completed_delivery(tmp_path, task_type="worldgen")


def test_delivery_rejects_non_finite_validation_metrics(tmp_path: Path, monkeypatch):
    _write_delivery_shell(tmp_path)
    stats = tmp_path / "output" / "stats" / "val_step1999.json"
    stats.write_text('{"psnr": NaN, "ssim": 0.95, "lpips": 0.05}')
    monkeypatch.setitem(sys.modules, "spz", SimpleNamespace(
        UnpackOptions=lambda: object(),
        load_spz=lambda *_args: _fake_cloud(alphas=np.zeros(50_000, dtype=np.float32)),
    ))

    with pytest.raises(DeliveryValidationError, match="validation_psnr_below_28"):
        validate_completed_delivery(tmp_path, task_type="worldgen")
