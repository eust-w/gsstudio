import hashlib
import json
from pathlib import Path

from PIL import Image

from server.pipeline.panogen_adapter import (
    DEFAULT_SOURCE_IMAGE_HEIGHT,
    DEFAULT_SOURCE_IMAGE_WIDTH,
    SOURCE_PROMPT_MODE,
    WORLD_GEOMETRY_PROMPT_SUFFIX,
    expand_worldgen_prompt,
    generate_source_and_panorama,
)


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        height, width = kwargs["image_size"]
        return ["reasoning"], [Image.new("RGB", (width, height), (20, 40, 60))]


class FakePipeline:
    def __init__(self):
        self.model = FakeModel()
        self.panorama_calls = []

    def __call__(self, image, **kwargs):
        self.panorama_calls.append((image, kwargs))
        return Image.new("RGB", (24, 12), (60, 40, 20))


def test_prompt_mode_generates_source_then_panorama_with_one_pipeline(tmp_path):
    pipeline = FakePipeline()

    report = generate_source_and_panorama(
        pipeline,
        input_mode="prompt",
        input_image=tmp_path / "input" / "input.png",
        panorama_path=tmp_path / "scene" / "panorama.png",
        report_path=tmp_path / "source_generation.json",
        prompt="a robotics office",
        seed=1024,
        source_attempt=0,
    )

    assert pipeline.model.calls[0]["image"] is None
    assert pipeline.model.calls[0]["prompt"] == expand_worldgen_prompt("a robotics office")
    assert pipeline.model.calls[0]["image_size"] == (
        DEFAULT_SOURCE_IMAGE_HEIGHT,
        DEFAULT_SOURCE_IMAGE_WIDTH,
    )
    assert pipeline.model.calls[0]["bot_task"] == "image"
    assert pipeline.model.calls[0]["use_system_prompt"] == "dynamic"
    assert "max_new_tokens" not in pipeline.model.calls[0]
    assert pipeline.panorama_calls[0][0] == str(tmp_path / "input" / "input.png")
    assert pipeline.panorama_calls[0][1]["prompt"] == expand_worldgen_prompt("a robotics office")
    assert pipeline.panorama_calls[0][1]["seed"] == 1024
    assert report["state"] == "completed"
    assert report["artifact_path"] == "input/input.png"
    assert report["sha256"] == hashlib.sha256((tmp_path / "input" / "input.png").read_bytes()).hexdigest()
    assert report["width"] == DEFAULT_SOURCE_IMAGE_WIDTH
    assert report["height"] == DEFAULT_SOURCE_IMAGE_HEIGHT
    assert report["original_prompt"] == "a robotics office"
    assert report["effective_prompt"].endswith(WORLD_GEOMETRY_PROMPT_SUFFIX)
    assert report["source_prompt_mode"] == SOURCE_PROMPT_MODE
    assert report["requested_source_width"] == DEFAULT_SOURCE_IMAGE_WIDTH
    assert report["requested_source_height"] == DEFAULT_SOURCE_IMAGE_HEIGHT


def test_recovery_report_can_reference_canonical_uploaded_source(tmp_path):
    pipeline = FakePipeline()
    source = tmp_path / "input" / "selected_source.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (944, 512), (10, 20, 30)).save(source)
    attempt = tmp_path / "recovery_attempts" / "attempt-001"

    report = generate_source_and_panorama(
        pipeline,
        input_mode="image",
        input_image=source,
        panorama_path=attempt / "scene" / "panorama.png",
        report_path=attempt / "source_generation.json",
        prompt="",
        seed=1025,
        source_attempt=1,
    )

    assert report["state"] == "completed"
    assert report["artifact_path"] == "input/selected_source.png"
    assert (attempt / "scene" / "panorama.png").is_file()


def test_prompt_expansion_is_deterministic_and_idempotent():
    expanded = expand_worldgen_prompt("真实场景 地瓜机器人办公室")

    assert expanded.startswith("真实场景 地瓜机器人办公室")
    assert WORLD_GEOMETRY_PROMPT_SUFFIX in expanded
    assert "无玻璃材质" in expanded
    assert "无鱼眼畸变" in expanded
    assert "哑光材质" in expanded
    assert "非全景、非360度、非鱼眼、非超广角" in expanded
    assert "禁止拱门、圆柱、弧形隔断和曲面吊顶" in expanded
    assert "无玻璃门、无玻璃柜" in expanded
    assert expand_worldgen_prompt(expanded) == expanded


def test_valid_prompt_source_checkpoint_is_reused(tmp_path):
    pipeline = FakePipeline()
    source = tmp_path / "input" / "input.png"
    source.parent.mkdir(parents=True)
    Image.new(
        "RGB",
        (DEFAULT_SOURCE_IMAGE_WIDTH, DEFAULT_SOURCE_IMAGE_HEIGHT),
        (1, 2, 3),
    ).save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report_path = tmp_path / "source_generation.json"
    report_path.write_text(json.dumps({
        "mode": "prompt",
        "state": "source_ready",
        "source_attempt": 0,
        "seed": 1024,
        "artifact_path": "input/input.png",
        "sha256": digest,
        "width": DEFAULT_SOURCE_IMAGE_WIDTH,
        "height": DEFAULT_SOURCE_IMAGE_HEIGHT,
        "effective_prompt": expand_worldgen_prompt("a robotics office"),
        "source_prompt_mode": SOURCE_PROMPT_MODE,
        "requested_source_width": DEFAULT_SOURCE_IMAGE_WIDTH,
        "requested_source_height": DEFAULT_SOURCE_IMAGE_HEIGHT,
    }))

    report = generate_source_and_panorama(
        pipeline,
        input_mode="prompt",
        input_image=source,
        panorama_path=tmp_path / "scene" / "panorama.png",
        report_path=report_path,
        prompt="a robotics office",
        seed=1024,
        source_attempt=0,
    )

    assert pipeline.model.calls == []
    assert report["state"] == "completed"
    assert pipeline.panorama_calls


def test_legacy_auto_aspect_prompt_checkpoint_is_regenerated(tmp_path):
    pipeline = FakePipeline()
    source = tmp_path / "input" / "input.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (1024, 1024), (1, 2, 3)).save(source)
    report_path = tmp_path / "source_generation.json"
    report_path.write_text(json.dumps({
        "mode": "prompt",
        "source_attempt": 0,
        "seed": 1024,
        "artifact_path": "input/input.png",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "width": 1024,
        "height": 1024,
    }))

    report = generate_source_and_panorama(
        pipeline,
        input_mode="prompt",
        input_image=source,
        panorama_path=tmp_path / "scene" / "panorama.png",
        report_path=report_path,
        prompt="a robotics office",
        seed=1024,
        source_attempt=0,
    )

    assert len(pipeline.model.calls) == 1
    assert report["width"] == DEFAULT_SOURCE_IMAGE_WIDTH
    assert report["height"] == DEFAULT_SOURCE_IMAGE_HEIGHT


def test_legacy_recaption_checkpoint_is_regenerated(tmp_path):
    pipeline = FakePipeline()
    source = tmp_path / "input" / "input.png"
    source.parent.mkdir(parents=True)
    Image.new(
        "RGB",
        (DEFAULT_SOURCE_IMAGE_WIDTH, DEFAULT_SOURCE_IMAGE_HEIGHT),
        (1, 2, 3),
    ).save(source)
    report_path = tmp_path / "source_generation.json"
    report_path.write_text(json.dumps({
        "mode": "prompt",
        "source_attempt": 0,
        "seed": 1024,
        "artifact_path": "input/input.png",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "width": DEFAULT_SOURCE_IMAGE_WIDTH,
        "height": DEFAULT_SOURCE_IMAGE_HEIGHT,
        "effective_prompt": expand_worldgen_prompt("a robotics office"),
        "requested_source_width": DEFAULT_SOURCE_IMAGE_WIDTH,
        "requested_source_height": DEFAULT_SOURCE_IMAGE_HEIGHT,
    }))

    generate_source_and_panorama(
        pipeline,
        input_mode="prompt",
        input_image=source,
        panorama_path=tmp_path / "scene" / "panorama.png",
        report_path=report_path,
        prompt="a robotics office",
        seed=1024,
        source_attempt=0,
    )

    assert len(pipeline.model.calls) == 1
    assert pipeline.model.calls[0]["bot_task"] == "image"
