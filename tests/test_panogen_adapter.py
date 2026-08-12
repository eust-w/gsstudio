import hashlib
import json
from pathlib import Path

from PIL import Image

from server.pipeline.panogen_adapter import generate_source_and_panorama


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate_image(self, **kwargs):
        self.calls.append(kwargs)
        return ["reasoning"], [Image.new("RGB", (12, 8), (20, 40, 60))]


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
    assert pipeline.model.calls[0]["prompt"] == "a robotics office"
    assert pipeline.panorama_calls[0][0] == str(tmp_path / "input" / "input.png")
    assert pipeline.panorama_calls[0][1]["seed"] == 1024
    assert report["state"] == "completed"
    assert report["artifact_path"] == "input/input.png"
    assert report["sha256"] == hashlib.sha256((tmp_path / "input" / "input.png").read_bytes()).hexdigest()
    assert report["width"] == 12
    assert report["height"] == 8


def test_valid_prompt_source_checkpoint_is_reused(tmp_path):
    pipeline = FakePipeline()
    source = tmp_path / "input" / "input.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (9, 7), (1, 2, 3)).save(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report_path = tmp_path / "source_generation.json"
    report_path.write_text(json.dumps({
        "mode": "prompt",
        "state": "source_ready",
        "source_attempt": 0,
        "seed": 1024,
        "artifact_path": "input/input.png",
        "sha256": digest,
        "width": 9,
        "height": 7,
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
