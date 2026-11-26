"""Use the API's installed runtime and explicit import roots in every stage."""
from __future__ import annotations

import os
from pathlib import Path
import sys


def stage_environment(hyworld_repo: Path, config: dict) -> dict[str, str]:
    roots = [str(Path(__file__).resolve().parents[2]), str(hyworld_repo.resolve())]
    if os.environ.get("PYTHONPATH"):
        roots.append(os.environ["PYTHONPATH"])
    return {"PYTHONPATH": os.pathsep.join(roots),
            "HF_HOME": str(Path(config.get("hf_home", "/models/hf-cache")).expanduser().resolve()),
            "TORCH_HOME": str(Path(config.get("torch_home", os.environ.get("TORCH_HOME", "/models/torch-cache"))).expanduser().resolve())}


def pipeline_commands(commands: list, hyworld_repo: Path, config: dict) -> list:
    """Bind subprocesses to this interpreter, including distributed launch."""
    base = stage_environment(hyworld_repo, config)
    normalized = []
    for name, command, environment, cwd in commands:
        command = list(command)
        if command[0] == "python":
            command[0] = sys.executable
        elif command[0] == "torchrun":
            command = [sys.executable, "-m", "torch.distributed.run", *command[1:]]
        # Existing-model validation/repair needs only this application's
        # renderer; a missing source-inference checkout must not block it.
        if name == "navigation_stress":
            cwd = Path(__file__).resolve().parents[2]
        normalized.append((name, command, {**environment, **base}, cwd.resolve()))
    return normalized
