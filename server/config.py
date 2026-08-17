"""GSStudio configuration from environment variables."""
import os
from pathlib import Path


class Config:
    # Server
    HOST: str = os.environ.get("GSSTUDIO_HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("GSSTUDIO_PORT", "8000"))
    BASIC_AUTH_USER: str = os.environ.get("GSSTUDIO_BASIC_AUTH_USER", "")
    BASIC_AUTH_PASSWORD: str = os.environ.get("GSSTUDIO_BASIC_AUTH_PASSWORD", "")
    SERVICE_TOKEN: str = os.environ.get("GSSTUDIO_SERVICE_TOKEN", "")

    # Data directories
    DATA_DIR: Path = Path(os.environ.get("GSSTUDIO_DATA_DIR", "./data"))
    HF_HOME: str = os.environ.get("GSSTUDIO_HF_HOME", "/models/hf-cache")

    # HY-World-2.0 repo path
    HYWORLD_REPO: Path = Path(os.environ.get("GSSTUDIO_HYWORLD_REPO", "/opt/hyworld2"))
    PANOGEN_PYTHON: str = os.environ.get("GSSTUDIO_PANOGEN_PYTHON", "/opt/panogen/bin/python")

    # GPU
    NUM_GPUS: int = int(os.environ.get("GSSTUDIO_NUM_GPUS", "5"))

    # vLLM
    VLLM_HOST: str = os.environ.get("VLLM_HOST", "localhost")
    VLLM_PORT: int = int(os.environ.get("VLLM_PORT", "8100"))
    VLLM_MODEL: str = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    # When true, the worker manages the vLLM sidecar lifecycle via
    # supervisorctl (stop before panogen, restart after). Disable for
    # local dev or when vLLM is managed externally.
    MANAGE_VLLM: bool = os.environ.get("GSSTUDIO_MANAGE_VLLM", "true").lower() in ("true", "1", "yes")

    # Mock pipeline for local dev
    MOCK_PIPELINE: bool = os.environ.get("GSSTUDIO_MOCK_PIPELINE", "false").lower() in ("true", "1", "yes")

    # Database
    DB_PATH: Path = DATA_DIR / "gsstudio.db"

    # Task directories
    @property
    def tasks_dir(self) -> Path:
        return self.DATA_DIR / "tasks"

    def task_dir(self, task_id: str) -> Path:
        return self.tasks_dir / task_id

    def task_input_dir(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "input"

    def task_scene_dir(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "scene"

    def task_output_dir(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "output"

    def task_logs_dir(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "logs"


config = Config()
