"""Pydantic models for GSStudio API."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskType(str, Enum):
    WORLDGEN = "worldgen"
    WORLDGEN_FROM_PANO = "worldgen_from_pano"
    WORLDRECON = "worldrecon"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageInfo(BaseModel):
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


class TaskCreate(BaseModel):
    type: TaskType
    name: str
    prompt: Optional[str] = None
    config: dict = Field(default_factory=dict)


class TaskUpdate(BaseModel):
    name: str


class ArtifactInfo(BaseModel):
    name: str
    path: str
    size: int = 0
    type: str = "other"  # image, video, 3dgs, pointcloud, other


class TaskResponse(BaseModel):
    id: str
    name: str
    type: TaskType
    status: TaskStatus
    current_stage: Optional[str] = None
    progress: float = 0.0
    created_at: datetime
    updated_at: datetime


class TaskDetail(TaskResponse):
    stages: list[StageInfo] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)
    artifacts: list[ArtifactInfo] = Field(default_factory=list)
    error: Optional[str] = None


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    total: int


class GPUInfo(BaseModel):
    index: int
    name: str = "Unknown"
    used_by: Optional[str] = None
    vram_used_gb: float = 0.0
    vram_total_gb: float = 80.0


class GPUStatus(BaseModel):
    gpus: list[GPUInfo] = Field(default_factory=list)
    active_task: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
