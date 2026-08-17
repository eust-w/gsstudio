"""SQLite-based task persistence."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from .models import StageInfo, StageStatus, TaskStatus, TaskType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    current_stage TEXT,
    progress REAL DEFAULT 0.0,
    config TEXT DEFAULT '{}',
    stages TEXT DEFAULT '[]',
    artifacts TEXT DEFAULT '[]',
    error TEXT,
    error_code TEXT,
    error_summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_IDEMPOTENCY_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_idempotency (
    idempotency_key TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE,
    request_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class IdempotencyConflictError(ValueError):
    """An idempotency key was reused for a different task request."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(_SCHEMA)
        conn.execute(_IDEMPOTENCY_SCHEMA)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        for name in ("error_code", "error_summary"):
            if name not in columns:
                conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} TEXT")


def _get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["config"] = json.loads(d.get("config") or "{}")
    d["stages"] = json.loads(d.get("stages") or "[]")
    d["artifacts"] = json.loads(d.get("artifacts") or "[]")
    return d


def new_task_id() -> str:
    return uuid.uuid4().hex[:12]


def create_task(
    db_path: Path,
    name: str,
    task_type: str,
    config: Optional[dict] = None,
    *,
    task_id: Optional[str] = None,
) -> str:
    created_task_id, _ = create_task_idempotent(
        db_path,
        name,
        task_type,
        config,
        task_id=task_id,
    )
    return created_task_id


def create_task_idempotent(
    db_path: Path,
    name: str,
    task_type: str,
    config: Optional[dict] = None,
    *,
    task_id: Optional[str] = None,
    idempotency_key: str = "",
    request_fingerprint: str = "",
) -> tuple[str, bool]:
    """Create one task or replay the task already bound to the same request.

    ``BEGIN IMMEDIATE`` serializes competing submissions before either can
    reserve the same key, preventing workflow retries from launching duplicate
    paid GPU work.
    """
    task_id = task_id or new_task_id()
    now = _utcnow_iso()
    key = idempotency_key.strip()
    fingerprint = request_fingerprint.strip()
    with _get_conn(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if key:
            existing = conn.execute(
                "SELECT task_id, request_fingerprint FROM task_idempotency WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if existing:
                if existing["request_fingerprint"] != fingerprint:
                    conn.rollback()
                    raise IdempotencyConflictError(
                        "Idempotency-Key is already bound to a different request"
                    )
                conn.commit()
                return str(existing["task_id"]), True
        conn.execute(
            "INSERT INTO tasks (id, name, type, status, config, stages, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (task_id, name, task_type, TaskStatus.PENDING.value, json.dumps(config or {}), "[]", now, now),
        )
        if key:
            conn.execute(
                "INSERT INTO task_idempotency (idempotency_key, task_id, request_fingerprint, created_at) VALUES (?,?,?,?)",
                (key, task_id, fingerprint, now),
            )
        conn.commit()
    return task_id, False


def get_task_by_idempotency_key(db_path: Path, idempotency_key: str) -> Optional[dict]:
    key = idempotency_key.strip()
    if not key:
        return None
    with _get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT tasks.* FROM task_idempotency JOIN tasks ON tasks.id = task_idempotency.task_id "
            "WHERE task_idempotency.idempotency_key = ?",
            (key,),
        ).fetchone()
        return _row_to_dict(row) if row else None


def get_task(db_path: Path, task_id: str) -> Optional[dict]:
    with _get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _row_to_dict(row) if row else None


def list_tasks(
    db_path: Path,
    status: Optional[str] = None,
    task_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> Tuple[list, int]:
    with _get_conn(db_path) as conn:
        where_parts = []
        params = []
        if status:
            where_parts.append("status = ?")
            params.append(status)
        if task_type:
            where_parts.append("type = ?")
            params.append(task_type)

        where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        total = conn.execute(f"SELECT COUNT(*) FROM tasks {where}", params).fetchone()[0]
        order = "ASC" if status == TaskStatus.PENDING.value else "DESC"
        rows = conn.execute(
            f"SELECT * FROM tasks {where} ORDER BY created_at {order}, id {order} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        return [_row_to_dict(r) for r in rows], total


def update_task(db_path: Path, task_id: str, **updates) -> None:
    updates["updated_at"] = _utcnow_iso()

    # Serialize JSON fields
    for key in ("config", "stages", "artifacts"):
        if key in updates and not isinstance(updates[key], str):
            updates[key] = json.dumps(updates[key], default=str)

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [task_id]
    with _get_conn(db_path) as conn:
        conn.execute(f"UPDATE tasks SET {set_clause} WHERE id = ?", values)


def delete_task(db_path: Path, task_id: str) -> bool:
    with _get_conn(db_path) as conn:
        conn.execute("DELETE FROM task_idempotency WHERE task_id = ?", (task_id,))
        cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cursor.rowcount > 0


def update_stage(db_path: Path, task_id: str, stage_name: str, status: StageStatus, error: Optional[str] = None) -> None:
    task = get_task(db_path, task_id)
    if not task:
        return

    stages = [StageInfo(**s) for s in task["stages"]]
    now = datetime.now(timezone.utc)

    for s in stages:
        if s.name == stage_name:
            s.status = status
            if status == StageStatus.RUNNING:
                s.started_at = now
            elif status in (StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.SKIPPED):
                s.completed_at = now
                if s.started_at:
                    s.duration_seconds = (now - s.started_at).total_seconds()
            if error:
                s.error = error
            break

    completed = sum(1 for s in stages if s.status in (StageStatus.COMPLETED, StageStatus.SKIPPED))
    progress = completed / len(stages) if stages else 0.0

    update_task(
        db_path, task_id,
        stages=[s.model_dump(mode="json") for s in stages],
        current_stage=stage_name if status == StageStatus.RUNNING else None,
        progress=progress,
    )


def set_task_stages(db_path: Path, task_id: str, stage_names: list[str]) -> None:
    stages = [StageInfo(name=n).model_dump(mode="json") for n in stage_names]
    update_task(db_path, task_id, stages=stages)
