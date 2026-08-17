"""Stable, checksum-backed delivery manifests for external workflow clients."""
from __future__ import annotations

import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from .pipeline.preflight import CONTRACT_VERSION, write_json_atomic
from .pipeline.recovery import auto_recovery_summary


class ArtifactPathError(ValueError):
    """An artifact path is absolute or escapes its owning task directory."""


def resolve_task_artifact(task_dir: Path, file_path: str) -> Path:
    """Resolve one legacy artifact path while proving task-root containment."""
    relative = Path(file_path)
    if not file_path.strip() or relative.is_absolute():
        raise ArtifactPathError("Artifact path must be relative to the task directory")

    root = task_dir.resolve()
    candidates = (
        root / relative,
        root / "output" / relative,
        root / "scene" / relative,
        root / "input" / relative,
    )
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except FileNotFoundError:
            continue
        except ValueError as exc:
            raise ArtifactPathError("Artifact path escapes the task directory") from exc
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(file_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_role(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".spz":
        return "gaussian_spz"
    if suffix == ".ply" and ("point_cloud" in name or "gaussian" in name):
        return "gaussian_ply"
    if suffix == ".ply":
        return "point_cloud"
    if suffix in {".pt", ".pth", ".ckpt"}:
        return "checkpoint"
    if name in {"position_meta_info.json", "viewer_metadata_validation.json"}:
        return "viewer_metadata"
    if suffix == ".json":
        return "quality_or_audit_report"
    return "other"


def build_delivery_manifest(
    task_dir: Path,
    task: dict[str, Any],
    artifacts: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic manifest suitable for cross-system import."""
    quality = auto_recovery_summary(task_dir)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for artifact in artifacts if artifacts is not None else task.get("artifacts", []):
        relative = str(artifact.get("path") or "").strip()
        if not relative or relative == "delivery_manifest.json" or relative in seen:
            continue
        path = resolve_task_artifact(task_dir, relative)
        seen.add(relative)
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        entries.append({
            "name": str(artifact.get("name") or path.name),
            "path": relative,
            "type": str(artifact.get("type") or "other"),
            "role": _artifact_role(path),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
            "mime_type": mime_type,
            "download_path": f"/api/tasks/{task['id']}/artifacts/{quote(relative, safe='/')}",
        })

    contract = {}
    try:
        contract = json.loads((task_dir / "task_contract.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        contract = task.get("config") if isinstance(task.get("config"), dict) else {}
    return {
        "schema_version": "gsstudio.delivery/v1",
        "contract_version": int(contract.get("contract_version", CONTRACT_VERSION) or CONTRACT_VERSION),
        "task_id": task["id"],
        "task_name": task.get("name"),
        "task_type": task.get("type"),
        "status": task.get("status"),
        "updated_at": task.get("updated_at"),
        "success_policy": contract.get("success_policy", "strict_free_view"),
        "delivery_tier": quality.get("delivery_tier"),
        "input_lane": quality.get("input_lane") or contract.get("input_lane"),
        "product_state": quality.get("product_state"),
        "free_view_eligible": quality.get("free_view_eligible") is True,
        "error_code": task.get("error_code"),
        "error_summary": task.get("error_summary"),
        "quality_path": f"/api/tasks/{task['id']}/quality",
        "artifacts": sorted(entries, key=lambda item: (item["role"], item["path"])),
    }


def write_delivery_manifest(
    task_dir: Path,
    task: dict[str, Any],
    artifacts: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest = build_delivery_manifest(task_dir, task, artifacts)
    write_json_atomic(task_dir / "delivery_manifest.json", manifest)
    return manifest
