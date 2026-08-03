"""Durable, atomic state for a single resumable video generation run."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.models import (
    RunManifest,
    RunOptions,
    RunStatus,
    ShotStatus,
    ShotTaskState,
    validate_storyboard,
)


MANIFEST_FILENAME = "run_manifest.json"
STORYBOARD_FILENAME = "storyboard.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunWorkspace:
    """Own run identity, checkpoints, recovery, and task truth behind one interface."""

    def __init__(self, path: Path, manifest: RunManifest):
        self.path = path.resolve()
        self.manifest_path = self.path / MANIFEST_FILENAME
        self.manifest = manifest

    @classmethod
    def create(cls, output_root: Path, options: RunOptions) -> "RunWorkspace":
        output_root = output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = output_root / run_id
        path.mkdir(parents=False, exist_ok=False)
        now = _now()
        workspace = cls(
            path,
            RunManifest(
                run_id=run_id,
                created_at=now,
                updated_at=now,
                options=options,
            ),
        )
        workspace._save_manifest()
        return workspace

    @classmethod
    def resume(cls, path: str | Path) -> "RunWorkspace":
        resolved = Path(path).expanduser().resolve()
        manifest_path = resolved / MANIFEST_FILENAME
        if not resolved.is_dir():
            raise ValueError(f"Resume workspace does not exist: {resolved}")
        if not manifest_path.is_file():
            raise ValueError(f"Resume workspace has no {MANIFEST_FILENAME}: {resolved}")
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = RunManifest.model_validate(data)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Invalid run manifest: {manifest_path}: {exc}") from exc
        return cls(resolved, manifest)

    @property
    def options(self) -> RunOptions:
        return self.manifest.options

    def load_storyboard(self) -> dict[str, Any]:
        path = self.path / STORYBOARD_FILENAME
        if not path.is_file():
            raise ValueError(f"Resume workspace has no validated storyboard: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid storyboard JSON: {path}: {exc}") from exc
        validated = validate_storyboard(data)
        from pipeline.storyboard import _apply_coverage_defaults

        _apply_coverage_defaults(validated["shots"])
        for shot in validated["shots"]:
            state = self.manifest.shots.get(str(shot["shot_id"]))
            if not state or state.status != ShotStatus.success:
                continue
            observed = {
                key: value
                for key, value in state.observed_end_state.items()
                if str(value).strip()
            }
            if observed:
                shot["observed_end_state"] = observed
        return validate_storyboard(validated)

    def save_storyboard(self, storyboard: dict[str, Any]) -> dict[str, Any]:
        validated = validate_storyboard(storyboard)
        from pipeline.storyboard import _apply_coverage_defaults

        _apply_coverage_defaults(validated["shots"])
        validated = validate_storyboard(validated)
        self._atomic_write_json(self.path / STORYBOARD_FILENAME, validated)
        for shot in validated["shots"]:
            key = str(shot["shot_id"])
            self.manifest.shots.setdefault(key, ShotTaskState(shot_id=shot["shot_id"]))
        self.checkpoint("storyboard_ready", RunStatus.running)
        return validated

    def checkpoint(self, stage: str, status: RunStatus = RunStatus.running) -> None:
        self.manifest.stage = stage
        self.manifest.status = status
        self.manifest.error = None
        self._save_manifest()

    def record_shot(
        self,
        *,
        shot_id: int,
        status: str,
        provider_task_id: str | None = None,
        local_path: str | None = None,
        last_frame_url: str | None = None,
        quality_score: int = 0,
        technical_quality_score: int = 0,
        semantic_accepted: bool | None = None,
        observed_end_state: dict[str, str] | None = None,
        model_used: str = "",
        resolution_used: str = "",
        attempts: int = 0,
        errors: list[str] | None = None,
    ) -> None:
        key = str(shot_id)
        previous = self.manifest.shots.get(key)
        relative_path = self._relative_to_workspace(local_path)
        previous_errors = list(previous.errors) if previous else []
        merged_errors = previous_errors + [
            error for error in (errors or []) if error not in previous_errors
        ]
        self.manifest.shots[key] = ShotTaskState(
            shot_id=shot_id,
            status=ShotStatus(status),
            provider_task_id=provider_task_id,
            local_path=relative_path or (previous.local_path if previous else None),
            last_frame_url=self._relative_to_workspace(last_frame_url),
            quality_score=quality_score,
            technical_quality_score=technical_quality_score,
            semantic_accepted=semantic_accepted,
            observed_end_state=observed_end_state or {},
            model_used=model_used,
            resolution_used=resolution_used,
            attempts=max(attempts, previous.attempts if previous else 0),
            errors=merged_errors,
        )
        self.manifest.stage = "generating_shots"
        self.manifest.status = RunStatus.running
        self._save_manifest()

    def resumable_provider_tasks(self) -> dict[int, str]:
        return {
            state.shot_id: state.provider_task_id
            for state in self.manifest.shots.values()
            if state.status == ShotStatus.running and state.provider_task_id
        }

    def accepted_shot_artifacts(self) -> dict[int, dict[str, Any]]:
        """Return successful local takes as authoritative resume inputs."""
        accepted: dict[int, dict[str, Any]] = {}
        for state in self.manifest.shots.values():
            if state.status != ShotStatus.success or not state.local_path:
                continue
            local_path = self._resolve_artifact(state.local_path)
            if not local_path or not Path(local_path).is_file():
                continue
            accepted[state.shot_id] = {
                "local_path": local_path,
                "last_frame_url": self._resolve_artifact(state.last_frame_url),
                "quality_score": state.quality_score,
                "technical_quality_score": state.technical_quality_score,
                "semantic_accepted": state.semantic_accepted,
                "observed_end_state": dict(state.observed_end_state),
                "model_used": state.model_used,
                "resolution_used": state.resolution_used,
                "attempts": state.attempts,
                "errors": list(state.errors),
            }
        return accepted

    def mark_failed(self, error: str) -> None:
        self.manifest.status = RunStatus.failed
        self.manifest.error = error
        self._save_manifest()

    def mark_interrupted(self, error: str = "Run interrupted") -> None:
        self.manifest.status = RunStatus.interrupted
        self.manifest.error = error
        self._save_manifest()

    def mark_succeeded(self, final_path: str) -> None:
        self.manifest.status = RunStatus.succeeded
        self.manifest.stage = "completed"
        self.manifest.final_path = self._relative_to_workspace(final_path)
        self.manifest.error = None
        self._save_manifest()

    def completed_output(self) -> str | None:
        if self.manifest.status != RunStatus.succeeded or not self.manifest.final_path:
            return None
        path = self.path / self.manifest.final_path
        return str(path) if path.is_file() and path.stat().st_size > 0 else None

    def _relative_to_workspace(self, value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            return value
        try:
            return str(path.resolve().relative_to(self.path))
        except ValueError:
            return value

    def _resolve_artifact(self, value: str | None) -> str | None:
        if not value or value.startswith(("http://", "https://", "data:")):
            return value
        path = Path(value)
        return str(path if path.is_absolute() else self.path / path)

    def _save_manifest(self) -> None:
        self.manifest.updated_at = _now()
        self._atomic_write_json(
            self.manifest_path,
            self.manifest.model_dump(mode="json", exclude_none=True),
        )

    @staticmethod
    def _atomic_write_json(path: Path, data: Any) -> None:
        temp_path = path.with_name(f".{path.name}.tmp")
        temp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
