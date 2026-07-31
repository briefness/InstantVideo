"""Validated storyboard and durable run state tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.models import RunOptions, RunStatus, validate_storyboard
from pipeline.run_state import RunWorkspace


def storyboard(shot_id: int = 1) -> dict:
    return {
        "title": "Test",
        "aspect_ratio": "16:9",
        "resolution": "480p",
        "shots": [
            {
                "shot_id": shot_id,
                "duration": 5,
                "scene_description": "A test scene",
                "prompt_en": "A cinematic test scene with stable motion",
            }
        ],
    }


def test_storyboard_rejects_invalid_generation_inputs():
    missing_prompt = storyboard()
    del missing_prompt["shots"][0]["prompt_en"]
    with pytest.raises(ValidationError):
        validate_storyboard(missing_prompt)

    duplicate_ids = storyboard()
    duplicate_ids["shots"].append(dict(duplicate_ids["shots"][0]))
    with pytest.raises(ValidationError, match="shot_id must be unique"):
        validate_storyboard(duplicate_ids)

    for duration in (3, 16):
        invalid_duration = storyboard()
        invalid_duration["shots"][0]["duration"] = duration
        with pytest.raises(ValidationError):
            validate_storyboard(invalid_duration)


def test_run_options_reject_unsupported_mini_and_platform_settings():
    with pytest.raises(ValidationError, match="unsupported Seedance Mini resolution"):
        RunOptions(request="A test video", resolution="1080p")

    with pytest.raises(ValidationError, match="unsupported export platforms"):
        RunOptions(request="A test video", platforms=["xiaohongshu"])


def test_workspace_persists_resumable_provider_task_atomically(tmp_path: Path):
    workspace = RunWorkspace.create(
        tmp_path,
        RunOptions(request="A test video"),
    )
    workspace.save_storyboard(storyboard())
    local_video = workspace.path / "shots" / "shot_001.mp4"
    workspace.record_shot(
        shot_id=1,
        status="running",
        provider_task_id="ark-task-1",
        local_path=str(local_video),
        attempts=1,
    )

    resumed = RunWorkspace.resume(workspace.path)
    assert resumed.resumable_provider_tasks() == {1: "ark-task-1"}
    assert resumed.manifest.shots["1"].local_path == "shots/shot_001.mp4"
    assert not (workspace.path / ".run_manifest.json.tmp").exists()

    manifest = json.loads((workspace.path / "run_manifest.json").read_text())
    assert manifest["status"] == "running"
    assert manifest["shots"]["1"]["provider_task_id"] == "ark-task-1"

    resumed.record_shot(shot_id=1, status="running")
    assert resumed.manifest.shots["1"].provider_task_id is None


def test_completed_workspace_reuses_existing_final_video(tmp_path: Path):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="A test video"))
    workspace.save_storyboard(storyboard())
    final_path = workspace.path / "final.mp4"
    final_path.write_bytes(b"video")
    workspace.mark_succeeded(str(final_path))

    resumed = RunWorkspace.resume(workspace.path)
    assert resumed.manifest.status == RunStatus.succeeded
    assert resumed.completed_output() == str(final_path)


def test_corrupt_manifest_fails_before_resume(tmp_path: Path):
    workspace = tmp_path / "broken"
    workspace.mkdir()
    (workspace / "run_manifest.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid run manifest"):
        RunWorkspace.resume(workspace)
