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

    misspelled_contract = storyboard()
    misspelled_contract["shots"][0]["require_visible_entities"] = ["robot"]
    with pytest.raises(ValidationError, match="require_visible_entities"):
        validate_storyboard(misspelled_contract)

    for duration in (3, 16):
        invalid_duration = storyboard()
        invalid_duration["shots"][0]["duration"] = duration
        with pytest.raises(ValidationError):
            validate_storyboard(invalid_duration)


def test_storyboard_normalizes_bounded_axis_change_aliases():
    aliases = {
        "none": "hold",
        "no_change": "hold",
        "same_axis": "hold",
        "re-establish": "reestablish",
    }

    for raw_value, expected in aliases.items():
        data = storyboard()
        data["shots"][0]["camera"] = {"axis_change": raw_value}
        normalized = validate_storyboard(data)
        assert normalized["shots"][0]["camera"]["axis_change"] == expected

    invalid = storyboard()
    invalid["shots"][0]["camera"] = {"axis_change": "flip"}
    with pytest.raises(ValidationError):
        validate_storyboard(invalid)


def test_storyboard_normalizes_llm_action_phase_labels_by_meaning_and_order():
    data = storyboard()
    data["shots"][0]["action_beats"] = [
        {
            "phase": "advance",
            "actor": "robot",
            "action": "advances through the corridor",
        },
        {
            "phase": "scan complete",
            "actor": "robot",
            "action": "finishes the scan",
        },
    ]

    normalized = validate_storyboard(data)

    assert [beat["phase"] for beat in normalized["shots"][0]["action_beats"]] == [
        "trigger",
        "aftermath",
    ]


def test_storyboard_normalizes_unknown_action_phase_labels_by_sequence_position():
    data = storyboard()
    data["shots"][0]["action_beats"] = [
        {"phase": "step one", "actor": "robot", "action": "raises scanner"},
        {"phase": "step two", "actor": "robot", "action": "scans the room"},
        {"phase": "step three", "actor": "robot", "action": "lowers scanner"},
    ]

    normalized = validate_storyboard(data)

    assert [beat["phase"] for beat in normalized["shots"][0]["action_beats"]] == [
        "trigger",
        "peak",
        "aftermath",
    ]


def test_storyboard_normalizes_llm_character_classification_aliases():
    data = storyboard()
    data["characters"] = [
        {"name": "robot", "mobility": "tank treads", "reference_mode": "main character"},
        {"name": "zombies", "mobility": "humanoid", "reference_mode": "horde"},
    ]

    normalized = validate_storyboard(data)

    assert normalized["characters"][0]["mobility"] == "tracked"
    assert normalized["characters"][0]["reference_mode"] == "identity"
    assert normalized["characters"][1]["mobility"] == "bipedal"
    assert normalized["characters"][1]["reference_mode"] == "group"


def test_storyboard_normalizes_coverage_contract_aliases():
    data = storyboard()
    data["shots"][0].update({
        "composition_change": "major reframing",
        "coverage_role": "combat two-shot",
        "required_visible_entities": "robot",
        "interaction_geometry": {"occlusion_policy": "no occlusion"},
    })

    normalized = validate_storyboard(data)
    shot = normalized["shots"][0]

    assert shot["composition_change"] == "large"
    assert shot["coverage_role"] == "interaction"
    assert shot["required_visible_entities"] == ["robot"]
    assert shot["interaction_geometry"]["occlusion_policy"] == "none"


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
    assert resumed.manifest.shots["1"].last_frame_url is None


def test_workspace_restores_accepted_shot_artifacts_as_canonical_state(
    tmp_path: Path,
):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="A test video"))
    workspace.save_storyboard(storyboard())
    video = workspace.path / "shots" / "shot_001.mp4"
    tail = workspace.path / "shots" / "shot_001_lastframe.jpg"
    video.parent.mkdir(exist_ok=True)
    video.write_bytes(b"video")
    tail.write_bytes(b"tail")
    workspace.record_shot(
        shot_id=1,
        status="success",
        local_path=str(video),
        last_frame_url=str(tail),
        semantic_accepted=True,
        observed_end_state={
            "location": "street",
            "subject": "robot facing right",
            "action_phase": "burst completed",
        },
    )

    resumed = RunWorkspace.resume(workspace.path)
    artifact = resumed.accepted_shot_artifacts()[1]

    assert artifact["local_path"] == str(video)
    assert artifact["last_frame_url"] == str(tail)
    assert artifact["observed_end_state"]["action_phase"] == "burst completed"


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
