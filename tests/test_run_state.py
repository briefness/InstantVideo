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


def test_storyboard_normalizes_causal_interaction_mode():
    data = storyboard()
    data["shots"][0]["interaction_geometry"] = {
        "interaction_mode": "directed path",
        "source": "visible emitted force",
        "effect_region": "narrow path from actor to target",
        "reaction_scope": "only subjects intersecting the path",
        "unaffected_behavior": "subjects outside the path continue unchanged",
    }

    shot = validate_storyboard(data)["shots"][0]

    assert shot["interaction_geometry"]["interaction_mode"] == "directed_path"
    assert shot["interaction_geometry"]["reaction_scope"] == (
        "only subjects intersecting the path"
    )


def test_storyboard_normalizes_effect_contract_without_schema_failure():
    storyboard = validate_storyboard({
        "title": "effect contract",
        "total_duration": 5,
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "scene_description": "A generic interaction scene.",
            "prompt_en": "A generic source acts on a generic target.",
            "interaction_geometry": {
                "interaction_mode": "beam path",
                "effect_phase": "firing impact",
                "outcome_scope": "some targets",
                "effect_motion": "straight fixed path",
            },
        }],
    })

    geometry = storyboard["shots"][0]["interaction_geometry"]
    assert geometry["interaction_mode"] == "directed_path"
    assert geometry["effect_phase"] == "active"
    assert geometry["outcome_scope"] == "subset"
    assert geometry["effect_motion"] == "static"


def test_storyboard_normalizes_narrative_function_without_losing_state():
    data = storyboard()
    data["story_arc"] = {
        "goal": "restore access",
        "stakes": "the route remains blocked",
        "turning_point": "the first method fails",
        "resolution": "a viable route opens",
    }
    data["shots"][0]["narrative_beat"] = {
        "function": "inciting incident",
        "state_before": "the route appears usable",
        "state_change": "an obstacle is revealed",
        "state_after": "the route is visibly blocked",
    }

    normalized = validate_storyboard(data)

    assert normalized["shots"][0]["narrative_beat"]["function"] == "setup"
    assert normalized["story_arc"]["resolution"] == "a viable route opens"


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


def test_workspace_normalizes_prefixed_relative_artifact_path(
    tmp_path: Path, monkeypatch
):
    workspace = RunWorkspace.create(
        tmp_path / "runs",
        RunOptions(request="A test video"),
    )
    workspace.save_storyboard(storyboard())
    video = workspace.path / "shots" / "shot_001.mp4"
    video.parent.mkdir(exist_ok=True)
    video.write_bytes(b"video")
    monkeypatch.chdir(tmp_path)

    workspace.record_shot(
        shot_id=1,
        status="success",
        local_path=str(video.relative_to(tmp_path)),
        semantic_accepted=True,
    )

    assert workspace.manifest.shots["1"].local_path == "shots/shot_001.mp4"
    assert workspace.accepted_shot_artifacts()[1]["local_path"] == str(video)


def test_workspace_round_trips_prompt_recovery_ledger(tmp_path: Path):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="A test video"))
    workspace.save_storyboard(storyboard())
    attempts = [
        {
            "attempt": 1,
            "profile": "normal",
            "fingerprint": "a" * 64,
            "outcome": "failed",
            "provider_error_locus": "input_text",
            "provider_error_code": "InputTextSensitiveContentDetected",
        },
        {
            "attempt": 2,
            "profile": "policy_safe",
            "fingerprint": "b" * 64,
            "outcome": "succeeded",
        },
    ]

    workspace.record_shot(
        shot_id=1,
        status="running",
        prompt_profile="normal",
        prompt_fingerprint="a" * 64,
        prompt_attempts=[{**attempts[0], "outcome": "pending"}],
        attempts=1,
    )
    workspace.record_shot(
        shot_id=1,
        status="success",
        provider_error_locus=None,
        prompt_profile="policy_safe",
        prompt_fingerprint="b" * 64,
        recovery_actions=["recompile_input_text_policy_safe"],
        prompt_attempts=attempts,
        attempts=2,
    )

    state = RunWorkspace.resume(workspace.path).manifest.shots["1"]
    assert state.prompt_profile == "policy_safe"
    assert state.prompt_fingerprint == "b" * 64
    assert state.recovery_actions == ["recompile_input_text_policy_safe"]
    assert [attempt.outcome for attempt in state.prompt_attempts] == [
        "failed",
        "succeeded",
    ]


def test_workspace_persists_structured_provider_outcome(tmp_path: Path):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="A test video"))
    workspace.save_storyboard(storyboard())

    workspace.record_shot(
        shot_id=1,
        status="failed",
        provider_task_id="ark-task-policy",
        provider_error_type="copyright_policy",
        provider_error_code="OutputVideoSensitiveContentDetected.PolicyViolation",
        provider_error_message="The output may be related to copyright restrictions.",
        attempts=1,
    )

    state = RunWorkspace.resume(workspace.path).manifest.shots["1"]
    assert state.provider_error_type == "copyright_policy"
    assert state.provider_error_code == (
        "OutputVideoSensitiveContentDetected.PolicyViolation"
    )
    assert "copyright restrictions" in state.provider_error_message


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


def test_resume_keeps_planned_and_observed_end_states_separate(tmp_path: Path):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="A test video"))
    planned = storyboard()
    planned["shots"][0]["end_state"] = {
        "prop_state": "door remains closed",
    }
    workspace.save_storyboard(planned)
    workspace.record_shot(
        shot_id=1,
        status="success",
        observed_end_state={"prop_state": "door is visibly open"},
    )

    restored = RunWorkspace.resume(workspace.path).load_storyboard()["shots"][0]

    assert restored["end_state"]["prop_state"] == "door remains closed"
    assert restored["observed_end_state"]["prop_state"] == "door is visibly open"


def test_rejected_take_is_retained_without_replacing_canonical_take(tmp_path: Path):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="A test video"))
    workspace.save_storyboard(storyboard())
    accepted_path = workspace.path / "shots" / "shot_001.mp4"
    rejected_path = workspace.path / "shots" / "shot_001_rejected_01.mp4"
    accepted_path.parent.mkdir(exist_ok=True)
    accepted_path.write_bytes(b"accepted")
    rejected_path.write_bytes(b"rejected")

    workspace.record_shot(
        shot_id=1,
        status="success",
        local_path=str(accepted_path),
        semantic_accepted=True,
        observed_end_state={"action_phase": "door opened"},
    )
    workspace.record_shot(
        shot_id=1,
        status="failed",
        local_path=str(rejected_path),
        semantic_accepted=False,
        observed_end_state={"action_phase": "door closed again"},
        errors=["state reversal"],
    )

    resumed = RunWorkspace.resume(workspace.path)
    state = resumed.manifest.shots["1"]
    artifacts = resumed.accepted_shot_artifacts()

    assert [take.disposition for take in state.take_history] == [
        "accepted",
        "rejected",
    ]
    assert state.canonical_take_id == state.take_history[0].take_id
    assert artifacts[1]["local_path"] == str(accepted_path)
    assert artifacts[1]["observed_end_state"]["action_phase"] == "door opened"


def test_rejected_only_take_never_becomes_resume_input(tmp_path: Path):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="A test video"))
    workspace.save_storyboard(storyboard())
    rejected_path = workspace.path / "shots" / "shot_001_rejected_01.mp4"
    rejected_path.parent.mkdir(exist_ok=True)
    rejected_path.write_bytes(b"rejected")

    workspace.record_shot(
        shot_id=1,
        status="failed",
        local_path=str(rejected_path),
        semantic_accepted=False,
        observed_end_state={"action_phase": "invalid result"},
    )

    resumed = RunWorkspace.resume(workspace.path)

    assert resumed.manifest.shots["1"].canonical_take_id is None
    assert resumed.manifest.shots["1"].take_history[0].disposition == "rejected"
    assert resumed.accepted_shot_artifacts() == {}


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
