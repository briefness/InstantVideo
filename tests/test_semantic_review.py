import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.semantic_review import (
    SemanticReview,
    SemanticReviewUnavailableError,
    SemanticTakeReviewer,
    _completion_token_budget,
    _review_contract,
)


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = json.dumps({
            "accepted": True,
            "required_entities_visible": {"robot": True, "zombies": True},
            "action_geometry_valid": True,
            "primary_action_completed": True,
            "environment_continuity_valid": True,
            "action_handoff_valid": True,
            "screen_direction_valid": True,
            "prop_continuity_valid": True,
            "observed_end_state": {
                "location": "ruined street",
                "subject": "robot facing zombies",
                "action_phase": "burst completed",
                "camera": "medium two-subject shot",
            },
            "failure_reason": "",
        })
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_semantic_output_budget_scales_with_evidence_schema():
    assert _completion_token_budget(False) == 4096
    assert _completion_token_budget(True) == 8192


class SequencedCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


def _completion(content, *, finish_reason="stop", refusal=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content, refusal=refusal),
        )],
        usage=SimpleNamespace(completion_tokens=1200),
    )


def _accepted_review_json():
    return json.dumps({
        "accepted": True,
        "required_entities_visible": [True],
        "action_geometry_valid": True,
        "primary_action_completed": True,
        "observed_end_state": {
            "location": "ruined street",
            "subject": "combat robot",
            "action_phase": "target locked",
        },
        "failure_reason": "",
    })


def test_causal_component_failure_rejects_take_with_specific_reason():
    review = SemanticReview.from_dict({
        "accepted": True,
        "required_entities_visible": {"source": True, "target": True},
        "action_geometry_valid": True,
        "effect_path_valid": False,
        "reaction_causality_valid": False,
        "primary_action_completed": True,
        "observed_end_state": {},
        "failure_reason": None,
    }, require_causality=True)

    assert review.accepted is False
    assert "作用路径或区域" in review.failure_reason
    assert "反应范围" in review.failure_reason


def test_narrative_state_failure_rejects_take_with_specific_reason():
    review = SemanticReview.from_dict({
        "accepted": True,
        "required_entities_visible": {"subject": True},
        "action_geometry_valid": True,
        "primary_action_completed": True,
        "narrative_state_change_valid": False,
        "observed_end_state": {},
        "failure_reason": None,
    }, require_narrative=True)

    assert review.accepted is False
    assert "故事状态" in review.failure_reason


def test_blocking_and_composition_failures_reject_take_with_specific_reason():
    review = SemanticReview.from_dict({
        "accepted": True,
        "required_entities_visible": {"subject": True, "target": True},
        "action_geometry_valid": True,
        "primary_action_completed": True,
        "blocking_valid": False,
        "composition_change_valid": False,
        "observed_end_state": {},
        "failure_reason": None,
    }, require_blocking=True, require_composition=True)

    assert review.accepted is False
    assert "角色调度" in review.failure_reason
    assert "构图变化" in review.failure_reason


def test_semantic_review_samples_contract_and_caches_by_video_hash(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "shot.mp4"
    video.write_bytes(b"accepted-video-bytes")
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def fake_extract(_video, output, timestamp=None):
        Path(output).write_bytes(f"frame-{timestamp}".encode())
        return output

    monkeypatch.setattr("pipeline.semantic_review.extract_frame", fake_extract)
    monkeypatch.setattr(
        "pipeline.semantic_review.get_video_duration", lambda _path: 5.0
    )
    reviewer = SemanticTakeReviewer(tmp_path, client=client, model="vision-test")
    shot = {
        "shot_id": 2,
        "primary_action": "robot fires one burst at zombies",
        "coverage_role": "interaction",
        "required_visible_entities": ["robot", "zombies"],
        "interaction_geometry": {
            "actor": "robot",
            "target": "zombies",
            "must_share_frame": True,
            "line_of_action_visible": True,
        },
    }

    first = reviewer.review(str(video), shot)
    second = reviewer.review(str(video), shot)

    assert first.accepted is True
    assert first.observed_end_state["subject"] == "robot facing zombies"
    assert second == first
    assert len(completions.calls) == 1
    message_content = completions.calls[0]["messages"][1]["content"]
    assert len([item for item in message_content if item["type"] == "image_url"]) == 5
    assert list((tmp_path / "semantic_reviews").glob("*.json"))


def test_local_similarity_rejects_large_cut_even_when_model_accepts(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "shot.mp4"
    previous = tmp_path / "previous.jpg"
    video.write_bytes(b"video")
    previous.write_bytes(b"previous-frame")
    response = {
        "accepted": True,
        "required_entities_visible": [True],
        "action_geometry_valid": True,
        "composition_change_valid": True,
        "primary_action_completed": True,
        "boundary_continuity_valid": True,
        "identity_continuity_valid": True,
        "environment_continuity_valid": True,
        "action_handoff_valid": True,
        "screen_direction_valid": True,
        "prop_continuity_valid": True,
        "observed_end_state": {
            "location": "ruined street",
            "subject": "combat robot",
            "action_phase": "new angle established",
        },
        "failure_reason": "",
    }
    completions = SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))]
    ))
    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr(
        "pipeline.semantic_review.frame_structure_similarity",
        lambda _first, _second: 0.96,
    )
    monkeypatch.setattr("pipeline.semantic_review.get_video_duration", lambda _path: 5.0)

    review = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="vision-test",
    ).review(
        str(video),
        {
            "shot_id": 2,
            "required_visible_entities": ["combat_robot"],
            "continuity_from_previous": "intentional_cut",
            "composition_change": "large",
        },
        previous_frame_path=str(previous),
    )

    assert review.accepted is False
    assert review.composition_change_valid is False
    assert "构图变化" in review.failure_reason


def test_state_handoff_evidence_rejects_progress_reset_despite_aggregate_acceptance(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "shot.mp4"
    previous = tmp_path / "previous.jpg"
    video.write_bytes(b"video")
    previous.write_bytes(b"previous-frame")
    response = {
        "accepted": True,
        "required_entities_visible": [True],
        "action_geometry_valid": True,
        "primary_action_completed": True,
        "boundary_continuity_valid": True,
        "identity_continuity_valid": True,
        "environment_continuity_valid": True,
        "action_handoff_valid": True,
        "screen_direction_valid": True,
        "prop_continuity_valid": True,
        "boundary_state_evidence": {
            "prior_state_preserved": False,
            "state_progress_not_reversed": False,
            "open_motion_handoff_valid": True,
            "persistent_entities_preserved": True,
            "scene_identity_preserved": True,
        },
        "observed_end_state": {
            "location": "same table",
            "subject": "physical sequence",
            "action_phase": "continuing",
        },
        "failure_reason": "",
    }
    completions = SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))]
    ))
    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr("pipeline.semantic_review.get_video_duration", lambda _path: 5.0)

    review = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="vision-test",
    ).review(
        str(video),
        {
            "shot_id": 2,
            "required_visible_entities": ["physical_sequence"],
            "continuity_from_previous": "intentional_cut",
            "composition_change": "small",
        },
        previous_frame_path=str(previous),
        boundary_context={
            "same_scene": True,
            "previous_observed_end_state": {
                "action_phase": "advanced state with prior results preserved"
            },
        },
    )

    assert review.accepted is False
    assert review.boundary_continuity_valid is False
    assert "state_progress_not_reversed" in review.failure_reason


def test_first_frame_handoff_checks_reframing_at_midpoint(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "shot.mp4"
    previous = tmp_path / "previous.jpg"
    video.write_bytes(b"video")
    previous.write_bytes(b"previous-frame")
    observed_comparison = {}
    response = {
        "accepted": True,
        "required_entities_visible": [True],
        "action_geometry_valid": True,
        "primary_action_completed": True,
        "composition_change_valid": True,
        "boundary_continuity_valid": True,
        "identity_continuity_valid": True,
        "environment_continuity_valid": True,
        "action_handoff_valid": True,
        "screen_direction_valid": True,
        "prop_continuity_valid": True,
        "boundary_state_evidence": {
            "prior_state_preserved": True,
            "state_progress_not_reversed": True,
            "open_motion_handoff_valid": True,
            "persistent_entities_preserved": True,
            "scene_identity_preserved": True,
        },
        "observed_end_state": {
            "location": "same set",
            "subject": "subject",
            "action_phase": "completed",
        },
        "failure_reason": "",
    }
    completions = SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))]
    ))

    def fake_extract(_video, output, timestamp=None):
        Path(output).write_bytes(f"frame-{timestamp}".encode())

    def fake_similarity(_previous, current):
        observed_comparison["frame"] = Path(current).read_bytes()
        return 0.5

    monkeypatch.setattr("pipeline.semantic_review.extract_frame", fake_extract)
    monkeypatch.setattr(
        "pipeline.semantic_review.frame_structure_similarity", fake_similarity
    )
    monkeypatch.setattr("pipeline.semantic_review.get_video_duration", lambda _path: 5.0)

    review = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="vision-test",
    ).review(
        str(video),
        {
            "shot_id": 2,
            "required_visible_entities": ["subject"],
            "continuity_from_previous": "intentional_cut",
            "composition_change": "medium",
        },
        previous_frame_path=str(previous),
        boundary_context={
            "same_scene": True,
            "state_reference_role": "first_frame",
        },
    )

    assert review.accepted is True
    assert observed_comparison["frame"] == b"frame-2.5"


def test_causal_review_uses_dense_temporal_sampling(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "causal-shot.mp4"
    video.write_bytes(b"causal-video")
    response = {
        "accepted": True,
        "required_entities_visible": [True, True],
        "action_geometry_valid": True,
        "effect_path_valid": True,
        "reaction_causality_valid": True,
        "primary_action_completed": True,
        "observed_end_state": {
            "location": "abstract scene",
            "subject": "source and target",
            "action_phase": "effect completed",
        },
        "failure_reason": "",
    }
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps(response)
            ))]
        )

    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr(
        "pipeline.semantic_review.get_video_duration", lambda _path: 6.0
    )
    shot = {
        "shot_id": 1,
        "required_visible_entities": ["source", "target"],
        "interaction_geometry": {
            "interaction_mode": "area_effect",
            "source": "expanding pulse",
            "effect_region": "marked circular area",
            "reaction_scope": "entities inside the circle",
            "unaffected_behavior": "entities outside remain unchanged",
        },
    }

    review = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )),
        model="vision-test",
    ).review(str(video), shot)

    assert review.accepted is True
    content = calls[0]["messages"][1]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 9
    system_prompt = calls[0]["messages"][0]["content"]
    assert "does not require a straight line" in system_prompt
    assert "source may be offscreen" in system_prompt


def test_setup_review_excludes_actor_only_preparation_from_physical_effect(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "setup-shot.mp4"
    video.write_bytes(b"setup-video")
    evidence = [
        {
            "physical_effect_visible": False,
            "reaction_visible": False,
            "effect_intersects_reaction": False,
            "out_of_scope_reaction_visible": False,
            "contracted_outcome_visible": False,
            "outcome_causally_connected": False,
        }
        for _ in range(9)
    ]
    response = {
        "accepted": True,
        "required_entities_visible": [True, True],
        "action_geometry_valid": True,
        "effect_path_valid": True,
        "reaction_causality_valid": True,
        "causal_sample_evidence": evidence,
        "primary_action_completed": True,
        "observed_end_state": {
            "location": "ruined street",
            "subject": "robot aiming at approaching zombies",
            "action_phase": "combat ready",
        },
        "failure_reason": "",
    }
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps(response)
            ))]
        )

    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr(
        "pipeline.semantic_review.get_video_duration", lambda _path: 5.0
    )
    shot = {
        "shot_id": 1,
        "required_visible_entities": ["combat_robot", "zombie_group"],
        "interaction_geometry": {
            "interaction_mode": "none",
            "effect_phase": "setup",
            "outcome_scope": "none",
            "effect_motion": "none",
        },
    }

    review = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )),
        model="vision-test",
    ).review(str(video), shot)

    assert review.accepted is True
    system_prompt = calls[0]["messages"][0]["content"]
    assert (
        "physical_effect_visible is false for actor-only aiming, charging, "
        "sensor activation, weapon power-up, or source-local glow"
    ) in system_prompt


def test_structured_causal_evidence_overrides_aggregate_model_acceptance(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "causal-shot.mp4"
    video.write_bytes(b"causal-video")
    evidence = [
        {
            "physical_effect_visible": True,
            "reaction_visible": True,
            "effect_intersects_reaction": True,
            "out_of_scope_reaction_visible": False,
            "contracted_outcome_visible": True,
            "outcome_causally_connected": True,
        }
        for _ in range(9)
    ]
    evidence[5] = {
        "physical_effect_visible": True,
        "reaction_visible": True,
        "effect_intersects_reaction": False,
        "out_of_scope_reaction_visible": True,
        "contracted_outcome_visible": True,
        "outcome_causally_connected": True,
    }
    response = {
        "accepted": True,
        "required_entities_visible": [True, True],
        "action_geometry_valid": True,
        "effect_path_valid": True,
        "reaction_causality_valid": True,
        "causal_sample_evidence": evidence,
        "primary_action_completed": True,
        "observed_end_state": {
            "location": "abstract scene",
            "subject": "source and target",
            "action_phase": "effect completed",
        },
        "failure_reason": "None",
    }
    completions = SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))]
    ))
    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr("pipeline.semantic_review.get_video_duration", lambda _path: 6.0)

    review = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="vision-test",
    ).review(str(video), {
        "shot_id": 1,
        "required_visible_entities": ["source", "target"],
        "interaction_geometry": {
            "interaction_mode": "directed_path",
            "effect_phase": "active",
            "outcome_scope": "subset",
            "effect_motion": "static",
            "source": "visible source",
            "effect_region": "straight path",
            "reaction_scope": "intersected targets",
            "unaffected_behavior": "outside targets continue",
        },
    })

    assert review.accepted is False
    assert review.effect_path_valid is False
    assert review.reaction_causality_valid is False
    assert "范围外目标发生反应" in review.failure_reason


def test_complete_causal_evidence_overrides_false_aggregate_verdict(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "precontact-block.mp4"
    video.write_bytes(b"video")
    base = {
        "physical_effect_visible": False,
        "reaction_visible": False,
        "effect_intersects_reaction": False,
        "out_of_scope_reaction_visible": False,
        "contracted_outcome_visible": False,
        "outcome_causally_connected": False,
    }
    evidence = [base, base, {**base, "physical_effect_visible": True}]
    evidence.append({
        **base,
        "physical_effect_visible": True,
        "reaction_visible": True,
    })
    evidence.extend([{
        **base,
        "physical_effect_visible": True,
        "reaction_visible": True,
        "effect_intersects_reaction": True,
        "contracted_outcome_visible": True,
        "outcome_causally_connected": True,
    } for _ in range(5)])
    response = {
        "accepted": False,
        "required_entities_visible": [True, True],
        "action_geometry_valid": True,
        "effect_path_valid": False,
        "reaction_causality_valid": False,
        "causal_sample_evidence": evidence,
        "primary_action_completed": True,
        "observed_end_state": {
            "location": "generic arena",
            "subject": "two subjects locked at the contact point",
            "action_phase": "contact established",
        },
        "failure_reason": "sample 4 reacts before intersection",
    }
    completions = SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))]
    ))
    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr("pipeline.semantic_review.get_video_duration", lambda _path: 6.0)

    review = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="vision-test",
    ).review(str(video), {
        "shot_id": 1,
        "required_visible_entities": ["source", "target"],
        "interaction_geometry": {
            "interaction_mode": "direct_contact",
            "effect_phase": "active",
            "outcome_scope": "single",
            "effect_motion": "static",
            "source": "visible source",
            "effect_region": "contact point",
            "reaction_scope": "contacted target",
            "unaffected_behavior": "other subjects unchanged",
        },
    })

    assert review.accepted is True
    assert review.effect_path_valid is True
    assert review.reaction_causality_valid is True
    assert review.failure_reason == ""


def test_causal_evidence_issue_owns_failure_reason_over_model_free_text():
    from pipeline.causality import ACTION_EVIDENCE_FIELDS, compile_action_contract
    from pipeline.semantic_review import _apply_causal_evidence_verdict

    parsed = {
        "accepted": True,
        "effect_path_valid": True,
        "reaction_causality_valid": True,
        "failure_reason": "No failures",
    }
    contract = compile_action_contract({
        "interaction_geometry": {
            "effect_phase": "active",
            "interaction_mode": "directed_path",
            "outcome_scope": "single",
            "effect_motion": "static",
        },
    })

    _apply_causal_evidence_verdict(
        parsed,
        contract,
        [{field: False for field in ACTION_EVIDENCE_FIELDS}],
    )

    assert parsed["effect_path_valid"] is False
    assert parsed["reaction_causality_valid"] is False
    assert parsed["failure_reason"] == "生效阶段未看到物理作用；生效阶段未看到物理作用到达目标；生效阶段未看到作用区域内目标的同步反应；active 阶段未看到约定的动作端点；active 阶段未看到约定结果（叙事结果）及其完整作用范围；active 阶段的约定结果与物理作用之间缺少可见因果过渡"


def test_empty_semantic_response_retries_review_once_on_same_video(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "shot.mp4"
    video.write_bytes(b"accepted-video-bytes")
    completions = SequencedCompletions([
        _completion("", finish_reason="length"),
        _completion(_accepted_review_json()),
    ])
    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr(
        "pipeline.semantic_review.get_video_duration", lambda _path: 6.0
    )

    review = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="vision-test",
    ).review(str(video), {
        "shot_id": 1,
        "required_visible_entities": ["combat_robot"],
    })

    assert review.accepted is True
    assert len(completions.calls) == 2
    response_format = completions.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert all(call["max_completion_tokens"] >= 4096 for call in completions.calls)


def test_invalid_semantic_response_retry_is_bounded_and_diagnostic(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "shot.mp4"
    video.write_bytes(b"accepted-video-bytes")
    completions = SequencedCompletions([
        _completion("", finish_reason="length"),
        _completion("", finish_reason="stop"),
    ])
    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr(
        "pipeline.semantic_review.get_video_duration", lambda _path: 6.0
    )

    reviewer = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="vision-test",
    )
    with pytest.raises(
        SemanticReviewUnavailableError,
        match="2 次.*finish_reason=stop.*content_length=0",
    ):
        reviewer.review(str(video), {
            "shot_id": 1,
            "required_visible_entities": ["combat_robot"],
        })

    assert len(completions.calls) == 2


def test_causal_evidence_schema_mismatch_retries_same_review_once(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "setup-shot.mp4"
    video.write_bytes(b"accepted-video-bytes")
    sample = {
        "preparation_state_visible": True,
        "non_physical_cue_visible": True,
        "physical_effect_visible": False,
        "effect_reaches_target": False,
        "target_reaction_visible": False,
        "out_of_scope_reaction_visible": False,
        "phase_endpoint_visible": True,
        "narrative_outcome_visible": False,
        "outcome_causally_connected": False,
    }
    valid_evidence = [dict(sample) for _ in range(9)]

    def response(evidence):
        return _completion(json.dumps({
            "accepted": True,
            "required_entities_visible": [True, True],
            "action_geometry_valid": True,
            "causal_sample_evidence": evidence,
            "primary_action_completed": True,
            "observed_end_state": {
                "location": "ruined street",
                "subject": "robot aiming at approaching targets",
                "action_phase": "target lock complete",
            },
            "failure_reason": "",
        }))

    completions = SequencedCompletions([
        response(valid_evidence[:-1]),
        response(valid_evidence),
    ])
    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr(
        "pipeline.semantic_review.get_video_duration", lambda _path: 5.0
    )

    review = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="vision-test",
    ).review(str(video), {
        "shot_id": 1,
        "required_visible_entities": ["combat_robot", "target_group"],
        "interaction_geometry": {
            "interaction_mode": "none",
            "effect_phase": "setup",
            "outcome_scope": "none",
            "effect_motion": "none",
        },
    })

    assert review.accepted is True
    assert len(completions.calls) == 2
    response_format = completions.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    response_schema = response_format["json_schema"]["schema"]
    evidence_schema = response_schema["properties"]["causal_sample_evidence"]
    assert evidence_schema["minItems"] == 9
    assert evidence_schema["maxItems"] == 9
    assert evidence_schema["items"]["type"] == "object"
    assert evidence_schema["items"]["additionalProperties"] is False


def test_cached_review_recomputes_verdict_from_causal_evidence(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "setup-shot.mp4"
    video.write_bytes(b"setup-video-with-targeting-cue")
    sample = {
        "preparation_state_visible": True,
        "non_physical_cue_visible": True,
        "physical_effect_visible": False,
        "effect_reaches_target": True,
        "target_reaction_visible": False,
        "out_of_scope_reaction_visible": False,
        "phase_endpoint_visible": True,
        "narrative_outcome_visible": False,
        "outcome_causally_connected": True,
    }
    response = _completion(json.dumps({
        "accepted": False,
        "required_entities_visible": [True, True],
        "action_geometry_valid": True,
        "causal_sample_evidence": [dict(sample) for _ in range(9)],
        "primary_action_completed": True,
        "observed_end_state": {
            "location": "ruined street",
            "subject": "robot aiming at targets",
            "action_phase": "target lock complete",
        },
        "failure_reason": "准备阶段提前作用于目标",
    }))
    completions = SequencedCompletions([response])
    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr(
        "pipeline.semantic_review.get_video_duration", lambda _path: 5.0
    )
    reviewer = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="vision-test",
    )
    shot = {
        "shot_id": 1,
        "required_visible_entities": ["robot", "target_group"],
        "interaction_geometry": {
            "interaction_mode": "none",
            "effect_phase": "setup",
            "outcome_scope": "none",
            "effect_motion": "none",
        },
    }

    first = reviewer.review(str(video), shot)
    cache_path = next((tmp_path / "semantic_reviews").glob("*.json"))
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    cached["review"].update({
        "accepted": False,
        "effect_path_valid": False,
        "reaction_causality_valid": False,
        "failure_reason": "准备阶段提前作用于目标",
    })
    cache_path.write_text(json.dumps(cached), encoding="utf-8")
    second = reviewer.review(str(video), shot)

    assert first.accepted is True
    assert second.accepted is True
    assert second.failure_reason == ""
    assert len(completions.calls) == 1


def test_semantic_refusal_stops_without_retry(tmp_path: Path, monkeypatch):
    video = tmp_path / "shot.mp4"
    video.write_bytes(b"accepted-video-bytes")
    completions = SequencedCompletions([
        _completion("", finish_reason="content_filter", refusal="blocked"),
    ])
    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr(
        "pipeline.semantic_review.get_video_duration", lambda _path: 6.0
    )

    reviewer = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="vision-test",
    )
    with pytest.raises(SemanticReviewUnavailableError, match="服务端拒绝，未重试"):
        reviewer.review(str(video), {
            "shot_id": 1,
            "required_visible_entities": ["combat_robot"],
        })

    assert len(completions.calls) == 1


def test_semantic_review_rejects_string_booleans():
    with pytest.raises(ValueError, match="accepted must be a boolean"):
        SemanticReview.from_dict({
            "accepted": "false",
            "required_entities_visible": {"robot": "false"},
            "action_geometry_valid": "false",
            "primary_action_completed": "false",
            "observed_end_state": {"subject": "robot"},
        })


def test_null_failure_reason_is_replaced_with_objective_failure():
    review = SemanticReview.from_dict({
        "accepted": False,
        "required_entities_visible": {"combat_robot": False, "zombies": True},
        "action_geometry_valid": True,
        "primary_action_completed": True,
        "observed_end_state": {},
        "failure_reason": None,
    })

    assert review.failure_reason == "必需主体不可见: combat_robot"


def test_ordered_visibility_array_maps_to_contract_ids(tmp_path: Path, monkeypatch):
    video = tmp_path / "shot.mp4"
    video.write_bytes(b"video")
    response = {
        "accepted": True,
        "required_entities_visible": [True, True],
        "action_geometry_valid": True,
        "primary_action_completed": True,
        "observed_end_state": {
            "location": "ruined street",
            "subject": "combat robot facing zombies",
            "action_phase": "first burst completed",
        },
        "failure_reason": None,
    }
    completions = SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))]
    ))
    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr("pipeline.semantic_review.get_video_duration", lambda _path: 5.0)

    review = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="vision-test",
    ).review(str(video), {
        "shot_id": 1,
        "required_visible_entities": ["combat_robot", "zombies"],
    })

    assert review.required_entities_visible == {
        "combat_robot": True,
        "zombies": True,
    }
    assert review.accepted is True


def test_misaligned_visibility_dict_pauses_instead_of_rejecting_take(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "shot.mp4"
    video.write_bytes(b"video")
    response = {
        "accepted": False,
        "required_entities_visible": {"robot": True, "lead zombie": True},
        "action_geometry_valid": True,
        "primary_action_completed": True,
        "observed_end_state": {},
        "failure_reason": None,
    }
    completions = SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))]
    ))
    monkeypatch.setattr(
        "pipeline.semantic_review.extract_frame",
        lambda _video, output, timestamp=None: Path(output).write_bytes(b"frame"),
    )
    monkeypatch.setattr("pipeline.semantic_review.get_video_duration", lambda _path: 5.0)

    reviewer = SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="vision-test",
    )
    with pytest.raises(SemanticReviewUnavailableError, match="实体可见性"):
        reviewer.review(str(video), {
            "shot_id": 1,
            "required_visible_entities": ["combat_robot", "zombies"],
        })


def test_identity_anchor_purity_is_part_of_review_contract():
    contract = _review_contract({
        "shot_id": 1,
        "characters": ["hero"],
        "extract_character_ref": True,
    })

    assert contract["characters"] == ["hero"]
    assert contract["extract_character_ref"] is True


def test_boundary_handoff_fields_are_part_of_review_contract():
    contract = _review_contract({
        "shot_id": 2,
        "scene_id": "street",
        "continuity_from_previous": "intentional_cut",
        "composition_change": "large",
        "start_state": {"open_motion": "zombies continue advancing"},
    })

    assert contract["scene_id"] == "street"
    assert contract["continuity_from_previous"] == "intentional_cut"
    assert contract["composition_change"] == "large"
    assert contract["start_state"]["open_motion"] == "zombies continue advancing"


def test_narrative_beat_is_part_of_review_contract():
    contract = _review_contract({
        "shot_id": 2,
        "narrative_beat": {
            "function": "progress",
            "state_before": "the question is unresolved",
            "state_change": "new evidence becomes visible",
            "state_after": "the answer is understood",
        },
    })

    assert contract["narrative_beat"]["state_change"] == (
        "new evidence becomes visible"
    )
    assert contract["narrative_carriers"][:2] == [
        {"kind": "visible_change", "value": "new evidence becomes visible"},
        {"kind": "readable_endpoint", "value": "the answer is understood"},
    ]


def test_setup_review_contract_uses_action_contract_phase_projection():
    contract = _review_contract({
        "shot_id": 1,
        "primary_action": "operator arms the inactive device",
        "interaction_geometry": {
            "actor": "operator",
            "target": "marker",
            "effect_phase": "setup",
        },
        "action_beats": [{
            "phase": "trigger",
            "actor": "operator",
            "action": "arms the inactive device",
            "target": "marker",
            "visible_result": "RAW_TARGET_RESULT",
        }],
        "narrative_beat": {
            "function": "setup",
            "state_before": "before",
            "state_change": "RAW_NARRATIVE_CHANGE",
            "state_after": "RAW_NARRATIVE_OUTCOME",
        },
        "end_state": {
            "subject": "operator",
            "action_phase": "ready",
            "open_motion": "RAW_TARGET_MOTION",
        },
    })

    payload = json.dumps(contract, sort_keys=True)

    assert contract["primary_action"] == (
        "operator visibly prepares toward marker without a Physical Effect"
    )
    assert contract["action_beats"] == []
    for leaked in (
        "RAW_TARGET_RESULT",
        "RAW_NARRATIVE_CHANGE",
        "RAW_NARRATIVE_OUTCOME",
        "RAW_TARGET_MOTION",
    ):
        assert leaked not in payload


def test_production_slot_is_part_of_review_contract():
    slot = {
        "shot_id": 2,
        "duration": 6,
        "allowed_effect_phases": ["active"],
        "requires_visible_result": True,
    }

    contract = _review_contract({"shot_id": 2, "production_slot": slot})

    assert contract["production_slot"] == slot


def test_boundary_component_failure_rejects_take_with_specific_reason():
    review = SemanticReview.from_dict({
        "accepted": True,
        "required_entities_visible": {"robot": True},
        "action_geometry_valid": True,
        "primary_action_completed": True,
        "boundary_continuity_valid": True,
        "identity_continuity_valid": True,
        "environment_continuity_valid": False,
        "action_handoff_valid": False,
        "screen_direction_valid": True,
        "prop_continuity_valid": False,
        "observed_end_state": {},
        "failure_reason": None,
    })

    assert review.accepted is False
    assert "环境" in review.failure_reason
    assert "动作交接" in review.failure_reason
    assert "道具或武器" in review.failure_reason


def test_identity_crop_boxes_are_normalized_by_character_id():
    review = SemanticReview.from_dict({
        "accepted": True,
        "required_entities_visible": {"combat_robot": True},
        "action_geometry_valid": True,
        "primary_action_completed": True,
        "observed_end_state": {},
        "identity_crop_boxes": {"combat_robot": [0.05, 0.1, 0.55, 0.95]},
    })

    assert review.identity_crop_boxes == {
        "combat_robot": (0.05, 0.1, 0.55, 0.95)
    }


def test_accepted_identity_crop_boxes_require_exact_accepted_cache(tmp_path: Path):
    video = tmp_path / "accepted.mp4"
    video.write_bytes(b"accepted video")
    reviewer = SemanticTakeReviewer(tmp_path, client=object(), model="vision-test")
    video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    cache_path = reviewer.cache_dir / f"{video_hash}.json"
    cache_path.write_text(json.dumps({
        "video_hash": video_hash,
        "review": {
            "accepted": True,
            "identity_crop_boxes": {
                "combat_cleaner_robot": [0.1, 0.2, 0.8, 0.9],
            },
        },
    }), encoding="utf-8")

    assert reviewer.accepted_identity_crop_boxes(video) == {
        "combat_cleaner_robot": (0.1, 0.2, 0.8, 0.9),
    }

    cache_path.write_text(json.dumps({
        "video_hash": video_hash,
        "review": {"accepted": False, "identity_crop_boxes": {
            "combat_cleaner_robot": [0.1, 0.2, 0.8, 0.9],
        }},
    }), encoding="utf-8")
    assert reviewer.accepted_identity_crop_boxes(video) == {}

    cache_path.write_text(json.dumps({
        "video_hash": "mismatched-hash",
        "review": {"accepted": True, "identity_crop_boxes": {
            "combat_cleaner_robot": [0.1, 0.2, 0.8, 0.9],
        }},
    }), encoding="utf-8")
    assert reviewer.accepted_identity_crop_boxes(video) == {}


def test_semantic_cache_is_scoped_to_evaluator_and_reference_context(
    tmp_path: Path, monkeypatch
):
    video = tmp_path / "shot.mp4"
    previous = tmp_path / "previous.jpg"
    video.write_bytes(b"accepted-video-bytes")
    previous.write_bytes(b"previous-frame")
    first_completions = FakeCompletions()
    second_completions = FakeCompletions()

    def fake_extract(_video, output, timestamp=None):
        Path(output).write_bytes(f"frame-{timestamp}".encode())
        return output

    monkeypatch.setattr("pipeline.semantic_review.extract_frame", fake_extract)
    monkeypatch.setattr(
        "pipeline.semantic_review.get_video_duration", lambda _path: 5.0
    )
    shot = {
        "shot_id": 2,
        "primary_action": "robot fires",
        "required_visible_entities": ["robot", "zombies"],
    }
    SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=first_completions)),
        model="vision-a",
    ).review(str(video), shot, previous_frame_path=str(previous))
    SemanticTakeReviewer(
        tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=second_completions)),
        model="vision-b",
    ).review(str(video), shot, previous_frame_path=str(previous))

    assert len(first_completions.calls) == 1
    assert len(second_completions.calls) == 1
    content = first_completions.calls[0]["messages"][1]["content"]
    assert any(
        item.get("type") == "text" and "previous-shot tail" in item.get("text", "")
        for item in content
    )
