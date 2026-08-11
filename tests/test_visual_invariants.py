"""Shared Positive Locks projection tests."""

import config
from pipeline.provider_prompt import (
    compile_normal_provider_prompt,
    compile_policy_safe_prompt,
)
from pipeline.readiness import provider_submission_issues
from pipeline.semantic_review import SemanticReview, _review_contract
from pipeline.visual_invariants import (
    VISUAL_INVARIANTS_VERSION,
    compile_visual_invariants,
    visual_invariants_fingerprint,
)


def _locked_shot() -> dict:
    return {
        "shot_id": 1,
        "scene_id": "assembly_floor",
        "lighting": "cool overhead work lights",
        "required_visible_entities": ["service_unit", "workpiece"],
        "key_props": ["clamp"],
        "continuity_props": ["marked_toolbox"],
        "camera": {
            "screen_positions": {
                "service_unit": "left foreground beside RAW_POSITION_PROSE",
                "workpiece": "right midground",
            },
        },
        "prompt_en": "RAW_PROMPT_PROSE",
        "primary_action": "RAW_PRIMARY_ACTION",
        "narrative_beat": {"state_change": "RAW_NARRATIVE_CHANGE"},
        "interaction_geometry": {
            "effect_phase": "none",
            "interaction_mode": "none",
            "outcome_scope": "none",
            "effect_motion": "none",
        },
    }


def test_visual_invariants_compile_only_bounded_structured_fields():
    compiled = compile_visual_invariants(_locked_shot())
    provider_text = "; ".join(compiled.prompt_parts)

    assert compiled.review_projection == {
        "version": VISUAL_INVARIANTS_VERSION,
        "setting_identity": "assembly floor",
        "lighting": "cool overhead work lights",
        "required_entity_identity_and_relative_scale": [
            "service unit", "workpiece",
        ],
        "key_prop_identity_and_form": ["clamp"],
        "persistent_props": ["marked toolbox"],
        "initial_screen_positions": {
            "service unit": "left foreground",
            "workpiece": "right midground",
        },
    }
    for forbidden in (
        "RAW_PROMPT_PROSE",
        "RAW_PRIMARY_ACTION",
        "RAW_NARRATIVE_CHANGE",
        "RAW_POSITION_PROSE",
    ):
        assert forbidden not in provider_text


def test_legacy_shot_does_not_acquire_new_visual_invariant_contract():
    shot = _locked_shot()
    shot["interaction_geometry"] = {}

    assert compile_visual_invariants(shot).required is False
    assert visual_invariants_fingerprint(shot) is None


def test_normal_and_policy_safe_profiles_share_the_same_positive_locks():
    shot = _locked_shot()
    expected = compile_visual_invariants(shot).prompt_parts

    normal = compile_normal_provider_prompt(shot, {}, has_observed_start=False)
    safe = compile_policy_safe_prompt(
        shot,
        has_state_reference=False,
        image_role=None,
        reference_count=0,
    )

    assert expected
    assert all(part in normal for part in expected)
    assert all(part in safe for part in expected)
    for profile, prompt in (("normal", normal), ("policy_safe", safe)):
        assert provider_submission_issues(
            prompt,
            shot,
            {"resolution": "480p", "aspect_ratio": "16:9"},
            prompt_profile=profile,
            model=config.SEEDANCE_MODEL,
            resolution="480p",
            duration=5,
        ) == []


def test_provider_preflight_rejects_missing_positive_locks():
    shot = _locked_shot()
    prompt = compile_normal_provider_prompt(shot, {}, has_observed_start=False)
    for part in compile_visual_invariants(shot).prompt_parts:
        prompt = prompt.replace(part, "")

    issues = provider_submission_issues(
        prompt,
        shot,
        {"resolution": "480p", "aspect_ratio": "16:9"},
        prompt_profile="normal",
        model=config.SEEDANCE_MODEL,
        resolution="480p",
        duration=5,
    )

    assert any("Positive Locks" in issue for issue in issues)


def test_visual_invariant_fingerprint_ignores_action_prose_but_tracks_locks():
    shot = _locked_shot()
    original = visual_invariants_fingerprint(shot)
    shot["primary_action"] = "ANOTHER_ACTION"

    assert visual_invariants_fingerprint(shot) == original

    shot["scene_id"] = "different_floor"
    assert visual_invariants_fingerprint(shot) != original


def test_visual_invariant_failure_is_an_objective_semantic_rejection():
    review = SemanticReview.from_dict({
        "accepted": True,
        "required_entities_visible": {"service_unit": True},
        "action_geometry_valid": True,
        "primary_action_completed": True,
        "visual_invariants_valid": False,
        "observed_end_state": {},
        "failure_reason": "",
    }, require_visual_invariants=True)

    assert review.accepted is False
    assert "不变量" in review.failure_reason


def test_semantic_review_contract_uses_the_shared_invariant_projection():
    shot = _locked_shot()

    assert _review_contract(shot)["visual_invariants"] == (
        compile_visual_invariants(shot).review_projection
    )
