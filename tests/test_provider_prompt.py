"""Provider prompt profile tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.provider_prompt import (
    compile_normal_provider_prompt,
    compile_policy_safe_prompt,
)


def test_normal_profile_compiles_explicit_setup_from_visual_fields_and_contract():
    shot = {
        "scene_id": "calibration_bay",
        "scene_description": "RAW_SCENE_RESULT: the marked plants have already wilted",
        "prompt_en": "RAW_PROMPT: the water stream knocks every plant over",
        "primary_action": "RAW_PRIMARY_ACTION: the technician triggers a flood",
        "required_visible_entities": ["technician", "plant_bed"],
        "key_props": ["hose nozzle"],
        "lighting": "cool workshop side light",
        "camera": {
            "start_framing": "wide shot",
            "screen_positions": {
                "technician": "left foreground",
                "plant_bed": "right midground",
            },
        },
        "action_beats": [{
            "actor": "technician",
            "action": "RAW_BEAT_ACTION: floods the plants",
            "target": "plant_bed",
            "visible_result": "RAW_BEAT_RESULT: plants are soaked",
        }],
        "narrative_beat": {
            "function": "setup",
            "state_before": "RAW_NARRATIVE_BEFORE",
            "state_change": "RAW_NARRATIVE_RESULT",
            "state_after": "RAW_NARRATIVE_AFTER",
        },
        "start_state": {
            "action_phase": "RAW_START_ACTION_PHASE",
            "open_motion": "RAW_START_OPEN_MOTION",
        },
        "end_state": {
            "action_phase": "RAW_END_ACTION_PHASE",
            "pose_and_gaze": "PREPARATION_ENDPOINT",
            "prop_state": "RAW_PROP_RESULT",
            "open_motion": "RAW_END_OPEN_MOTION",
        },
        "interaction_geometry": {
            "actor": "technician",
            "target": "plant_bed",
            "effect_phase": "setup",
            "interaction_mode": "none",
            "outcome_scope": "none",
            "effect_motion": "none",
        },
    }

    prompt = compile_normal_provider_prompt(
        shot,
        {"style": "cinematic realism", "mood": "focused", "characters": []},
        has_observed_start=False,
    )

    assert "technician visibly prepares toward plant_bed" in prompt
    assert "PREPARATION_ENDPOINT" in prompt
    for forbidden in (
        "RAW_SCENE_RESULT",
        "RAW_PROMPT",
        "RAW_PRIMARY_ACTION",
        "RAW_BEAT_ACTION",
        "RAW_BEAT_RESULT",
        "RAW_NARRATIVE_BEFORE",
        "RAW_NARRATIVE_RESULT",
        "RAW_NARRATIVE_AFTER",
        "RAW_START_OPEN_MOTION",
        "RAW_START_ACTION_PHASE",
        "RAW_END_OPEN_MOTION",
        "RAW_END_ACTION_PHASE",
        "RAW_PROP_RESULT",
    ):
        assert forbidden not in prompt


def test_normal_profile_keeps_active_contract_result_but_not_raw_scene_prose():
    shot = {
        "scene_id": "assembly_floor",
        "scene_description": "RAW_SCENE_PROSE",
        "prompt_en": "RAW_PROMPT_PROSE",
        "primary_action": "press arm lowers one clamp",
        "action_beats": [{
            "actor": "press_arm",
            "action": "lowers one clamp",
            "target": "workpiece",
            "visible_result": "ACTIVE_CONTRACTED_RESULT",
        }],
        "interaction_geometry": {
            "actor": "press_arm",
            "target": "workpiece",
            "effect_phase": "active",
            "interaction_mode": "direct_contact",
            "outcome_scope": "single",
            "effect_motion": "sweep",
        },
    }

    prompt = compile_normal_provider_prompt(shot, {}, has_observed_start=False)

    assert "ACTIVE_CONTRACTED_RESULT" in prompt
    assert "effect motion static" in prompt
    assert "RAW_SCENE_PROSE" not in prompt
    assert "RAW_PROMPT_PROSE" not in prompt


def test_normal_profile_projects_declared_character_identity_without_action_prose():
    shot = {
        "scene_id": "studio_floor",
        "characters": ["service_unit"],
        "interaction_geometry": {"effect_phase": "none"},
    }
    storyboard = {
        "characters": [{
            "name": "service_unit",
            "description": "brushed steel shell, amber optical sensor",
            "mobility": "tracked",
        }],
    }

    prompt = compile_normal_provider_prompt(
        shot, storyboard, has_observed_start=False
    )

    assert "declared appearance for service unit" in prompt
    assert "brushed steel shell, amber optical sensor" in prompt
    assert "mobility tracked" in prompt


def test_normal_profile_never_claims_a_previous_tail_without_state_reference():
    shot = {
        "composition_change": "large",
        "interaction_geometry": {"effect_phase": "none"},
    }

    prompt = compile_normal_provider_prompt(shot, {}, has_observed_start=False)

    assert "supplied previous tail" not in prompt


@pytest.mark.parametrize(("phase", "mode", "expected"), [
    ("setup", "none", "intent only"),
    ("active", "direct_contact", "controlled contact action"),
    ("active", "directed_path", "readable path from source to target"),
    ("active", "area_effect", "contained area action"),
    ("active", "indirect_effect", "readable intermediary and result"),
    ("aftermath", "none", "do not introduce a new physical action"),
])
def test_policy_safe_profile_is_theme_agnostic_and_excludes_freeform_prose(
    phase, mode, expected
):
    shot = {
        "scene_id": "test_stage",
        "prompt_en": "RAW_PROMPT",
        "primary_action": "RAW_PRIMARY_ACTION",
        "negative_prompt": "RAW_NEGATIVE_PROMPT",
        "required_visible_entities": ["lead_unit", "target_group"],
        "camera": {
            "start_framing": "wide shot RAW_CAMERA",
            "screen_positions": {
                "lead_unit": "left foreground RAW_POSITION",
                "target_group": "right background",
            },
        },
        "action_beats": [{
            "actor": "lead_unit",
            "action": "RAW_BEAT_ACTION",
            "target": "target_group",
            "visible_result": "RAW_VISIBLE_RESULT",
        }],
        "narrative_beat": {
            "state_before": "RAW_BEFORE",
            "state_change": "RAW_CHANGE",
            "state_after": "RAW_AFTER",
        },
        "interaction_geometry": {
            "actor": "lead_unit",
            "target": "target_group",
            "effect_phase": phase,
            "interaction_mode": mode,
            "outcome_scope": "single",
            "effect_motion": "static",
            "source": "RAW_SOURCE",
            "effect_region": "RAW_REGION",
            "reaction_scope": "RAW_SCOPE",
            "unaffected_behavior": "RAW_UNAFFECTED",
        },
    }

    prompt = compile_policy_safe_prompt(
        shot,
        has_state_reference=False,
        image_role=None,
        reference_count=0,
    )

    assert expected in prompt
    assert "lead unit, target group" in prompt
    assert "lead unit=left foreground" in prompt
    assert "target group=right background" in prompt
    if phase == "active":
        # The final result is Contract-owned and must survive a safe retry.
        assert "RAW_VISIBLE_RESULT" in prompt
        for forbidden in (
            "RAW_PROMPT",
            "RAW_PRIMARY_ACTION",
            "RAW_NEGATIVE_PROMPT",
            "RAW_BEAT_ACTION",
            "RAW_BEFORE",
            "RAW_CHANGE",
            "RAW_AFTER",
            "RAW_SOURCE",
            "RAW_REGION",
            "RAW_SCOPE",
            "RAW_UNAFFECTED",
        ):
            assert forbidden not in prompt
    else:
        assert "RAW_" not in prompt


def test_policy_safe_profile_preserves_reference_responsibilities():
    shot = {
        "required_visible_entities": ["subject"],
        "interaction_geometry": {"actor": "subject", "effect_phase": "none"},
    }

    prompt = compile_policy_safe_prompt(
        shot,
        has_state_reference=True,
        image_role="reference_image",
        reference_count=2,
    )

    assert "Image 1 controls only the accepted prior scene state" in prompt
    assert "remaining images control identity and appearance only" in prompt


def test_policy_safe_profile_keeps_shared_visual_contract_projection():
    prompt = compile_policy_safe_prompt(
        {
            "scene_id": "calibration_bay",
            "characters": ["service_unit"],
            "lighting": "cool side light",
            "key_props": ["sensor wand"],
            "interaction_geometry": {"effect_phase": "setup", "actor": "service_unit"},
        },
        storyboard={
            "style": "cinematic realism",
            "mood": "restrained",
            "characters": [{
                "name": "service_unit",
                "description": "brushed steel shell, amber optical sensor",
            }],
        },
        has_state_reference=False,
        image_role=None,
        reference_count=0,
    )

    assert "visual style: cinematic realism" in prompt
    assert "mood: restrained" in prompt
    assert "lighting: cool side light" in prompt
    assert "visible props: sensor wand" in prompt
    assert "declared appearance for service unit" in prompt


def test_policy_safe_profile_uses_canonical_single_scope_not_raw_sweep():
    prompt = compile_policy_safe_prompt(
        {
            "interaction_geometry": {
                "actor": "irrigation_unit",
                "target": "plant_bed",
                "effect_phase": "active",
                "interaction_mode": "directed_path",
                "outcome_scope": "single",
                "effect_motion": "sweep",
                "reaction_scope": "every plant in the bed",
            },
        },
        has_state_reference=False,
        image_role=None,
        reference_count=0,
    )

    assert "effect motion static" in prompt
    assert "one clearly isolated intended target" in prompt
    assert "every plant in the bed" not in prompt


def test_policy_safe_profile_preserves_contract_result_and_unaffected_boundary():
    shot = {
        "action_beats": [{
            "actor": "irrigation_unit",
            "action": "directs one water stream",
            "target": "plant_bed",
            "visible_result": "the selected plant becomes visibly wet while the neighboring plant remains dry",
        }],
        "interaction_geometry": {
            "actor": "irrigation_unit",
            "target": "plant_bed",
            "effect_phase": "active",
            "interaction_mode": "directed_path",
            "outcome_scope": "single",
            "effect_motion": "static",
            "unaffected_behavior": "neighboring plant remains dry and still",
        },
    }

    normal = compile_normal_provider_prompt(shot, {}, has_observed_start=False)
    safe = compile_policy_safe_prompt(
        shot,
        has_state_reference=False,
        image_role=None,
        reference_count=0,
    )

    endpoint = "the selected plant becomes visibly wet while the neighboring plant remains dry"
    assert endpoint in normal
    assert endpoint in safe
    assert "all entities outside that one intended target continue their prior motion" in safe
    assert "stop or recoil" not in safe


def test_policy_safe_retake_keeps_bounded_contract_direction_without_failure_prose():
    from pipeline.causality import compile_action_contract

    shot = {"interaction_geometry": {"effect_phase": "setup", "actor": "unit"}}
    prompt = compile_policy_safe_prompt(
        shot,
        has_state_reference=False,
        image_role=None,
        reference_count=0,
        retake_instruction=compile_action_contract(shot).safe_retake_instruction(),
    )

    assert "Targeted retake: enforce the contracted setup non-effect evidence only" in prompt
    assert "previous take failed" not in prompt
