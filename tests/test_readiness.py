"""Generation readiness gate tests."""

import pytest

from pipeline.causality import (
    blocking_geometry_issues,
    causal_evidence_issues,
    causal_storyboard_issues,
)
from pipeline.readiness import (
    coverage_readiness_issues,
    GenerationReadinessError,
    ensure_shot_ready,
    storyboard_readiness_issues,
)
from pipeline.participants import visible_character_names
from pipeline.production_plan import apply_production_plan, build_production_plan


def test_visible_participants_cover_all_structured_character_fields():
    shot = {
        "characters": ["robot"],
        "camera": {"screen_positions": {"zombies": "background"}},
        "blocking": {
            "guard": {"action_target": "robot"},
            "robot": {"eyeline_target": "observer"},
        },
        "action_beats": [{"actor": "observer", "target": "guard"}],
    }

    assert visible_character_names(
        shot, ["robot", "zombies", "guard", "observer"]
    ) == ["robot", "zombies", "guard", "observer"]


def test_multi_character_extraction_is_rejected_before_generation():
    issues = storyboard_readiness_issues({
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "prompt_en": "robot faces zombies",
            "characters": ["robot", "zombies"],
            "extract_character_ref": True,
        }],
    })

    assert any("单一 identity 角色" in issue for issue in issues)


def test_hidden_structured_participant_is_rejected_before_generation():
    issues = storyboard_readiness_issues({
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "prompt_en": "robot faces zombies",
            "characters": ["robot"],
            "extract_character_ref": True,
            "camera": {
                "screen_positions": {
                    "robot": "left foreground",
                    "zombies": "center background",
                }
            },
        }],
    })

    assert any("实际可见角色" in issue for issue in issues)
    assert any("单一 identity 角色" in issue for issue in issues)


def test_readiness_rejects_production_plan_drift_before_generation():
    plan = build_production_plan("balanced", 5)
    storyboard = {
        "content_focus": "balanced",
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "prompt_en": "a complete cinematic moment",
            "scene_description": "interior",
            "camera": {"start_framing": "medium shot"},
            "coverage_role": "aftermath",
        }],
    }
    apply_production_plan(storyboard, plan)
    storyboard["shots"][0]["duration"] = 6

    issues = storyboard_readiness_issues(storyboard)

    assert any("duration 必须保持计划值" in issue for issue in issues)


def test_seamless_shot_without_accepted_tail_frame_fails_once():
    with pytest.raises(GenerationReadinessError, match="上一镜尾帧"):
        ensure_shot_ready(
            {"shot_id": 2, "continuity_from_previous": "seamless"},
            previous_frame=None,
            character_refs={},
        )


def test_small_same_scene_cut_without_accepted_tail_frame_fails_once():
    with pytest.raises(GenerationReadinessError, match="上一镜尾帧"):
        ensure_shot_ready(
            {
                "shot_id": 2,
                "scene_id": "street",
                "continuity_from_previous": "intentional_cut",
                "composition_change": "small",
            },
            previous_frame=None,
            previous_shot={"shot_id": 1, "scene_id": "street"},
            character_refs={},
        )


def test_large_same_scene_cut_requires_state_tail_without_locking_composition():
    with pytest.raises(GenerationReadinessError, match="缺少已接受的上一镜尾帧"):
        ensure_shot_ready(
            {
                "shot_id": 2,
                "scene_id": "street",
                "continuity_from_previous": "intentional_cut",
                "composition_change": "large",
            },
            previous_frame=None,
            previous_shot={
                "shot_id": 1,
                "scene_id": "street",
                "output_reference_depth": 0,
            },
            character_refs={},
        )


def test_multi_character_action_requires_blocking_before_paid_generation():
    issues = storyboard_readiness_issues({
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [
            {
                "shot_id": 1,
                "duration": 4,
                "prompt_en": "robot portrait",
                "characters": ["robot"],
                "extract_character_ref": True,
            },
            {
                "shot_id": 2,
                "duration": 5,
                "prompt_en": "robot fires at zombies",
                "primary_action": "robot fires one burst at zombies",
                "characters": ["robot", "zombies"],
            },
        ],
    })

    assert any("blocking 不完整" in issue for issue in issues)
    assert any("缺少 action_beats" in issue for issue in issues)


def test_blocking_geometry_rejects_camera_facing_actor_with_deeper_target():
    shot = {
        "shot_id": 1,
        "interaction_geometry": {"actor": "actor", "target": "target"},
        "camera": {
            "screen_positions": {
                "actor": "center foreground",
                "target": "center background",
            },
        },
        "blocking": {
            "actor": {
                "body_orientation": "front toward camera",
                "facing_target": "target",
                "action_target": "target",
            },
        },
    }

    assert any("身体朝向与目标景深矛盾" in issue for issue in blocking_geometry_issues(shot))


def test_blocking_geometry_accepts_actor_facing_deeper_target():
    shot = {
        "shot_id": 1,
        "interaction_geometry": {"actor": "actor", "target": "target"},
        "camera": {
            "screen_positions": {
                "actor": "center foreground",
                "target": "center background",
            },
        },
        "blocking": {
            "actor": {
                "body_orientation": "back three-quarter toward background",
                "facing_target": "target",
                "action_target": "target",
            },
        },
    }

    assert blocking_geometry_issues(shot) == []


def test_impossible_closeup_interaction_is_rejected_before_paid_generation():
    storyboard = {
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 2,
            "duration": 5,
            "scene_description": "Ruined street firefight",
            "prompt_en": "Extreme close-up of the robot firing at distant zombies.",
            "primary_action": "robot fires at zombies",
            "characters": ["robot", "zombies"],
            "coverage_role": "interaction",
            "required_visible_entities": ["robot", "zombies"],
            "camera": {
                "start_framing": "extreme close-up",
                "end_framing": "extreme close-up",
                "composition": "shallow depth of field",
                "screen_positions": {
                    "robot": "center foreground",
                    "zombies": "distant background",
                },
            },
            "blocking": {
                "robot": {
                    "frame_position": "center foreground",
                    "body_orientation": "toward background",
                    "facing_target": "zombies",
                    "eyeline_target": "zombies",
                    "action_target": "zombies",
                },
                "zombies": {
                    "frame_position": "distant background",
                    "body_orientation": "toward foreground",
                    "facing_target": "robot",
                    "eyeline_target": "robot",
                    "action_target": "robot",
                },
            },
            "action_beats": [{
                "phase": "peak",
                "actor": "robot",
                "action": "fires one burst",
                "target": "zombies",
                "visible_result": "bullets strike the lead zombie",
            }],
            "interaction_geometry": {
                "actor": "robot",
                "target": "zombies",
                "must_share_frame": True,
                "line_of_action_visible": True,
                "actor_screen_position": "center foreground",
                "target_screen_position": "distant background",
                "occlusion_policy": "none",
            },
        }],
    }

    issues = storyboard_readiness_issues(storyboard)

    assert any("极近景" in issue and "同框" in issue for issue in issues)
    assert any("远处目标" in issue and "命中结果" in issue for issue in issues)


def test_directed_path_interaction_requires_a_complete_causal_scope():
    issues = storyboard_readiness_issues({
        "characters": [
            {"name": "actor", "reference_mode": "identity"},
            {"name": "target", "reference_mode": "identity"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "prompt_en": "A visible directed interaction crosses the frame.",
            "primary_action": "actor projects force toward target",
            "characters": ["actor", "target"],
            "interaction_geometry": {
                "actor": "actor",
                "target": "target",
                "must_share_frame": True,
                "line_of_action_visible": True,
                "interaction_mode": "directed_path",
                "source": "",
                "effect_region": "",
                "reaction_scope": "",
                "unaffected_behavior": "",
            },
        }],
    })

    assert any("作用来源" in issue for issue in issues)
    assert any("作用区域" in issue for issue in issues)
    assert any("反应范围" in issue for issue in issues)
    assert any("范围外行为" in issue for issue in issues)


def test_area_effect_causality_is_checked_without_a_visible_action_line():
    issues = storyboard_readiness_issues({
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "prompt_en": "A visible area effect changes the environment.",
            "primary_action": "an expanding force affects the marked area",
            "interaction_geometry": {
                "interaction_mode": "area_effect",
                "source": "expanding force",
                "effect_region": "",
                "reaction_scope": "subjects inside the marked area",
                "unaffected_behavior": "subjects outside the area remain unchanged",
            },
        }],
    })

    assert any("作用区域" in issue for issue in issues)
    assert not any("作用路径可见" in issue for issue in issues)


def test_area_effect_does_not_require_offscreen_source_actor_in_frame():
    issues = coverage_readiness_issues({
        "shot_id": 1,
        "required_visible_entities": ["affected_group"],
        "action_beats": [{
            "phase": "peak",
            "actor": "offscreen_trigger",
            "action": "activates an expanding effect",
            "target": "affected_group",
            "visible_result": "members inside the marked region react",
        }],
        "interaction_geometry": {
            "actor": "offscreen_trigger",
            "target": "affected_group",
            "interaction_mode": "area_effect",
            "source": "an effect entering from outside frame",
            "effect_region": "the visibly marked central region",
            "reaction_scope": "members inside the region",
            "unaffected_behavior": "members outside continue unchanged",
            "must_share_frame": False,
            "line_of_action_visible": False,
        },
    })

    assert not any("actor/target" in issue for issue in issues)


def test_setup_phase_cannot_be_promoted_to_a_physical_effect():
    issues = causal_storyboard_issues({
        "shots": [{
            "shot_id": 1,
            "interaction_geometry": {
                "actor": "robot",
                "target": "crowd",
                "interaction_mode": "directed_path",
                "effect_phase": "setup",
                "outcome_scope": "none",
                "effect_motion": "static",
            },
        }],
    }, required=True)

    assert any("准备阶段" in issue and "interaction_mode=none" in issue for issue in issues)


def test_aftermath_cannot_expand_partial_effect_to_whole_group():
    issues = causal_storyboard_issues({
        "shots": [
            {
                "shot_id": 1,
                "interaction_geometry": {
                    "actor": "source",
                    "target": "crowd",
                    "interaction_mode": "directed_path",
                    "effect_phase": "active",
                    "outcome_scope": "subset",
                    "effect_motion": "static",
                },
            },
            {
                "shot_id": 2,
                "interaction_geometry": {
                    "actor": "source",
                    "target": "crowd",
                    "interaction_mode": "directed_path",
                    "effect_phase": "aftermath",
                    "outcome_scope": "all",
                    "effect_motion": "none",
                },
            },
        ],
    }, required=True)

    assert any("结果范围" in issue and "无原因扩大" in issue for issue in issues)


def test_static_directed_path_cannot_claim_whole_group_outcome():
    issues = causal_storyboard_issues({
        "shots": [{
            "shot_id": 1,
            "interaction_geometry": {
                "actor": "source",
                "target": "crowd",
                "interaction_mode": "directed_path",
                "effect_phase": "active",
                "outcome_scope": "all",
                "effect_motion": "static",
            },
        }],
    }, required=True)

    assert any("全部目标" in issue and "sweep" in issue for issue in issues)


def test_sample_evidence_rejects_reaction_outside_effect_path():
    issues = causal_evidence_issues(
        {
            "interaction_geometry": {
                "effect_phase": "active",
                "outcome_scope": "subset",
            },
        },
        [
            {
                "physical_effect_visible": True,
                "reaction_visible": True,
                "effect_intersects_reaction": False,
                "out_of_scope_reaction_visible": True,
                "contracted_outcome_visible": False,
                "outcome_causally_connected": False,
            },
        ],
    )

    assert any("未与作用区域相交" in issue for issue in issues)
    assert any("范围外目标发生反应" in issue for issue in issues)


def test_sample_evidence_rejects_physical_effect_during_setup():
    issues = causal_evidence_issues(
        {
            "interaction_geometry": {
                "effect_phase": "setup",
                "outcome_scope": "none",
            },
        },
        [{
            "physical_effect_visible": True,
            "reaction_visible": False,
            "effect_intersects_reaction": False,
            "out_of_scope_reaction_visible": False,
            "contracted_outcome_visible": False,
            "outcome_causally_connected": False,
        }],
    )

    assert any("准备阶段提前出现物理作用" in issue for issue in issues)


def test_setup_does_not_treat_preexisting_target_motion_as_effect_reaction():
    issues = causal_evidence_issues(
        {"interaction_geometry": {"effect_phase": "setup"}},
        [{
            "physical_effect_visible": False,
            "reaction_visible": True,
            "effect_intersects_reaction": False,
            "out_of_scope_reaction_visible": False,
            "contracted_outcome_visible": False,
            "outcome_causally_connected": False,
        }],
    )

    assert issues == []


@pytest.mark.parametrize("phase", ["active", "aftermath"])
def test_effect_phase_requires_the_contracted_outcome_to_be_visible(phase):
    issues = causal_evidence_issues(
        {
            "interaction_geometry": {
                "effect_phase": phase,
                "outcome_scope": "subset",
            },
        },
        [{
            "physical_effect_visible": phase == "active",
            "reaction_visible": phase == "active",
            "effect_intersects_reaction": phase == "active",
            "out_of_scope_reaction_visible": False,
            "contracted_outcome_visible": False,
            "outcome_causally_connected": False,
        }],
    )

    assert any("约定结果" in issue for issue in issues)


def test_aftermath_cannot_show_a_new_physical_effect():
    issues = causal_evidence_issues(
        {"interaction_geometry": {"effect_phase": "aftermath"}},
        [{
            "physical_effect_visible": True,
            "reaction_visible": False,
            "effect_intersects_reaction": False,
            "out_of_scope_reaction_visible": False,
            "contracted_outcome_visible": True,
            "outcome_causally_connected": False,
        }],
    )

    assert any("aftermath" in issue and "新物理作用" in issue for issue in issues)


def test_active_outcome_cannot_appear_before_its_physical_cause():
    issues = causal_evidence_issues(
        {"interaction_geometry": {"effect_phase": "active"}},
        [
            {
                "physical_effect_visible": False,
                "reaction_visible": False,
                "effect_intersects_reaction": False,
                "out_of_scope_reaction_visible": False,
                "contracted_outcome_visible": True,
                "outcome_causally_connected": True,
            },
            {
                "physical_effect_visible": True,
                "reaction_visible": True,
                "effect_intersects_reaction": True,
                "out_of_scope_reaction_visible": False,
                "contracted_outcome_visible": True,
                "outcome_causally_connected": True,
            },
        ],
    )

    assert any("先于物理原因" in issue for issue in issues)


def test_active_outcome_must_be_visibly_connected_to_its_effect():
    issues = causal_evidence_issues(
        {"interaction_geometry": {"effect_phase": "active"}},
        [{
            "physical_effect_visible": True,
            "reaction_visible": True,
            "effect_intersects_reaction": True,
            "out_of_scope_reaction_visible": False,
            "contracted_outcome_visible": True,
            "outcome_causally_connected": False,
        }],
    )

    assert any("约定结果与物理作用之间缺少可见因果过渡" in issue for issue in issues)


def test_unspecified_phase_does_not_hide_legacy_visible_interaction():
    issues = coverage_readiness_issues(
        {
            "shot_id": 1,
            "action_beats": [{
                "phase": "peak",
                "actor": "source",
                "target": "target",
                "visible_result": "target visibly changes",
            }],
            "interaction_geometry": {
                "effect_phase": "unspecified",
                "interaction_mode": "none",
            },
        },
        require_causality_contract=True,
    )

    assert any("interaction_mode" in issue for issue in issues)


@pytest.mark.parametrize("phase", ["setup", "aftermath"])
def test_non_active_phase_visible_result_is_not_promoted_to_interaction(phase):
    issues = coverage_readiness_issues(
        {
            "shot_id": 1,
            "required_visible_entities": ["subject"],
            "action_beats": [{
                "phase": "aftermath",
                "actor": "subject",
                "target": "group",
                "visible_result": "the planned state is visible",
            }],
            "interaction_geometry": {
                "actor": "subject",
                "target": "group",
                "effect_phase": phase,
                "interaction_mode": "none",
            },
        },
        require_causality_contract=True,
    )

    assert not any("interaction_mode" in issue for issue in issues)
    assert not any("actor/target" in issue for issue in issues)


def test_partial_narrative_contract_is_rejected_before_generation():
    issues = storyboard_readiness_issues({
        "story_arc": {
            "goal": "reach a safe state",
            "stakes": "failure leaves the problem unresolved",
            "turning_point": "the first approach fails",
            "resolution": "",
        },
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "prompt_en": "A subject visibly changes the current situation.",
            "primary_action": "the subject acts",
            "narrative_beat": {
                "function": "progress",
                "state_before": "problem is active",
                "state_change": "",
                "state_after": "problem is reduced",
            },
        }],
    })

    assert any("story_arc.resolution" in issue for issue in issues)
    assert any("narrative_beat.state_change" in issue for issue in issues)


def test_multiple_active_action_actors_are_rejected_before_generation():
    blocking = {}
    for name, side, target in (
        ("robot", "left", "zombies"),
        ("zombies", "right", "robot"),
    ):
        blocking[name] = {
            "frame_position": side,
            "body_orientation": "toward opponent",
            "facing_target": target,
            "eyeline_target": target,
            "action_target": target,
        }
    storyboard = {
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 3,
            "duration": 5,
            "scene_description": "Ruined street firefight",
            "prompt_en": "Zombies charge while the robot fires.",
            "primary_action": "zombies charge at the robot",
            "characters": ["robot", "zombies"],
            "camera": {"screen_positions": {"robot": "left", "zombies": "right"}},
            "blocking": blocking,
            "action_beats": [
                {"phase": "trigger", "actor": "zombies", "action": "charge", "target": "robot"},
                {"phase": "peak", "actor": "robot", "action": "fires", "target": "zombies"},
            ],
        }],
    }

    issues = storyboard_readiness_issues(storyboard)

    assert any("多个主动动作执行者" in issue for issue in issues)


def test_short_action_story_does_not_spend_a_shot_on_identity_extraction():
    issues = storyboard_readiness_issues({
        "content_focus": "action",
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "prompt_en": "robot faces zombies",
            "primary_action": "robot fires one burst at zombies",
            "characters": ["robot", "zombies"],
            "extract_character_ref": False,
            "action_beats": [{
                "phase": "peak",
                "actor": "robot",
                "action": "fires one burst",
                "target": "zombies",
                "visible_result": "zombies recoil",
            }],
            "blocking": {
                "robot": {
                    "frame_position": "left",
                    "body_orientation": "profile right",
                    "facing_target": "zombies",
                    "eyeline_target": "zombies",
                    "action_target": "zombies",
                },
                "zombies": {
                    "frame_position": "right",
                    "body_orientation": "profile left",
                    "facing_target": "robot",
                    "eyeline_target": "robot",
                    "action_target": "robot",
                },
            },
        }],
    })

    assert not any("首次出现必须是可提取" in issue for issue in issues)


def test_long_action_story_can_open_with_identity_character_in_group_shot():
    issues = storyboard_readiness_issues({
        "content_focus": "action",
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [
            {
                "shot_id": 1,
                "duration": 15,
                "prompt_en": "robot and zombies face each other",
                "characters": ["robot", "zombies"],
                "extract_character_ref": False,
            },
            {
                "shot_id": 2,
                "duration": 15,
                "prompt_en": "robot crosses the ruined street",
                "characters": ["robot"],
                "extract_character_ref": False,
            },
        ],
    })

    assert not any("首次出现必须是可提取" in issue for issue in issues)
