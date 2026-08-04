"""Storyboard continuity and motion-budget contract tests."""

import pytest

from pipeline.causality import (
    blocking_geometry_issues,
    causal_evidence_issues,
    compile_interaction_blocking,
    normalize_causal_scope,
)
from pipeline.readiness import storyboard_readiness_issues
from pipeline.storyboard import (
    _BUILTIN_SYSTEM_PROMPT,
    _apply_defaults,
    _build_correction_prompt,
    _compile_storyboard_contract,
    _infer_content_focus,
    _scene_id,
    _shot_advances_action_conflict,
    _validate_storyboard_richness,
)


def _shot(shot_id: int, **overrides) -> dict:
    shot = {
        "shot_id": shot_id,
        "duration": 5,
        "scene_description": "【废墟路口·远景】测试镜头",
        "prompt_en": "word " * 80,
        "camera": {"speed": "slow", "start_framing": "wide shot"},
        "mood": f"mood-{shot_id}",
        "characters": ["hero"],
        "key_props": [],
    }
    shot.update(overrides)
    return shot


def _contract_shot(
    shot_id: int,
    duration: int,
    primary_action: str,
    **overrides,
) -> dict:
    shot = _shot(
        shot_id,
        duration=duration,
        primary_action=primary_action,
        extract_character_ref=shot_id == 1,
        camera={
            "speed": "slow",
            "start_framing": "wide shot",
            "end_framing": "medium shot",
        },
        start_state={
            "location": "arena",
            "subject": "fighters ready",
            "action_phase": "start",
            "camera": "wide shot",
        },
        end_state={
            "location": "arena",
            "subject": "fighters complete the beat",
            "action_phase": "end",
            "camera": "medium shot",
        },
    )
    shot.update(overrides)
    return shot


def test_equal_shot_durations_are_not_a_critical_storyboard_error():
    storyboard = {
        "title": "equal duration sequence",
        "shots": [
            _contract_shot(1, 5, "hero observes the room"),
            _contract_shot(2, 5, "hero opens the door"),
            _contract_shot(3, 5, "hero exits the room"),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(storyboard)

    assert not is_critical
    assert not any("时长全部相同" in warning for warning in warnings)


def test_content_focus_is_inferred_from_explicit_request():
    assert _infer_content_focus("制作一个30秒的孙悟空大战龟仙人") == "action"
    assert _infer_content_focus("智能手表产品宣传片") == "product"
    assert _infer_content_focus("制作一个猫咖日常短视频") == "balanced"


def test_spatial_compiler_derives_axis_and_blocking_from_actor_target():
    shot = {
        "characters": ["actor", "target"],
        "interaction_geometry": {"actor": "actor", "target": "target"},
        "camera": {},
        "blocking": {},
    }

    compile_interaction_blocking(shot)

    assert shot["camera"]["screen_positions"] == {
        "actor": "left foreground",
        "target": "right midground",
    }
    assert shot["blocking"]["actor"]["facing_target"] == "target"
    assert shot["blocking"]["target"]["facing_target"] == "actor"


def test_defaults_compile_unique_local_participant_alias_to_catalog_id():
    storyboard = {
        "characters": [
            {"name": "lead_orbit", "reference_mode": "identity"},
            {"name": "orbit_group", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "characters": ["lead_orbit", "the final orbit"],
            "required_visible_entities": ["lead_orbit", "the final orbit"],
            "camera": {"screen_positions": {
                "lead_orbit": "left foreground",
                "the final orbit": "right midground",
            }},
            "blocking": {
                "lead_orbit": {"action_target": "the final orbit"},
                "the final orbit": {"action_target": "lead_orbit"},
            },
            "action_beats": [{
                "actor": "lead_orbit",
                "target": "the final orbit",
                "visible_result": "the target visibly changes state",
            }],
            "interaction_geometry": {
                "effect_phase": "active",
                "interaction_mode": "directed_path",
                "outcome_scope": "single",
                "effect_motion": "static",
                "actor": "lead_orbit",
                "target": "the final orbit",
            },
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    shot = storyboard["shots"][0]
    assert shot["characters"] == ["lead_orbit", "orbit_group"]
    assert shot["interaction_geometry"]["target"] == "orbit_group"
    assert not any(
        "未定义角色" in issue for issue in storyboard_readiness_issues(storyboard)
    )


def test_defaults_compile_action_beat_alias_against_canonical_roster():
    storyboard = {
        "characters": [
            {"name": "lead_orbit", "reference_mode": "identity"},
            {"name": "orbit_group", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "characters": ["lead_orbit", "the final orbit"],
            "action_beats": [{
                "actor": "lead_orbit",
                "target": "the final remaining orbit",
                "visible_result": "the target visibly changes state",
            }],
            "interaction_geometry": {
                "effect_phase": "active",
                "interaction_mode": "directed_path",
                "outcome_scope": "single",
                "effect_motion": "static",
                "actor": "lead_orbit",
                "target": "the final orbit",
            },
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    shot = storyboard["shots"][0]
    assert shot["characters"] == ["lead_orbit", "orbit_group"]
    assert shot["interaction_geometry"]["target"] == "orbit_group"
    assert shot["action_beats"][0] == {
        "actor": "lead_orbit",
        "target": "orbit_group",
        "visible_result": "the target visibly changes state",
    }


def test_defaults_do_not_bind_a_prop_target_to_the_remaining_participant():
    storyboard = {
        "characters": [
            {"name": "chef", "reference_mode": "identity"},
            {"name": "assistant", "reference_mode": "identity"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "characters": ["chef", "assistant"],
            "key_props": ["mixing bowl"],
            "action_beats": [{
                "actor": "chef",
                "target": "mixing bowl",
                "visible_result": "the bowl visibly changes state",
            }],
            "interaction_geometry": {
                "effect_phase": "active",
                "interaction_mode": "direct_contact",
                "outcome_scope": "single",
                "effect_motion": "static",
                "actor": "chef",
                "target": "mixing bowl",
            },
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    shot = storyboard["shots"][0]
    assert shot["action_beats"][0]["target"] == "mixing bowl"
    assert shot["interaction_geometry"]["target"] == "mixing bowl"


def test_defaults_move_a_declared_prop_alias_out_of_characters():
    storyboard = {
        "characters": [
            {"name": "chef", "reference_mode": "identity"},
            {"name": "assistant", "reference_mode": "identity"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "characters": ["chef", "the mixing bowl"],
            "key_props": ["mixing bowl"],
            "continuity_props": ["the mixing bowl"],
            "action_beats": [{
                "actor": "chef",
                "target": "the mixing bowl",
                "visible_result": "the bowl visibly changes state",
            }],
            "interaction_geometry": {
                "effect_phase": "active",
                "interaction_mode": "direct_contact",
                "outcome_scope": "single",
                "effect_motion": "static",
                "actor": "chef",
                "target": "the mixing bowl",
            },
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    shot = storyboard["shots"][0]
    assert shot["characters"] == ["chef"]
    assert shot["continuity_props"] == ["mixing bowl"]
    assert shot["action_beats"][0]["target"] == "mixing bowl"
    assert shot["interaction_geometry"]["target"] == "mixing bowl"


def test_richness_uses_readiness_as_its_hard_validation_gate():
    storyboard = {
        "characters": [{"name": "catalog_hero", "reference_mode": "identity"}],
        "shots": [
            _contract_shot(
                1,
                5,
                "the hero completes one clear action",
                characters=["local_alias"],
                extract_character_ref=False,
            ),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(storyboard)

    assert is_critical
    assert any("未定义角色 local_alias" in warning for warning in warnings)


def test_richness_reports_each_readiness_issue_once():
    storyboard = {
        "characters": [{"name": "catalog_hero", "reference_mode": "identity"}],
        "shots": [
            _contract_shot(
                1,
                5,
                "the hero completes one clear action",
                characters=["catalog_hero"],
                extract_character_ref=False,
                action_beats=[{
                    "actor": "catalog_hero",
                    "target": "mystery_group",
                    "visible_result": "a state changes",
                }],
            ),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(storyboard)

    assert is_critical
    assert sum("mystery_group" in warning for warning in warnings) == 1


def test_participant_catalog_constraint_is_present_in_generation_and_repair_prompts():
    repair = _build_correction_prompt(
        "original request",
        {"characters": [{"name": "catalog_hero"}], "shots": []},
        ["🚨 Shot 1: 未定义角色 local_alias"],
    )

    constraint = "actor 和角色 target 必须逐字复用顶层 characters.name"
    assert constraint in _BUILTIN_SYSTEM_PROMPT
    assert constraint in repair
    assert "key_props/continuity_props" in _BUILTIN_SYSTEM_PROMPT
    assert "key_props/continuity_props" in repair


def test_causal_compiler_prevents_aftermath_scope_expansion():
    shots = [
        {
            "shot_id": 1,
            "scene_id": "scene",
            "interaction_geometry": {
                "target": "group",
                "effect_phase": "active",
                "outcome_scope": "subset",
            },
        },
        {
            "shot_id": 2,
            "scene_id": "scene",
            "interaction_geometry": {
                "target": "group",
                "effect_phase": "aftermath",
                "outcome_scope": "all",
            },
        },
    ]

    corrections = normalize_causal_scope(shots)

    assert shots[1]["interaction_geometry"]["outcome_scope"] == "subset"
    assert corrections


def test_setup_still_rejects_any_emitted_effect_or_early_outcome():
    shot = {
        "interaction_geometry": {
            "effect_phase": "setup",
            "interaction_mode": "none",
            "outcome_scope": "none",
            "effect_motion": "none",
        }
    }
    preparation_only = [{
        "physical_effect_visible": False,
        "reaction_visible": False,
        "effect_intersects_reaction": False,
        "out_of_scope_reaction_visible": False,
        "contracted_outcome_visible": False,
        "outcome_causally_connected": False,
    }]
    emitted_effect = [{**preparation_only[0], "physical_effect_visible": True}]
    early_outcome = [{**preparation_only[0], "contracted_outcome_visible": True}]

    assert causal_evidence_issues(shot, preparation_only) == []
    assert "准备阶段提前出现物理作用" in causal_evidence_issues(
        shot, emitted_effect
    )
    assert "准备阶段提前出现约定结果（叙事结果）" in causal_evidence_issues(
        shot, early_outcome
    )


def test_camera_movement_cannot_be_primary_subject_action():
    storyboard = {
        "title": "fight",
        "shots": [
            _contract_shot(
                1,
                4,
                "Drone slowly pushes in toward two stationary fighters",
            ),
            _contract_shot(2, 5, "fighter throws one straight punch"),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard,
        user_request="孙悟空大战龟仙人",
    )

    assert is_critical
    assert any("运镜不能充当 primary_action" in warning for warning in warnings)
    assert not any("动作节拍过载" in warning for warning in warnings)


def test_action_request_rejects_one_short_combat_beat_in_thirty_seconds():
    storyboard = {
        "title": "fight",
        "shots": [
            _contract_shot(1, 4, "fighters hold their stance"),
            _contract_shot(2, 5, "fighter raises his staff into guard position"),
            _contract_shot(3, 6, "opponent advances across the arena"),
            _contract_shot(4, 8, "fighter fires one blast that strikes the opponent"),
            _contract_shot(5, 7, "fighters walk away from the arena"),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard,
        user_request="制作一个30秒的孙悟空大战龟仙人",
    )

    assert is_critical
    action_warning = next(
        warning for warning in warnings if "动作重心不足" in warning
    )
    assert "Shot 1=" in action_warning
    assert "Shot 5=" in action_warning


def test_chinese_title_uses_structured_theme_anchors_not_english_prompt_text():
    storyboard = {
        "title": "「末日清道夫」机器人清除丧尸",
        "characters": [
            {"name": "combat_bot", "reference_mode": "identity"},
            {"name": "zombie_horde", "reference_mode": "group"},
        ],
        "shots": [
            _contract_shot(
                1, 4, "combat bot fires one burst at the zombie horde",
                characters=["combat_bot", "zombie_horde"],
                extract_character_ref=False,
            ),
            _contract_shot(
                2, 5, "combat bot crushes the lead zombie",
                characters=["combat_bot", "zombie_horde"],
            ),
            _contract_shot(
                3, 6, "combat bot scans the cleared street",
                characters=["combat_bot"],
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="制作一个15秒的视频，机器人末日清除丧尸"
    )

    assert not any("主题锚定不足" in warning for warning in warnings)


def test_product_process_causality_does_not_run_for_action_story():
    storyboard = {
        "title": "robot cleanup",
        "shots": [
            _contract_shot(
                1,
                5,
                "robot takes a sip while monitoring the street",
                prompt_en="The robot takes a sip while monitoring the ruined street. " * 8,
            ),
            _contract_shot(
                2,
                5,
                "robot prepares its weapon",
                prompt_en="The robot prepares its weapon before moving forward. " * 8,
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="机器人末日清除丧尸"
    )

    assert not any("时序倒置" in warning for warning in warnings)


def test_explicit_product_process_still_checks_causal_order():
    storyboard = {
        "title": "coffee process",
        "shots": [
            _contract_shot(
                1,
                5,
                "customer drinks the finished coffee",
                prompt_en="The customer drinks and enjoys the finished coffee. " * 10,
            ),
            _contract_shot(
                2,
                5,
                "barista grinds the coffee beans",
                prompt_en="The barista grinds and prepares fresh coffee beans. " * 10,
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="展示从咖啡豆到成品的完整制作过程"
    )

    assert any("时序倒置" in warning for warning in warnings)


def test_effects_and_environment_key_props_are_not_persistent_prop_continuity():
    storyboard = {
        "title": "robot cleanup",
        "shots": [
            _contract_shot(1, 5, "robot enters the street", key_props=["rifle"]),
            _contract_shot(
                2,
                5,
                "robot fires the rifle",
                key_props=["rifle", "muzzle flash", "bullet casings", "zombie bodies"],
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="机器人末日清除丧尸"
    )

    assert not any("物件凭空出现" in warning for warning in warnings)


def test_unexplained_persistent_prop_still_fails_continuity_check():
    storyboard = {
        "title": "robot cleanup",
        "shots": [
            _contract_shot(
                1,
                5,
                "robot enters the street",
                continuity_props=["rifle"],
            ),
            _contract_shot(
                2,
                5,
                "robot surveys the street",
                continuity_props=["rifle", "supply crate"],
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="机器人末日清除丧尸"
    )

    assert any("物件凭空出现" in warning for warning in warnings)


def test_short_action_budget_does_not_require_a_separate_identity_intro_shot():
    storyboard = {
        "title": "robot cleanup",
        "characters": [
            {"name": "combat_bot", "reference_mode": "identity"},
            {"name": "zombie_horde", "reference_mode": "group"},
        ],
        "shots": [
            _contract_shot(
                1,
                5,
                "combat bot fires one burst at the zombie horde",
                characters=["combat_bot", "zombie_horde"],
                extract_character_ref=False,
            ),
            _contract_shot(
                2,
                5,
                "combat bot crushes the lead zombie",
                characters=["combat_bot", "zombie_horde"],
                extract_character_ref=False,
            ),
            _contract_shot(
                3,
                5,
                "combat bot scans the cleared street",
                characters=["combat_bot"],
                extract_character_ref=False,
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="制作一个15秒的视频，机器人末日清除丧尸"
    )

    assert not any("身份锚点" in warning for warning in warnings)
    assert not any("角色首次出现" in warning for warning in warnings)


def test_long_action_can_introduce_identity_character_inside_the_conflict():
    storyboard = {
        "title": "robot cleanup",
        "characters": [
            {"name": "combat_robot", "reference_mode": "identity"},
            {"name": "zombie_horde", "reference_mode": "group"},
        ],
        "shots": [
            _contract_shot(
                shot_id,
                6,
                "combat robot faces the zombie horde",
                characters=["combat_robot", "zombie_horde"],
                extract_character_ref=False,
            )
            for shot_id in range(1, 6)
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="制作一个30秒的视频，机器人末日清除丧尸"
    )

    assert not any("身份锚点" in warning for warning in warnings)
    assert not any("角色首次出现" in warning for warning in warnings)


def test_insert_shot_requirement_depends_on_content_focus():
    storyboard = {
        "title": "subject",
        "shots": [
            _contract_shot(1, 4, "fighter punches the opponent"),
            _contract_shot(2, 5, "opponent blocks the punch"),
            _contract_shot(3, 6, "fighter counters with one kick"),
        ],
    }

    action_warnings, _ = _validate_storyboard_richness(
        storyboard,
        user_request="两名武术家擂台对决",
    )
    product_warnings, _ = _validate_storyboard_richness(
        storyboard,
        user_request="运动手表产品宣传片",
    )

    assert not any("缺少 insert shot" in warning for warning in action_warnings)
    assert any("缺少 insert shot" in warning for warning in product_warnings)


def test_long_action_sequence_requires_causal_coverage_and_framing_range():
    storyboard = {
        "title": "robot cleanup",
        "shots": [
            _contract_shot(
                shot_id,
                duration,
                "robot fires one burst at the zombie horde",
                coverage_role="action_subject",
                camera={
                    "speed": "fast",
                    "start_framing": framing,
                    "end_framing": framing,
                },
            )
            for shot_id, duration, framing in (
                (1, 6, "wide shot"),
                (2, 7, "medium wide shot"),
                (3, 8, "medium shot"),
                (4, 9, "wide shot"),
            )
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard,
        user_request="制作一个30秒的视频，机器人在末日城市清除丧尸",
    )

    assert is_critical
    assert any("动作结果视角缺失" in warning for warning in warnings)
    assert any("动作镜头职责集中" in warning for warning in warnings)
    assert any("动作景别层次不足" in warning for warning in warnings)


def test_long_action_sequence_accepts_causal_coverage_without_fixed_camera_moves():
    storyboard = {
        "title": "robot cleanup",
        "shots": [
            _contract_shot(
                shot_id,
                duration,
                action,
                coverage_role=coverage_role,
                camera={
                    "speed": "fixed",
                    "start_framing": framing,
                    "end_framing": framing,
                },
            )
            for shot_id, duration, action, coverage_role, framing in (
                (1, 6, "zombies charge toward the robot", "establish", "wide shot"),
                (2, 7, "robot fires one burst at the horde", "interaction", "medium shot"),
                (3, 8, "lead zombie recoils from the impact", "target_reaction", "close-up"),
                (4, 9, "robot holds the cleared intersection", "aftermath", "medium wide shot"),
            )
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard,
        user_request="制作一个30秒的视频，机器人在末日城市清除丧尸",
    )

    assert not any("动作结果视角缺失" in warning for warning in warnings)
    assert not any("动作镜头职责集中" in warning for warning in warnings)
    assert not any("动作景别层次不足" in warning for warning in warnings)


def test_required_narrative_contract_rejects_broken_state_handoff():
    storyboard = {
        "story_arc": {
            "goal": "change the situation",
            "stakes": "the problem persists",
            "turning_point": "new information changes the approach",
            "resolution": "the situation reaches a visible outcome",
        },
        "shots": [
            {
                **_contract_shot(1, 5, "subject reveals an obstacle"),
                "narrative_beat": {
                    "function": "setup",
                    "state_before": "the path appears clear",
                    "state_change": "an obstacle appears",
                    "state_after": "the path is blocked",
                },
            },
            {
                **_contract_shot(2, 6, "subject changes course"),
                "narrative_beat": {
                    "function": "progress",
                    "state_before": "an unrelated reset state",
                    "state_change": "a new route is attempted",
                    "state_after": "the subject approaches the goal",
                },
            },
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard, require_narrative_contract=True
    )

    assert is_critical
    assert any("故事状态交接断裂" in warning for warning in warnings)


def test_compound_contact_sequence_requires_shot_split():
    storyboard = {
        "title": "robot zombie fight",
        "characters": [
            {"name": "clearance_bot", "mobility": "tracked"},
            {"name": "zombie_horde", "mobility": "bipedal"},
        ],
        "shots": [
            _contract_shot(1, 4, "zombie charges toward the robot"),
            _contract_shot(
                2,
                6,
                "A zombie leaps onto the robot's shoulder; "
                "robot crushes its neck and tears it off",
                characters=["clearance_bot", "zombie_horde"],
            ),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard,
        user_request="机器人末日清除丧尸",
    )

    assert is_critical
    assert any("动作节拍过载" in warning for warning in warnings)


def test_causal_action_and_target_state_are_not_mistaken_for_three_beats():
    storyboard = {
        "title": "robot zombie fight",
        "characters": [
            {"name": "clearance_bot", "mobility": "tracked"},
            {"name": "zombie_horde", "mobility": "bipedal"},
        ],
        "shots": [
            _contract_shot(1, 4, "zombie horde charges toward the robot"),
            _contract_shot(
                2,
                6,
                "Robot fires plasma blasts at the charging horde, "
                "incinerating the front line attackers",
                characters=["clearance_bot", "zombie_horde"],
                camera={
                    "speed": "slow",
                    "start_framing": "wide shot",
                    "end_framing": "medium shot",
                    "screen_positions": {
                        "clearance_bot": "left",
                        "zombie_horde": "right",
                    },
                    "axis_change": "establish",
                },
            ),
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard,
        user_request="机器人末日清除丧尸",
    )

    assert not any("动作节拍过载" in warning for warning in warnings)


def test_two_tightly_causal_action_stages_are_not_overloaded():
    storyboard = {
        "title": "robot zombie fight",
        "characters": [{"name": "robot", "mobility": "tracked"}],
        "shots": [
            _contract_shot(
                1,
                5,
                "robot advances toward the lead zombie then crushes it",
                characters=["robot"],
            )
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="机器人末日清除丧尸"
    )

    assert not any("动作节拍过载" in warning for warning in warnings)


def test_three_explicit_action_stages_are_overloaded():
    storyboard = {
        "title": "robot zombie fight",
        "characters": [{"name": "robot", "mobility": "tracked"}],
        "shots": [
            _contract_shot(
                1,
                5,
                "robot advances then stops then crushes the lead zombie",
                characters=["robot"],
            )
        ],
    }

    warnings, _ = _validate_storyboard_richness(
        storyboard, user_request="机器人末日清除丧尸"
    )

    assert any("动作节拍过载" in warning for warning in warnings)


def test_action_must_match_single_character_mobility():
    storyboard = {
        "title": "clearance robot",
        "characters": [
            {
                "name": "clearance_bot",
                "description": "A heavy combat robot with a tracked base instead of legs",
                "mobility": "tracked",
            }
        ],
        "shots": [
            _contract_shot(
                1,
                5,
                "Robot fires one plasma blast",
                characters=["clearance_bot"],
            ),
            _contract_shot(
                2,
                5,
                "Robot steps over the remains and walks toward the skyline",
                characters=["clearance_bot"],
            ),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard,
        user_request="机器人清除丧尸",
    )

    assert is_critical
    assert any("移动形态冲突" in warning for warning in warnings)


def test_action_scene_cannot_flip_screen_axis_without_reestablishing():
    storyboard = {
        "title": "arena fight",
        "characters": [
            {"name": "hero", "mobility": "bipedal"},
            {"name": "villain", "mobility": "bipedal"},
        ],
        "shots": [
            _contract_shot(
                1,
                4,
                "hero punches villain",
                characters=["hero", "villain"],
                camera={
                    "speed": "fast",
                    "start_framing": "wide shot",
                    "end_framing": "wide shot",
                    "screen_positions": {"hero": "left", "villain": "right"},
                    "axis_change": "establish",
                },
            ),
            _contract_shot(
                2,
                5,
                "villain blocks the punch",
                characters=["hero", "villain"],
                camera={
                    "speed": "fast",
                    "start_framing": "medium shot",
                    "end_framing": "medium shot",
                    "screen_positions": {"hero": "right", "villain": "left"},
                    "axis_change": "hold",
                },
            ),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard,
        user_request="两名武术家擂台对决",
    )

    assert is_critical
    assert any("空间轴反转" in warning for warning in warnings)


def test_action_scene_requires_facing_eyeline_and_action_target():
    storyboard = {
        "title": "robot zombie fight",
        "characters": [
            {"name": "robot", "mobility": "tracked"},
            {"name": "zombies", "mobility": "bipedal", "reference_mode": "group"},
        ],
        "shots": [
            _contract_shot(
                1,
                4,
                "robot fires one burst at zombies",
                characters=["robot", "zombies"],
                extract_character_ref=False,
                action_beats=[{
                    "phase": "peak",
                    "actor": "robot",
                    "action": "fires one burst",
                    "target": "zombies",
                    "visible_result": "front rank recoils",
                }],
                camera={
                    "speed": "fast",
                    "start_framing": "wide shot",
                    "end_framing": "wide shot",
                    "screen_positions": {"robot": "left", "zombies": "right"},
                    "axis_change": "establish",
                },
            )
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(
        storyboard, user_request="机器人清除丧尸"
    )

    assert is_critical
    assert any("角色调度不完整" in warning for warning in warnings)


def test_multi_character_frame_cannot_become_identity_reference():
    storyboard = {
        "title": "robot zombie fight",
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [
            _contract_shot(
                1,
                4,
                "robot aims at zombies",
                characters=["robot", "zombies"],
                extract_character_ref=True,
            )
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(storyboard)

    assert is_critical
    assert any("角色参考归属不明确" in warning for warning in warnings)


def test_explicit_scene_id_is_independent_from_display_description():
    first = _shot(1, scene_id="intersection")
    second = _shot(
        2,
        scene_id="intersection",
        scene_description="【近距离交火·中近景】同一地点的新景别",
    )

    assert _scene_id(first) == _scene_id(second) == "intersection"


def test_defaults_do_not_infer_seamless_from_stable_scene_id():
    storyboard = {
        "shots": [
            _shot(1, scene_id="highway"),
            _shot(2, scene_id="intersection"),
            _shot(
                3,
                scene_id="intersection",
                scene_description="【近距离交火】同一地点继续",
            ),
        ]
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert [s["continuity_from_previous"] for s in storyboard["shots"]] == [
        "none",
        "intentional_cut",
        "intentional_cut",
    ]


def test_defaults_clear_unusable_multi_character_identity_extraction():
    storyboard = {
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [
            _shot(
                1,
                characters=["robot", "zombies"],
                extract_character_ref=True,
            )
        ],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][0]["extract_character_ref"] is False


def test_defaults_sync_structured_visible_participants_before_reference_check():
    storyboard = {
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [
            _shot(
                1,
                characters=["robot"],
                extract_character_ref=True,
                camera={
                    "screen_positions": {
                        "robot": "left foreground",
                        "zombies": "center background",
                    }
                },
            )
        ],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][0]["characters"] == ["robot", "zombies"]
    assert storyboard["shots"][0]["extract_character_ref"] is False


def test_defaults_compile_visible_impact_into_shootable_geometry():
    storyboard = {
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "scene_description": "【ruined street】 firefight",
            "prompt_en": "Robot fires one burst at the zombies.",
            "characters": ["robot", "zombies"],
            "camera": {
                "screen_positions": {"robot": "left", "zombies": "right"}
            },
            "action_beats": [{
                "phase": "peak",
                "actor": "robot",
                "action": "fires one burst",
                "target": "zombies",
                "visible_result": "the front zombie recoils",
            }],
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    shot = storyboard["shots"][0]
    assert shot["coverage_role"] == "interaction"
    assert shot["required_visible_entities"] == ["robot", "zombies"]
    assert shot["interaction_geometry"]["must_share_frame"] is True
    assert shot["interaction_geometry"]["line_of_action_visible"] is True


@pytest.mark.parametrize(
    ("mode", "field"),
    (
        ("directed_path", "line_of_action_visible"),
        ("direct_contact", "must_share_frame"),
    ),
)
def test_defaults_derive_causal_mode_invariants(mode, field):
    storyboard = {
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "scene_description": "Abstract interaction",
            "prompt_en": "cinematic interaction detail " * 30,
            "interaction_geometry": {
                "actor": "source",
                "target": "target",
                "interaction_mode": mode,
                "source": "visible origin",
                "effect_region": "contracted effect region",
                "reaction_scope": "only the contracted target",
                "unaffected_behavior": "everything outside remains unchanged",
                "must_share_frame": False,
                "line_of_action_visible": False,
            },
            "action_beats": [{
                "phase": "peak",
                "actor": "source",
                "action": "causes one effect",
                "target": "target",
                "visible_result": "",
            }],
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][0]["interaction_geometry"][field] is True


@pytest.mark.parametrize(
    ("phase", "expected"),
    (
        (
            "setup",
            {
                "interaction_mode": "none",
                "outcome_scope": "none",
                "effect_motion": "none",
            },
        ),
        (
            "aftermath",
            {
                "interaction_mode": "none",
                "outcome_scope": "subset",
                "effect_motion": "none",
            },
        ),
    ),
)
def test_defaults_derive_effect_phase_invariants(phase, expected):
    storyboard = {
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "interaction_geometry": {
                "effect_phase": phase,
                "interaction_mode": "directed_path",
                "outcome_scope": "subset",
                "effect_motion": "sweep",
            },
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    geometry = storyboard["shots"][0]["interaction_geometry"]
    assert {key: geometry[key] for key in expected} == expected


@pytest.mark.parametrize(
    ("mode", "scope", "motion"),
    (
        ("direct_contact", "single", "static"),
        ("directed_path", "subset", "static"),
        ("directed_path", "all", "sweep"),
        ("area_effect", "all", "static"),
        ("indirect_effect", "subset", "propagate"),
    ),
)
def test_defaults_compile_missing_active_effect_motion(mode, scope, motion):
    storyboard = {
        "shots": [{
            "shot_id": 1,
            "duration": 6,
            "interaction_geometry": {
                "effect_phase": "active",
                "interaction_mode": mode,
                "outcome_scope": scope,
                "effect_motion": "unspecified",
            },
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][0]["interaction_geometry"]["effect_motion"] == motion


def test_defaults_force_sweep_for_whole_directed_path_outcome():
    storyboard = {
        "shots": [{
            "shot_id": 1,
            "duration": 6,
            "interaction_geometry": {
                "effect_phase": "active",
                "interaction_mode": "directed_path",
                "outcome_scope": "all",
                "effect_motion": "static",
            },
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][0]["interaction_geometry"]["effect_motion"] == "sweep"


def test_defaults_reduce_short_multi_target_sweep_to_single_static():
    storyboard = {
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "interaction_geometry": {
                "effect_phase": "active",
                "interaction_mode": "directed_path",
                "outcome_scope": "subset",
                "effect_motion": "sweep",
            },
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    geometry = storyboard["shots"][0]["interaction_geometry"]
    assert geometry["outcome_scope"] == "single"
    assert geometry["effect_motion"] == "static"


def test_defaults_infer_active_motion_from_explicit_path_and_target():
    storyboard = {
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "interaction_geometry": {
                "effect_phase": "active",
                "actor": "actor",
                "target": "target",
                "line_of_action_visible": True,
                "effect_motion": "none",
            },
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    geometry = storyboard["shots"][0]["interaction_geometry"]
    assert geometry["interaction_mode"] == "directed_path"
    assert geometry["outcome_scope"] == "single"
    assert geometry["effect_motion"] == "static"
    assert geometry["reaction_scope"] == (
        "one clearly isolated intended target within the visible effect region"
    )


@pytest.mark.parametrize(
    ("geometry", "expected"),
    (
        (
            {
                "effect_phase": "resolution",
                "interaction_mode": "directed path",
                "outcome_scope": "some targets",
            },
            ("aftermath", "none", "subset", "none"),
        ),
        (
            {
                "effect_phase": "firing impact",
                "interaction_mode": "directed path",
                "outcome_scope": "some targets",
            },
            ("active", "directed_path", "subset", "static"),
        ),
    ),
)
def test_contract_compiler_derives_invariants_after_alias_normalization(
    geometry, expected
):
    storyboard = {
        "shots": [_shot(1, interaction_geometry=geometry)],
    }

    compiled = _compile_storyboard_contract(
        storyboard, "16:9", "480p", "cinematic"
    )

    result = compiled["shots"][0]["interaction_geometry"]
    assert (
        result["effect_phase"],
        result["interaction_mode"],
        result["outcome_scope"],
        result["effect_motion"],
    ) == expected


def test_defaults_merge_active_causal_entities_into_visibility_contract():
    storyboard = {
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "required_visible_entities": ["source"],
            "interaction_geometry": {
                "actor": "source",
                "target": "target_group",
                "effect_phase": "active",
                "interaction_mode": "directed_path",
                "outcome_scope": "subset",
                "effect_motion": "static",
            },
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][0]["required_visible_entities"] == [
        "source",
        "target_group",
    ]


def test_structured_active_effect_advances_conflict_without_keyword_matching():
    shot = _shot(
        1,
        primary_action="执行当前作用",
        interaction_geometry={
            "effect_phase": "active",
            "interaction_mode": "area_effect",
            "outcome_scope": "subset",
            "effect_motion": "static",
        },
    )

    assert _shot_advances_action_conflict(shot) is True


def test_defaults_canonicalize_interaction_targets_to_stable_entity_ids():
    storyboard = {
        "characters": [
            {"name": "subject", "reference_mode": "identity"},
            {"name": "hive_group", "reference_mode": "group"},
        ],
        "shots": [
            _shot(
                1,
                characters=["subject", "hive_group"],
                interaction_geometry={
                    "actor": "subject",
                    "target": "front hive",
                    "effect_phase": "active",
                    "interaction_mode": "area_effect",
                    "outcome_scope": "subset",
                    "effect_motion": "static",
                },
            ),
            _shot(
                2,
                characters=["subject", "hive_group"],
                interaction_geometry={
                    "actor": "subject",
                    "target": "remaining hive",
                    "effect_phase": "aftermath",
                    "interaction_mode": "none",
                    "outcome_scope": "subset",
                    "effect_motion": "none",
                },
            ),
        ],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert [
        shot["interaction_geometry"]["target"] for shot in storyboard["shots"]
    ] == ["hive_group", "hive_group"]


def test_defaults_compile_narrative_handoff_from_previous_result():
    storyboard = {
        "shots": [
            {
                "shot_id": 1,
                "duration": 5,
                "narrative_beat": {
                    "function": "setup",
                    "state_before": "the route is open",
                    "state_change": "a barrier appears",
                    "state_after": "the route is blocked",
                },
            },
            {
                "shot_id": 2,
                "duration": 5,
                "narrative_beat": {
                    "function": "progress",
                    "state_before": "an unrelated paraphrase",
                    "state_change": "the subject removes the barrier",
                    "state_after": "the route is open again",
                },
            },
        ],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][1]["narrative_beat"]["state_before"] == (
        "the route is blocked"
    )


def test_defaults_compile_screen_positions_from_character_blocking():
    blocking = {
        "robot": {
            "frame_position": "screen-left foreground",
            "body_orientation": "profile toward screen-right",
            "facing_target": "zombies",
            "eyeline_target": "zombies",
            "action_target": "zombies",
        },
        "zombies": {
            "frame_position": "screen-right midground",
            "body_orientation": "profile toward screen-left",
            "facing_target": "robot",
            "eyeline_target": "robot",
            "action_target": "robot",
        },
    }
    storyboard = {
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "scene_description": "Ruined street firefight",
            "prompt_en": "cinematic action detail " * 30,
            "characters": ["robot", "zombies"],
            "primary_action": "robot fires one burst at zombies",
            "action_beats": [{
                "phase": "peak",
                "actor": "robot",
                "action": "fires one burst",
                "target": "zombies",
                "visible_result": "the front zombie recoils",
            }],
            "camera": {
                "start_framing": "wide shot",
                "end_framing": "medium shot",
            },
            "blocking": blocking,
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")
    warnings, _ = _validate_storyboard_richness(
        storyboard,
        user_request="机器人在末日城市清除丧尸",
    )

    assert storyboard["shots"][0]["camera"]["screen_positions"] == {
        "robot": "screen-left foreground",
        "zombies": "screen-right midground",
    }
    assert not any("空间轴未定义" in warning for warning in warnings)


def test_defaults_compile_redundant_blocking_from_actor_target_positions():
    storyboard = {
        "characters": [
            {"name": "source", "reference_mode": "identity"},
            {"name": "receiver", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "scene_description": "Abstract interaction space",
            "prompt_en": "cinematic interaction " * 30,
            "primary_action": "source affects receiver",
            "characters": ["source", "receiver"],
            "camera": {
                "screen_positions": {
                    "source": "screen-left foreground",
                    "receiver": "screen-right background",
                }
            },
            "blocking": {
                "source": {"body_orientation": "front toward camera"},
                "receiver": {"body_orientation": "away from source"},
            },
            "action_beats": [{
                "phase": "peak",
                "actor": "source",
                "action": "applies one visible effect",
                "target": "receiver",
                "visible_result": "receiver visibly changes state",
            }],
            "interaction_geometry": {
                "actor": "source",
                "target": "receiver",
                "interaction_mode": "directed_path",
                "effect_phase": "active",
                "outcome_scope": "single",
                "effect_motion": "static",
            },
        }],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    source = storyboard["shots"][0]["blocking"]["source"]
    receiver = storyboard["shots"][0]["blocking"]["receiver"]
    assert source["body_orientation"] == (
        "three-quarter toward screen-right and background, away from camera"
    )
    assert source["facing_target"] == "receiver"
    assert source["eyeline_target"] == "receiver"
    assert source["action_target"] == "receiver"
    assert receiver["facing_target"] == "source"
    assert receiver["action_target"] == "source"
    assert not blocking_geometry_issues(storyboard["shots"][0])


def test_short_robot_action_replay_keeps_only_noncritical_scene_warning():
    characters = ["robot", "zombies"]
    blocking = {
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
    }
    actions = (
        "robot fires one burst at zombies",
        "zombies charge at the robot",
        "robot advances toward the lead zombie then crushes it",
    )
    storyboard = {
        "title": "robot apocalypse cleanup",
        "content_focus": "action",
        "characters": [
            {"name": "robot", "mobility": "tracked", "reference_mode": "identity"},
            {"name": "zombies", "mobility": "bipedal", "reference_mode": "group"},
        ],
        "shots": [
            _contract_shot(
                index,
                duration,
                action,
                scene_id="ruined_city_street",
                characters=characters,
                extract_character_ref=index == 1,
                camera={
                    "speed": "slow" if index == 3 else "fast",
                    "start_framing": framing,
                    "end_framing": framing,
                    "screen_positions": {"robot": "left", "zombies": "right"},
                    "axis_change": "establish" if index == 1 else "hold",
                },
                blocking=blocking,
                action_beats=[{
                    "phase": "peak",
                    "actor": "robot" if index != 2 else "zombies",
                    "action": action,
                    "target": "zombies" if index != 2 else "robot",
                    "visible_result": "the opposing side visibly recoils",
                }],
            )
            for index, duration, framing, action in zip(
                (1, 2, 3),
                (4, 5, 6),
                ("wide shot", "close-up", "medium shot"),
                actions,
            )
        ],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")
    warnings, is_critical = _validate_storyboard_richness(
        storyboard, user_request="制作一个15秒的视频，机器人末日清除丧尸"
    )

    assert not is_critical, warnings
    assert any("场景集中" in warning for warning in warnings)
    assert not any("动作重心不足" in warning for warning in warnings)
    assert not any("动作节拍过载" in warning for warning in warnings)
    assert not any("角色参考归属不明确" in warning for warning in warnings)


def test_explicit_compatible_seamless_continuation_is_preserved():
    storyboard = {
        "shots": [
            _shot(
                1,
                scene_id="intersection",
                camera={"start_framing": "wide shot", "end_framing": "wide shot"},
            ),
            _shot(
                2,
                scene_id="intersection",
                continuity_from_previous="seamless",
                camera={"start_framing": "wide shot", "end_framing": "medium shot"},
            ),
        ]
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][1]["continuity_from_previous"] == "seamless"


def test_incompatible_framing_forces_intentional_cut_in_same_scene():
    storyboard = {
        "shots": [
            _shot(
                1,
                scene_id="intersection",
                camera={
                    "start_framing": "medium shot",
                    "end_framing": "extreme wide shot",
                },
            ),
            _shot(
                2,
                scene_id="intersection",
                continuity_from_previous="seamless",
                camera={
                    "start_framing": "extreme close-up macro",
                    "end_framing": "extreme close-up macro",
                },
            ),
        ]
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][1]["continuity_from_previous"] == "intentional_cut"


def test_seamless_tail_frame_chain_is_bounded():
    storyboard = {
        "shots": [
            _shot(
                shot_id,
                scene_id="intersection",
                continuity_from_previous="seamless",
                camera={"start_framing": "wide shot", "end_framing": "wide shot"},
            )
            for shot_id in range(1, 5)
        ]
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert [s["continuity_from_previous"] for s in storyboard["shots"]] == [
        "none",
        "seamless",
        "seamless",
        "intentional_cut",
    ]


def test_extreme_close_up_character_reference_is_critical():
    storyboard = {
        "title": "robot",
        "shots": [
            _shot(
                1,
                characters=["hero"],
                extract_character_ref=True,
                camera={
                    "start_framing": "extreme close-up",
                    "end_framing": "close-up",
                },
                primary_action="hero activates one sensor",
                start_state={
                    "location": "intersection",
                    "subject": "hero hidden in smoke",
                    "action_phase": "inactive",
                    "camera": "extreme close-up",
                },
                end_state={
                    "location": "intersection",
                    "subject": "sensor visible",
                    "action_phase": "active",
                    "camera": "close-up",
                },
            ),
            _shot(
                2,
                characters=[],
                primary_action="dust settles",
                start_state={
                    "location": "intersection",
                    "subject": "empty street",
                    "action_phase": "dust moving",
                    "camera": "wide shot",
                },
                end_state={
                    "location": "intersection",
                    "subject": "empty street",
                    "action_phase": "dust settled",
                    "camera": "wide shot",
                },
            ),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(storyboard)

    assert is_critical
    assert any("角色参考镜头" in warning for warning in warnings)


def test_defaults_disable_unusable_closeup_character_reference():
    storyboard = {
        "characters": [{"name": "hero", "reference_mode": "identity"}],
        "shots": [
            _shot(
                1,
                characters=["hero"],
                extract_character_ref=True,
                camera={
                    "start_framing": "extreme close-up",
                    "end_framing": "close-up",
                },
                start_state={"camera": "extreme close-up"},
                end_state={"camera": "close-up"},
            )
        ],
    }

    _apply_defaults(storyboard, "16:9", "480p", "cinematic")

    assert storyboard["shots"][0]["extract_character_ref"] is False
    warnings, _ = _validate_storyboard_richness(storyboard)
    assert not any("角色参考镜头只有特写" in warning for warning in warnings)


def test_later_single_character_shot_can_extract_reference():
    storyboard = {
        "title": "robot",
        "shots": [
            _shot(1, characters=["hero"], extract_character_ref=False),
            _shot(2, characters=["hero"], extract_character_ref=True),
        ],
    }

    warnings, _ = _validate_storyboard_richness(storyboard)

    assert not any("角色首次出现" in warning for warning in warnings)


def test_fast_shot_over_five_seconds_is_critical():
    storyboard = {
        "title": "robot",
        "shots": [
            _shot(1, characters=[]),
            _shot(
                2,
                duration=8,
                camera={"speed": "fast", "start_framing": "close-up"},
            ),
        ],
    }

    warnings, is_critical = _validate_storyboard_richness(storyboard)

    assert is_critical
    assert any("fast" in warning and "5s" in warning for warning in warnings)


def test_missing_motion_contract_is_critical():
    storyboard = {
        "title": "robot",
        "shots": [_shot(1, characters=[]), _shot(2)],
    }

    warnings, is_critical = _validate_storyboard_richness(storyboard)

    assert is_critical
    assert any("动作契约不完整" in warning for warning in warnings)


def test_richness_does_not_duplicate_deterministic_blocking_geometry_warning():
    storyboard = {
        "content_focus": "action",
        "characters": [
            {"name": "actor", "reference_mode": "identity"},
            {"name": "target", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "prompt_en": "actor attacks target",
            "characters": ["actor", "target"],
            "primary_action": "actor attacks target",
            "camera": {"screen_positions": {
                "actor": "center foreground",
                "target": "center background",
            }},
            "blocking": {
                "actor": {"body_orientation": "front toward camera"},
            },
            "interaction_geometry": {"actor": "actor", "target": "target"},
        }],
    }

    warnings, _ = _validate_storyboard_richness(storyboard)

    assert sum("身体朝向与目标景深矛盾" in warning for warning in warnings) == 1


def test_correction_prompt_repairs_unregistered_nested_entities():
    prompt = _build_correction_prompt(
        "make an action storyboard",
        {"characters": [{"name": "actor"}]},
        ["🚨 Shot 1: action_beats.target 引用未注册实体 mystery_group"],
    )

    assert "actor 和角色 target 必须逐字复用顶层 characters.name" in prompt
