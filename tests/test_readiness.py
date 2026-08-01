"""Generation readiness gate tests."""

import pytest

from pipeline.readiness import (
    GenerationReadinessError,
    ensure_shot_ready,
    storyboard_readiness_issues,
)
from pipeline.participants import visible_character_names


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


def test_seamless_shot_without_accepted_tail_frame_fails_once():
    with pytest.raises(GenerationReadinessError, match="上一镜尾帧"):
        ensure_shot_ready(
            {"shot_id": 2, "continuity_from_previous": "seamless"},
            previous_frame=None,
            character_refs={},
        )


def test_same_scene_cut_without_accepted_tail_frame_fails_once():
    with pytest.raises(GenerationReadinessError, match="上一镜尾帧"):
        ensure_shot_ready(
            {
                "shot_id": 2,
                "scene_id": "street",
                "continuity_from_previous": "intentional_cut",
            },
            previous_frame=None,
            previous_shot={"shot_id": 1, "scene_id": "street"},
            character_refs={},
        )


def test_large_same_scene_cut_requires_tail_within_reference_budget():
    with pytest.raises(GenerationReadinessError, match="上一镜尾帧"):
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
