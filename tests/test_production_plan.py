"""Deterministic production topology shared by planning and execution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.production_plan import (
    apply_production_plan,
    build_production_plan,
    classify_framing,
    production_plan_issues,
)
from pipeline.storyboard import _should_use_previous_tail_reference


def test_short_action_plan_reserves_execution_then_aftermath():
    plan = build_production_plan("action", 15)

    assert len(plan["slots"]) == 3
    assert sum(slot["duration"] for slot in plan["slots"]) == 15
    assert [slot["duration"] for slot in plan["slots"]] == [5, 5, 5]
    assert [slot["allowed_effect_phases"] for slot in plan["slots"]] == [
        ["active"], ["active"], ["aftermath"],
    ]
    assert [slot["outcome_scope"] for slot in plan["slots"]] == [
        "single", "single", "single",
    ]
    assert all(slot["requires_visible_result"] for slot in plan["slots"])
    assert [slot["coverage_roles"] for slot in plan["slots"]] == [
        ["interaction"], ["action_subject"], ["aftermath"],
    ]


def test_longer_action_plan_uses_setup_active_aftermath_topology():
    plan = build_production_plan("action", 30)
    slots = plan["slots"]

    assert len(slots) == 5
    assert sum(slot["duration"] for slot in slots) == 30
    assert all(4 <= slot["duration"] <= 15 for slot in slots)
    assert [slot["allowed_effect_phases"] for slot in slots] == [
        ["setup"], ["active"], ["active"], ["active"], ["aftermath"],
    ]
    assert [slot["requires_visible_result"] for slot in slots] == [
        False, True, True, True, True,
    ]
    assert [slot["outcome_scope"] for slot in slots] == [
        "none", "single", "single", "single", "single",
    ]
    assert all(len(slot["coverage_roles"]) == 1 for slot in slots)
    assert all(
        previous["framing_family"] != current["framing_family"]
        for previous, current in zip(slots, slots[1:])
    )


def test_balanced_plan_does_not_force_action_semantics():
    plan = build_production_plan("balanced", 15)

    assert all(
        slot["allowed_effect_phases"] == ["none", "setup", "active", "aftermath"]
        for slot in plan["slots"]
    )
    assert not any(slot["requires_visible_result"] for slot in plan["slots"])


def test_every_supported_request_duration_compiles_to_executable_slots():
    for target_duration in range(5, 121):
        plan = build_production_plan("action", target_duration)
        durations = [slot["duration"] for slot in plan["slots"]]

        assert all(4 <= duration <= 15 for duration in durations)
        assert plan["planned_duration"] == sum(durations)
        assert abs(plan["planned_duration"] - target_duration) <= 1
        phases = [slot["allowed_effect_phases"] for slot in plan["slots"]]
        if len(plan["slots"]) <= 2:
            assert phases == [["active"]] * len(plan["slots"])
        elif len(plan["slots"]) == 3:
            assert phases == [["active"], ["active"], ["aftermath"]]
        else:
            assert phases[0] == ["setup"]
            assert phases[-1] == ["aftermath"]
            assert all(phase == ["active"] for phase in phases[1:-1])
        expected_scopes = [
            "none" if phase == ["setup"] else "single"
            for phase in phases
        ]
        assert [slot["outcome_scope"] for slot in plan["slots"]] == expected_scopes
        assert all(len(slot["coverage_roles"]) == 1 for slot in plan["slots"])
        assert all(
            previous["framing_family"] != current["framing_family"]
            for previous, current in zip(plan["slots"], plan["slots"][1:])
        )


def test_framing_vocabulary_has_one_shared_classifier():
    assert classify_framing("close shot on the visible result") == "close_detail"
    assert classify_framing("detail insert") == "close_detail"
    assert classify_framing("medium wide shot") == "medium"
    assert classify_framing("establishing long shot") == "wide"


def test_plan_compiler_owns_duration_phase_and_reference_topology():
    plan = build_production_plan("action", 15)
    storyboard = {
        "content_focus": "action",
        "shots": [
            {
                "shot_id": 99,
                "duration": 3,
                "continuity_from_previous": "seamless",
                "composition_change": "large",
                "narrative_beat": {"function": "setup"},
                "interaction_geometry": {"effect_phase": "setup"},
            }
            for _ in range(3)
        ],
    }

    apply_production_plan(storyboard, plan)

    assert [shot["shot_id"] for shot in storyboard["shots"]] == [1, 2, 3]
    assert [shot["duration"] for shot in storyboard["shots"]] == [5, 5, 5]
    assert [
        shot["interaction_geometry"]["effect_phase"]
        for shot in storyboard["shots"]
    ] == ["active", "active", "aftermath"]
    assert storyboard["shots"][1]["continuity_from_previous"] == "intentional_cut"
    assert storyboard["shots"][1]["composition_change"] == "medium"
    assert storyboard["shots"][1]["camera"]["start_framing"] == "medium shot"
    assert all(
        shot["coverage_role"] == shot["production_slot"]["coverage_roles"][0]
        for shot in storyboard["shots"]
    )
    assert all(
        shot["interaction_geometry"]["outcome_scope"] == "single"
        for shot in storyboard["shots"]
    )
    topology_issues = production_plan_issues(storyboard)
    assert not any(
        marker in issue
        for issue in topology_issues
        for marker in (
            "duration", "effect_phase", "continuity_from_previous",
            "composition_change", "production_slot", "shot_id",
        )
    )


def test_plan_rejects_missing_or_mutated_slots_before_paid_generation():
    plan = build_production_plan("action", 30)
    storyboard = {
        "content_focus": "action",
        "production_plan": plan,
        "shots": [
            {
                "shot_id": slot["shot_id"],
                "duration": slot["duration"],
                "coverage_role": slot["coverage_roles"][0],
                "narrative_beat": {"function": slot["narrative_function"]},
                "interaction_geometry": {
                    "effect_phase": slot["allowed_effect_phases"][0]
                },
                "action_beats": [{"visible_result": "a visible state change"}],
                "production_slot": slot,
            }
            for slot in plan["slots"][:-1]
        ],
    }
    storyboard["shots"][1]["duration"] += 1

    issues = production_plan_issues(storyboard)

    assert any("镜头数量" in issue for issue in issues)
    assert any("duration" in issue for issue in issues)


def test_reference_selection_uses_the_compiled_slot_policy():
    previous = {"scene_id": "workshop", "output_reference_depth": 0}
    current = {
        "scene_id": "workshop",
        "continuity_from_previous": "intentional_cut",
        "composition_change": "medium",
        "characters": [],
        "production_slot": {"reference_policy": "state_and_identity"},
    }

    assert _should_use_previous_tail_reference(current, previous)

    current["characters"] = ["identity_subject"]
    assert _should_use_previous_tail_reference(current, previous)

    previous["output_reference_depth"] = 99
    assert _should_use_previous_tail_reference(current, previous)


def test_same_scene_plan_never_treats_identity_as_a_substitute_for_state():
    plan = build_production_plan("action", 30)

    assert plan["slots"][0]["reference_policy"] == "independent"
    assert [slot["reference_policy"] for slot in plan["slots"]] == [
        "independent", "state_and_identity", "state_and_identity",
        "identity_only", "state_and_identity",
    ]


def test_reanchor_policy_is_scheduled_only_inside_longer_sequences():
    short = build_production_plan("balanced", 15)
    long = build_production_plan("balanced", 60)

    assert "identity_only" not in [slot["reference_policy"] for slot in short["slots"]]
    assert [
        index + 1
        for index, slot in enumerate(long["slots"])
        if slot["reference_policy"] == "identity_only"
    ] == [4, 7]
