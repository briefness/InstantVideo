"""Deterministic production topology shared by planning and execution."""

from pipeline.production_plan import (
    apply_production_plan,
    build_production_plan,
    classify_framing,
    production_plan_issues,
)
from pipeline.storyboard import _should_use_previous_tail_reference


def test_short_action_plan_reserves_execution_and_result_in_every_slot():
    plan = build_production_plan("action", 15)

    assert len(plan["slots"]) == 3
    assert sum(slot["duration"] for slot in plan["slots"]) == 15
    assert len({slot["duration"] for slot in plan["slots"]}) > 1
    assert all(slot["allowed_effect_phases"] == ["active"] for slot in plan["slots"])
    assert all(slot["requires_visible_result"] for slot in plan["slots"])


def test_longer_action_plan_keeps_editorial_breathing_room_around_active_core():
    plan = build_production_plan("action", 30)
    slots = plan["slots"]

    assert len(slots) == 5
    assert sum(slot["duration"] for slot in slots) == 30
    assert all(4 <= slot["duration"] <= 15 for slot in slots)
    assert slots[0]["allowed_effect_phases"] == ["setup", "active"]
    assert slots[-1]["allowed_effect_phases"] == ["active", "aftermath"]
    assert sum(slot["allowed_effect_phases"] == ["active"] for slot in slots) >= 3


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
        if len(durations) >= 3:
            assert len(set(durations)) > 1


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
    assert [shot["duration"] for shot in storyboard["shots"]] == [4, 6, 5]
    assert all(
        shot["interaction_geometry"]["effect_phase"] == "active"
        for shot in storyboard["shots"]
    )
    assert storyboard["shots"][1]["continuity_from_previous"] == "intentional_cut"
    assert storyboard["shots"][1]["composition_change"] == "small"
    assert storyboard["shots"][1]["camera"]["start_framing"] == "wide shot"
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
    assert all(
        slot["reference_policy"] in {"state_if_same_scene", "state_and_identity"}
        for slot in plan["slots"][1:]
    )
