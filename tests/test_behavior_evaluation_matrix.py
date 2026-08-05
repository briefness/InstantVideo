"""Cross-theme offline behavior matrix; never calls paid providers."""

from pathlib import Path

import pytest

from pipeline.causality import compile_action_contract
from pipeline.generator import VideoGenerator
from pipeline.models import RunOptions
from pipeline.narrative import compile_narrative_carriers
from pipeline.production_plan import build_production_plan
from pipeline.provider_prompt import (
    compile_normal_provider_prompt,
    compile_policy_safe_prompt,
)
from pipeline.run_state import PaidTakeBudgetExhaustedError, RunWorkspace


EVALUATION_MATRIX_VERSION = "behavior-evaluation-v1"


@pytest.mark.parametrize(
    ("focus", "duration"),
    [
        ("action", 15),
        ("action", 30),
        ("product", 15),
        ("product", 30),
        ("balanced", 15),
        ("balanced", 30),
    ],
)
def test_plan_matrix_is_executable_across_themes(focus, duration):
    plan = build_production_plan(focus, duration)

    assert plan["version"] == "production-plan-v1"
    assert plan["planned_duration"] == sum(
        slot["duration"] for slot in plan["slots"]
    )
    assert all(4 <= slot["duration"] <= 15 for slot in plan["slots"])
    assert all(
        slot["narrative_function"] in {"setup", "progress", "turn", "payoff"}
        for slot in plan["slots"]
    )
    assert plan["slots"][0]["reference_policy"] == "independent"


@pytest.mark.parametrize(
    "mode",
    ["direct_contact", "directed_path", "area_effect", "indirect_effect"],
)
def test_action_contract_matrix_keeps_one_scope_owner(mode):
    shot = {
        "primary_action": "source performs one controlled action",
        "action_beats": [{
            "phase": "peak",
            "actor": "source",
            "action": "performs one controlled action",
            "target": "receiver",
            "visible_result": "receiver visibly changes state",
        }],
        "narrative_beat": {
            "function": "progress",
            "state_before": "the task is unresolved",
            "state_change": "the receiver visibly changes state",
            "state_after": "the task advances",
        },
        "interaction_geometry": {
            "actor": "source",
            "target": "receiver",
            "interaction_mode": mode,
            "effect_phase": "active",
            "outcome_scope": "single",
            "effect_motion": "static",
        },
    }

    contract = compile_action_contract(shot)

    assert contract.mode == mode
    assert contract.outcome_scope == "single"
    assert contract.contracted_visible_result == "receiver visibly changes state"
    assert compile_narrative_carriers(shot)


def test_prompt_profiles_share_contract_but_policy_safe_drops_freeform_prose():
    shot = {
        "scene_id": "generic_scene",
        "scene_description": "RAW_SCENE_PROSE",
        "prompt_en": "RAW_PROMPT_PROSE",
        "coverage_role": "interaction",
        "required_visible_entities": ["source", "receiver"],
        "narrative_beat": {
            "function": "progress",
            "state_before": "the task is unresolved",
            "state_change": "the receiver visibly changes state",
            "state_after": "the task advances",
        },
        "primary_action": "RAW_PRIMARY_ACTION",
        "action_beats": [{
            "phase": "peak",
            "actor": "source",
            "action": "performs one action",
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
    }
    storyboard = {"style": "cinematic", "mood": "focused", "characters": []}

    normal = compile_normal_provider_prompt(shot, storyboard, has_observed_start=False)
    safe = compile_policy_safe_prompt(
        shot,
        storyboard=storyboard,
        has_state_reference=False,
        image_role=None,
        reference_count=0,
    )

    assert "RAW_SCENE_PROSE" not in normal
    assert "RAW_PROMPT_PROSE" not in normal
    assert "RAW_SCENE_PROSE" not in safe
    assert "RAW_PROMPT_PROSE" not in safe
    assert "directed_path" in normal
    assert "directed_path" in safe


@pytest.mark.parametrize("characters", [["hero"], ["hero", "partner"]])
def test_character_cardinality_matrix_preserves_only_declared_identities(characters):
    shot = {
        "scene_id": "generic_scene",
        "characters": characters,
        "required_visible_entities": characters,
    }
    storyboard = {
        "style": "cinematic",
        "mood": "focused",
        "characters": [
            {"name": "hero", "description": "silver field suit"},
            {"name": "partner", "description": "dark utility coat"},
        ],
    }

    prompt = compile_policy_safe_prompt(
        shot,
        storyboard=storyboard,
        has_state_reference=False,
        image_role=None,
        reference_count=0,
    )

    assert "declared appearance for hero: silver field suit" in prompt
    assert ("declared appearance for partner: dark utility coat" in prompt) == (
        "partner" in characters
    )


def test_reference_matrix_separates_same_scene_state_and_cross_scene_identity(
    tmp_path: Path,
):
    identity = tmp_path / "hero.jpg"
    identity.write_bytes(b"fixture")
    generator = object.__new__(VideoGenerator)
    generator.character_refs = {"hero": str(identity)}

    previous = {"scene_id": "street", "output_reference_depth": 1}
    same_scene = {
        "scene_id": "street",
        "continuity_from_previous": "intentional_cut",
        "characters": ["hero"],
        "production_slot": {"reference_policy": "state_and_identity"},
    }
    cross_scene = {**same_scene, "scene_id": "rooftop"}

    assert generator._build_image_refs(same_scene, "/tmp/tail.jpg", previous) == (
        ["/tmp/tail.jpg"],
        "first_frame",
    )
    assert generator._build_image_refs(cross_scene, "/tmp/tail.jpg", previous) == (
        [str(identity)],
        "reference_image",
    )


def test_planned_reanchor_never_claims_reset_without_canonical_identity():
    generator = object.__new__(VideoGenerator)
    generator.character_refs = {}
    shot = {
        "scene_id": "street",
        "continuity_from_previous": "intentional_cut",
        "characters": ["hero"],
        "production_slot": {"reference_policy": "identity_only"},
    }
    previous = {"scene_id": "street", "output_reference_depth": 2}

    refs, role = generator._build_image_refs(shot, "/tmp/tail.jpg", previous)

    assert refs == ["/tmp/tail.jpg"]
    assert role == "first_frame"
    assert generator._next_reference_depth(shot, previous, "/tmp/tail.jpg") == 3


def test_actual_reference_role_is_the_reference_depth_authority():
    generator = object.__new__(VideoGenerator)
    generator.character_refs = {"hero": "/tmp/canonical.jpg"}
    shot = {
        "scene_id": "street",
        "characters": ["hero"],
        "production_slot": {"reference_policy": "identity_only"},
    }
    previous = {"scene_id": "street", "output_reference_depth": 2}

    assert generator._next_reference_depth(
        shot,
        previous,
        "/tmp/tail.jpg",
        reference_role="reference_image",
    ) == 0
    assert generator._next_reference_depth(
        shot,
        previous,
        "/tmp/tail.jpg",
        reference_role="first_frame",
    ) == 3


def test_paid_budget_matrix_counts_new_takes_but_not_recovery_polling(tmp_path: Path):
    workspace = RunWorkspace.create(
        tmp_path,
        RunOptions(request="matrix", paid_take_budget=1),
    )
    reservation = workspace.reserve_paid_take(1)
    workspace.confirm_paid_take_submission(reservation, "task-1")
    workspace.reconcile_paid_take(reservation)

    with pytest.raises(PaidTakeBudgetExhaustedError):
        workspace.reserve_paid_take(1)

    resumed = RunWorkspace.resume(workspace.path)
    assert (
        resumed.manifest.paid_take_budget.reservations[0].provider_task_id
        == "task-1"
    )
