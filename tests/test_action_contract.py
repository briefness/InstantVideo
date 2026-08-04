"""Action Contract interface tests across prompting, review, and retakes."""

import json

import pytest

from pipeline.causality import (
    ACTION_EVIDENCE_FIELDS,
    causal_storyboard_issues,
    causality_readiness_issues,
    compile_action_contract,
)


def _evidence(**overrides):
    sample = {field: False for field in ACTION_EVIDENCE_FIELDS}
    sample.update(overrides)
    return sample


def test_setup_contract_distinguishes_non_physical_cue_from_effect_and_outcome():
    contract = compile_action_contract({
        "primary_action": "the device acquires its target",
        "interaction_geometry": {
            "effect_phase": "setup",
            "interaction_mode": "none",
            "outcome_scope": "none",
            "effect_motion": "none",
        },
        "end_state": {"action_phase": "target lock complete"},
    })

    assert contract.phase == "setup"
    assert contract.requires_evidence is True
    assert "non-physical cue" in contract.prompt_constraint
    assert "phase endpoint" in contract.review_instruction
    assert "narrative outcome" in contract.review_instruction


@pytest.mark.parametrize(("actor", "target", "preparation"), [
    ("calibration_unit", "test_marker", "aligns its optical sensor"),
    ("firefighter", "training_target", "raises the closed hose nozzle"),
    ("launch_vehicle", "launch_pad", "completes ignition checks"),
])
def test_setup_contract_owns_phase_safe_prompt_and_review_projection(
    actor, target, preparation
):
    shot = {
        "primary_action": preparation,
        "interaction_geometry": {
            "actor": actor,
            "target": target,
            "effect_phase": "setup",
            "interaction_mode": "none",
            "outcome_scope": "none",
            "effect_motion": "none",
            "unaffected_behavior": "TARGET_CONTINUES_PRIOR_MOTION",
        },
        "action_beats": [{
            "phase": "trigger",
            "actor": actor,
            "action": preparation,
            "target": target,
            "visible_result": "RAW_TARGET_RESULT",
        }],
        "narrative_beat": {
            "function": "setup",
            "state_before": "RAW_NARRATIVE_BEFORE",
            "state_change": "RAW_NARRATIVE_CHANGE",
            "state_after": "RAW_NARRATIVE_OUTCOME",
        },
        "end_state": {
            "subject": actor,
            "action_phase": "preparation complete",
            "pose_and_gaze": "ACTOR_READY_POSE",
            "prop_state": "SOURCE_LOCAL_CUE",
            "open_motion": "RAW_TARGET_OPEN_MOTION",
        },
    }

    contract = compile_action_contract(shot)
    provider_text = "; ".join(contract.prompt_parts)
    review_text = json.dumps(contract.review_projection, sort_keys=True)

    assert f"{actor} visibly prepares toward {target}" in provider_text
    assert "ACTOR_READY_POSE" in provider_text
    assert "TARGET_CONTINUES_PRIOR_MOTION" in provider_text
    for leaked in (
        "RAW_TARGET_RESULT",
        preparation,
        "RAW_NARRATIVE_CHANGE",
        "RAW_NARRATIVE_OUTCOME",
        "RAW_TARGET_OPEN_MOTION",
    ):
        assert leaked not in provider_text
        assert leaked not in review_text


def test_active_contract_preserves_contracted_result_and_narrative_outcome():
    shot = {
        "primary_action": "actor performs one effect",
        "interaction_geometry": {
            "actor": "actor",
            "target": "target",
            "effect_phase": "active",
            "interaction_mode": "directed_path",
            "outcome_scope": "single",
            "effect_motion": "static",
        },
        "action_beats": [{
            "phase": "peak",
            "actor": "actor",
            "action": "performs one effect",
            "target": "target",
            "visible_result": "CONTRACTED_TARGET_RESULT",
        }],
        "narrative_beat": {
            "function": "progress",
            "state_before": "before",
            "state_change": "CONTRACTED_NARRATIVE_CHANGE",
            "state_after": "CONTRACTED_NARRATIVE_OUTCOME",
        },
        "end_state": {"open_motion": "CONTRACTED_END_MOTION"},
    }

    contract = compile_action_contract(shot)
    provider_text = "; ".join(contract.prompt_parts)
    review_text = json.dumps(contract.review_projection, sort_keys=True)

    assert "CONTRACTED_TARGET_RESULT" in provider_text
    assert "only the contracted outcome and its permitted scope" in provider_text
    assert "only the permitted target scope" in review_text
    assert "CONTRACTED_TARGET_RESULT" in review_text
    for required in (
        "CONTRACTED_NARRATIVE_CHANGE",
        "CONTRACTED_NARRATIVE_OUTCOME",
        "CONTRACTED_END_MOTION",
    ):
        assert required not in provider_text
        assert required not in review_text


def test_single_scope_contract_owns_reaction_scope_for_prompt_and_review():
    shot = {
        "primary_action": "actor performs one directed action",
        "interaction_geometry": {
            "actor": "actor",
            "target": "target_group",
            "effect_phase": "active",
            "interaction_mode": "directed_path",
            "outcome_scope": "single",
            "effect_motion": "sweep",
            "reaction_scope": "RAW_THREE_TARGET_SCOPE",
            "unaffected_behavior": "RAW_GROUP_BEHAVIOR",
        },
    }

    contract = compile_action_contract(shot)
    provider_text = "; ".join(contract.prompt_parts)
    review_text = json.dumps(contract.review_projection, sort_keys=True)

    assert "one clearly isolated intended target within the visible effect region" in provider_text
    assert "all entities outside that one intended target continue their prior motion" in (
        provider_text
    )
    assert contract.review_projection["interaction_geometry"]["effect_motion"] == (
        "static"
    )
    for leaked in ("RAW_THREE_TARGET_SCOPE", "RAW_GROUP_BEHAVIOR"):
        assert leaked not in provider_text
        assert leaked not in review_text


def test_active_contract_uses_final_nonempty_visible_result_as_phase_endpoint():
    shot = {
        "production_slot": {"requires_visible_result": True},
        "interaction_geometry": {
            "effect_phase": "active",
            "interaction_mode": "directed_path",
        },
        "action_beats": [
            {"phase": "trigger", "visible_result": "TRIGGER_RESULT"},
            {"phase": "peak", "visible_result": "PEAK_RESULT"},
            {"phase": "aftermath", "visible_result": "FINAL_PHASE_ENDPOINT"},
        ],
    }
    contract = compile_action_contract(shot)
    provider_text = "; ".join(contract.prompt_parts)
    review_text = json.dumps(contract.review_projection, sort_keys=True)

    assert contract.contracted_visible_result == "FINAL_PHASE_ENDPOINT"
    assert "FINAL_PHASE_ENDPOINT" in provider_text
    assert "FINAL_PHASE_ENDPOINT" in review_text
    for intermediate in ("TRIGGER_RESULT", "PEAK_RESULT"):
        assert intermediate not in provider_text
        assert intermediate not in review_text


def test_setup_accepts_visible_preparation_endpoint_with_non_physical_cue():
    contract = compile_action_contract({
        "interaction_geometry": {"effect_phase": "setup"},
    })
    samples = [
        _evidence(),
        _evidence(
            preparation_state_visible=True,
            non_physical_cue_visible=True,
            phase_endpoint_visible=True,
        ),
    ]

    assert contract.evidence_issues(samples) == []


def test_setup_rejects_physical_effect_but_not_its_phase_endpoint():
    contract = compile_action_contract({
        "interaction_geometry": {"effect_phase": "setup"},
    })
    samples = [
        _evidence(
            preparation_state_visible=True,
            phase_endpoint_visible=True,
            physical_effect_visible=True,
        )
    ]

    assert contract.evidence_issues(samples) == ["准备阶段提前出现物理作用"]


def test_setup_does_not_treat_non_physical_targeting_cue_as_target_effect():
    contract = compile_action_contract({
        "interaction_geometry": {"effect_phase": "setup"},
    })
    samples = [
        _evidence(
            preparation_state_visible=True,
            non_physical_cue_visible=True,
            physical_effect_visible=False,
            effect_reaches_target=True,
            phase_endpoint_visible=True,
            narrative_outcome_visible=False,
            outcome_causally_connected=True,
        )
    ]

    assert contract.evidence_issues(samples) == []


def test_setup_still_rejects_observed_target_reaction_without_effect_path():
    contract = compile_action_contract({
        "interaction_geometry": {"effect_phase": "setup"},
    })
    samples = [
        _evidence(
            preparation_state_visible=True,
            phase_endpoint_visible=True,
            target_reaction_visible=True,
        )
    ]

    assert contract.evidence_issues(samples) == ["准备阶段提前作用于目标"]


def test_active_contract_rejects_out_of_scope_reaction_after_effect_path():
    contract = compile_action_contract({
        "interaction_geometry": {
            "effect_phase": "active",
            "interaction_mode": "directed_path",
            "outcome_scope": "single",
            "effect_motion": "static",
        },
    })
    samples = [
        _evidence(physical_effect_visible=True),
        _evidence(
            physical_effect_visible=True,
            effect_reaches_target=True,
            target_reaction_visible=True,
            out_of_scope_reaction_visible=True,
        ),
    ]

    issues = contract.evidence_issues(samples)

    assert "采样 2 出现范围外目标发生反应" in issues


def test_active_contract_accepts_causal_connection_completed_across_samples():
    contract = compile_action_contract({
        "interaction_geometry": {
            "effect_phase": "active",
            "interaction_mode": "directed_path",
            "outcome_scope": "subset",
            "effect_motion": "static",
        },
    })
    samples = [
        _evidence(
            physical_effect_visible=True,
            effect_reaches_target=True,
            target_reaction_visible=True,
            phase_endpoint_visible=True,
            outcome_causally_connected=True,
        ),
        _evidence(
            physical_effect_visible=True,
            effect_reaches_target=True,
            target_reaction_visible=True,
            phase_endpoint_visible=True,
            narrative_outcome_visible=True,
        ),
        _evidence(
            physical_effect_visible=True,
            effect_reaches_target=True,
            target_reaction_visible=True,
            phase_endpoint_visible=True,
            narrative_outcome_visible=True,
            outcome_causally_connected=True,
        ),
    ]

    assert contract.evidence_issues(samples) == []


def test_active_contract_allows_reaction_to_continue_after_prior_reach():
    contract = compile_action_contract({
        "interaction_geometry": {
            "effect_phase": "active",
            "interaction_mode": "directed_path",
            "outcome_scope": "subset",
            "effect_motion": "static",
        },
    })
    samples = [
        _evidence(physical_effect_visible=True),
        _evidence(
            physical_effect_visible=True,
            effect_reaches_target=True,
            target_reaction_visible=True,
            phase_endpoint_visible=True,
        ),
        _evidence(
            physical_effect_visible=True,
            target_reaction_visible=True,
            narrative_outcome_visible=True,
            outcome_causally_connected=True,
        ),
    ]

    assert contract.evidence_issues(samples) == []


def test_active_contract_rejects_reaction_before_any_reach():
    contract = compile_action_contract({
        "interaction_geometry": {
            "effect_phase": "active",
            "interaction_mode": "directed_path",
            "outcome_scope": "subset",
            "effect_motion": "static",
        },
    })

    issues = contract.evidence_issues([
        _evidence(physical_effect_visible=True, target_reaction_visible=True),
        _evidence(
            physical_effect_visible=True,
            effect_reaches_target=True,
            phase_endpoint_visible=True,
            narrative_outcome_visible=True,
            outcome_causally_connected=True,
        ),
    ])

    assert "采样 1 的反应目标未与作用区域相交" in issues
    assert "active 阶段的目标反应先于可见接触出现" in issues


@pytest.mark.parametrize("duration, expected", [(5, True), (6, False)])
def test_short_multi_target_directed_sweep_is_not_ready(duration, expected):
    issues = causality_readiness_issues({
        "shot_id": 1,
        "duration": duration,
        "interaction_geometry": {
            "effect_phase": "active",
            "interaction_mode": "directed_path",
            "outcome_scope": "subset",
            "effect_motion": "sweep",
            "source": "visible source",
            "effect_region": "visible directed path",
            "reaction_scope": "selected targets",
            "unaffected_behavior": "others remain unchanged",
            "line_of_action_visible": True,
        },
    })

    assert any("短镜头多目标定向扫掠过载" in issue for issue in issues) is expected


def test_evidence_prompt_excludes_contracted_independent_motion_from_reaction():
    contract = compile_action_contract({
        "interaction_geometry": {"effect_phase": "active"},
    })

    prompt = contract.evidence_prompt(9)

    assert "contracted unaffected behavior" in prompt
    assert "impact, injury, damage, forced displacement, or contracted outcome" in prompt
    assert "current sample" in prompt
    assert "continues after a prior reach" in prompt


@pytest.mark.parametrize("reaction_scope", ["", "none", "none yet", "n/a"])
def test_active_contract_requires_an_actual_reaction_scope(reaction_scope):
    issues = causal_storyboard_issues({
        "shots": [{
            "shot_id": 1,
            "interaction_geometry": {
                "effect_phase": "active",
                "interaction_mode": "indirect_effect",
                "outcome_scope": "all",
                "effect_motion": "propagate",
                "reaction_scope": reaction_scope,
            },
        }],
    }, required=True)

    assert any("reaction_scope" in issue for issue in issues)


def test_retake_instruction_is_compiled_from_the_same_action_contract():
    contract = compile_action_contract({
        "interaction_geometry": {
            "effect_phase": "active",
            "interaction_mode": "direct_contact",
            "outcome_scope": "single",
            "effect_motion": "static",
        },
    })

    instruction = contract.retake_instruction("target reaction was outside contact")

    assert "active direct_contact contract" in instruction
    assert "target reaction was outside contact" in instruction
    assert "do not add new actions" in instruction


@pytest.mark.parametrize(
    ("mode", "primary_action"),
    [
        ("direct_contact", "a hammer drives one nail into the joint"),
        ("directed_path", "a water stream reaches one marked plant"),
        ("area_effect", "a sprinkler wets plants inside one marked bed"),
        ("indirect_effect", "one falling tile tips the next tile"),
    ],
)
def test_active_evidence_contract_is_generic_across_physical_domains(
    mode,
    primary_action,
):
    contract = compile_action_contract({
        "primary_action": primary_action,
        "interaction_geometry": {
            "effect_phase": "active",
            "interaction_mode": mode,
            "outcome_scope": "single",
            "effect_motion": "static",
        },
    })
    samples = [
        _evidence(physical_effect_visible=True),
        _evidence(
            physical_effect_visible=True,
            effect_reaches_target=True,
            target_reaction_visible=True,
            phase_endpoint_visible=True,
            narrative_outcome_visible=True,
            outcome_causally_connected=True,
        ),
    ]

    assert contract.evidence_issues(samples) == []


def test_active_effect_without_target_consequence_does_not_pass():
    contract = compile_action_contract({
        "interaction_geometry": {
            "effect_phase": "active",
            "interaction_mode": "directed_path",
            "outcome_scope": "single",
            "effect_motion": "static",
        },
    })

    issues = contract.evidence_issues([
        _evidence(physical_effect_visible=True),
        _evidence(physical_effect_visible=True, phase_endpoint_visible=True),
    ])

    assert "生效阶段未看到物理作用到达目标" in issues
    assert "生效阶段未看到作用区域内目标的同步反应" in issues
    assert "active 阶段未看到约定结果（叙事结果）及其完整作用范围" in issues


def test_aftermath_requires_existing_outcome_without_new_effect():
    contract = compile_action_contract({
        "interaction_geometry": {
            "effect_phase": "aftermath",
            "interaction_mode": "none",
            "outcome_scope": "single",
            "effect_motion": "none",
        },
    })

    assert contract.evidence_issues([
        _evidence(
            phase_endpoint_visible=True,
            narrative_outcome_visible=True,
        )
    ]) == []
