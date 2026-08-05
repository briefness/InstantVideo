"""Narrative Carrier Module tests."""

from pipeline.narrative import (
    compile_narrative_carriers,
    narrative_prompt_constraint,
    narrative_review_instruction,
)


def _narrative_shot(**overrides):
    shot = {
        "coverage_role": "target_reaction",
        "narrative_beat": {
            "function": "turn",
            "state_before": "the route appears safe",
            "state_change": "the blocked exit becomes visible",
            "state_after": "the subject must choose another route",
        },
    }
    shot.update(overrides)
    return shot


def test_narrative_carriers_are_derived_without_enlarging_storyboard_schema():
    carriers = compile_narrative_carriers(_narrative_shot())

    assert carriers == (
        {"kind": "visible_change", "value": "the blocked exit becomes visible"},
        {
            "kind": "readable_endpoint",
            "value": "the subject must choose another route",
        },
        {
            "kind": "coverage",
            "value": "keep the intended receiver's visible response readable",
        },
    )


def test_prompt_and_review_compile_the_same_narrative_carriers():
    shot = _narrative_shot()
    prompt = narrative_prompt_constraint(shot)
    review = narrative_review_instruction(shot)

    for carrier in compile_narrative_carriers(shot):
        rendered = f"{carrier['kind']}={carrier['value']}"
        assert rendered in prompt
        assert rendered in review


def test_insert_uses_declared_prop_as_signature_detail():
    carriers = compile_narrative_carriers(
        _narrative_shot(coverage_role="insert", key_props=["cracked compass"])
    )

    assert {"kind": "signature_detail", "value": "cracked compass"} in carriers


def test_utility_shot_without_narrative_beat_gets_no_invented_psychology():
    shot = {
        "coverage_role": "insert",
        "key_props": ["pressure gauge"],
        "primary_action": "the gauge needle rises",
    }

    assert compile_narrative_carriers(shot) == ()
    assert narrative_prompt_constraint(shot) == ""
    assert narrative_review_instruction(shot) == ""
