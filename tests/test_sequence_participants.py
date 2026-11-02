"""Nested event geometry uses the shared entity registry without new alias rules."""
from copy import deepcopy

import pytest

from pipeline.causality import compile_action_contract
from pipeline.participants import (
    iter_structured_entity_references, normalize_shot_participants,
    normalize_structured_entity_references, shot_entity_registry,
    structured_entity_reference_issues,
)
from test_action_sequence import sequence_story


@pytest.mark.parametrize("entry", ["declared_aliases", "structured_references"])
def test_event_and_nested_geometry_aliases_resolve_together(entry):
    shot = sequence_story()["shots"][0]
    aliases = {"attacker": "Attacker unit", "defender": "Defender unit"}
    shot["characters"] = list(aliases.values()) if entry == "declared_aliases" else list(aliases)
    for beat in shot["action_beats"][:2]:
        for field in ("actor", "target"):
            beat[field] = aliases[beat[field]]
            beat["interaction_geometry"][field] = aliases[beat["interaction_geometry"][field]]
    if entry == "declared_aliases":
        normalize_shot_participants(shot, aliases)
    else:
        normalize_structured_entity_references(shot, shot_entity_registry(shot, aliases))
    for beat in shot["action_beats"][:2]:
        assert beat["actor"] == beat["interaction_geometry"]["actor"]
        assert beat["target"] == beat["interaction_geometry"]["target"]
    assert not structured_entity_reference_issues(shot, aliases)
    events = compile_action_contract(shot).review_projection["action_events"]
    assert [(event["actor"], event["target"]) for event in events] == [
        ("attacker", "defender"), ("defender", "attacker")]


@pytest.mark.parametrize("field", ["actor", "target"])
def test_unknown_nested_event_participant_cannot_bypass_registry(field):
    shot = sequence_story()["shots"][0]
    shot["action_beats"][0]["interaction_geometry"][field] = "unregistered_xyz"
    registry = shot_entity_registry(shot, ["attacker", "defender"])
    normalize_structured_entity_references(shot, registry)
    issues = structured_entity_reference_issues(shot, registry.characters)
    assert any(f"action_beats.interaction_geometry.{field}" in issue and "unregistered_xyz" in issue
               for issue in issues)
    assert shot["action_beats"][0]["interaction_geometry"][field] == "unregistered_xyz"


def test_registered_prop_target_is_valid_and_event_roles_remain_independent():
    shot = sequence_story()["shots"][0]
    shot["key_props"] = ["crate"]
    beat = shot["action_beats"][0]
    beat["target"] = "crate"
    beat["interaction_geometry"]["target"] = "crate"
    before = deepcopy(shot["action_beats"])
    normalize_structured_entity_references(shot, shot_entity_registry(shot, ["attacker", "defender"]))
    assert shot["action_beats"] == before
    assert not structured_entity_reference_issues(shot, ["attacker", "defender"])
    nested = [ref for ref in iter_structured_entity_references(shot)
              if ref.field.startswith("action_beats.interaction_geometry.")]
    assert [(ref.value, ref.kind) for ref in nested] == [
        ("attacker", "character"), ("crate", "target"),
        ("defender", "character"), ("attacker", "target")]
