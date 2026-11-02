"""Result ownership, independent required changes, and the original failed input.

New-protocol examples are explicit test annotations, not repaired model responses.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.action_authoring import AuthoredShot, authored_shot_response_schema, compile_action_units, compile_authored_storyboard
from pipeline.causality import compile_action_contract
from pipeline.models import ShotSpec
from authoring_cases import authoring_from_legacy
from test_director_intent import draft
from test_performance_outcomes import outcome_shot


def annotated_story():
    story = authoring_from_legacy(draft())
    story['locations'] = [{'id': 'training_roof', 'description': 'training roof'}]
    for shot in story['shots']:
        shot['key_props'] = ['Door', 'Dust']
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                # New-protocol fixture annotation, not a translation of captured model output.
                unit.pop('carry', None)
                unit['link_previous'] = False
                unit['resolution'] = {'target_state': 'remains able to continue', 'actor_state': None,
                    'scene_changes': [
                        {'entity': 'Door', 'state': 'open', 'importance': 'required'},
                        {'entity': 'Dust', 'state': 'briefly airborne', 'importance': 'incidental'}]}
    return story


def scene_shot():
    shot = valid_outcome_shot()
    shot['key_props'] = ['Door', 'Dust']
    shot['action_beats'][0]['scene_changes'] = annotated_story()['shots'][0]['action_plan'][1]['resolution']['scene_changes']
    return shot


def valid_outcome_shot():
    from test_action_authoring import plan_for
    shot = outcome_shot()
    shot['production_slot'] = {**plan_for()['slots'][0], **shot['production_slot']}
    return ShotSpec.model_validate(shot).model_dump(mode='json', exclude_none=True)


def test_original_failure_is_preserved_and_not_silently_reclassified():
    path = Path(__file__).parent / 'fixtures/real_outcome_ownership_failure.json'
    raw = json.loads(path.read_text())['shot']
    original = deepcopy(raw)
    # apply_patch adds a final newline; the captured response did not have one.
    assert hashlib.sha256(path.read_bytes().rstrip(b'\n')).hexdigest() == 'd2f4413e551768bd32e9be45e17bbc0333b6a1b3242422b68361f51e9a88bc0a'
    shot = AuthoredShot.model_validate(dict(raw, shot_id=1))
    with pytest.raises(ValueError, match='可见结果必须包含目标'):
        compile_action_units(shot, {'duration': 5, 'min_interaction_events': 2})
    assert raw == original


def test_new_schema_binds_primary_results_to_roles_instead_of_free_entity_names():
    schema = authored_shot_response_schema(['A', 'B'], require_outcomes=True)
    for variant in schema['$defs']['InteractionUnit']['oneOf']:
        assert 'outcomes' not in variant['properties']
        assert 'resolution' in variant['required']
    result = schema['$defs']['AuthoredResolution']
    assert set(result['required']) == {'target_state', 'actor_state', 'scene_changes'}
    assert result['additionalProperties'] is False
    assert 'entity' not in result['properties']


def test_compile_preserves_scene_changes_without_inventing_actions():
    story = annotated_story()
    original = deepcopy(story)
    slots = [{'shot_id': i, 'duration': 5, 'min_interaction_events': 2,
              'narrative_function': 'progress'} for i in range(1, 4)]
    result = compile_authored_storyboard(story, {'slots': slots})
    for raw, compiled in zip(story['shots'], result['shots']):
        assert len(raw['action_plan']) == len(compiled['action_beats'])
        for unit, beat in zip(raw['action_plan'], compiled['action_beats']):
            if unit['kind'] == 'interaction':
                assert beat['outcomes'] == [{'entity': unit['target'], 'state': 'remains able to continue'}]
                assert beat['scene_changes'] == unit['resolution']['scene_changes']
    assert story == original


def test_required_changes_reach_prompt_and_review_but_details_are_optional():
    shot = scene_shot()
    contract = compile_action_contract(shot)
    event = contract.review_projection['action_events'][0]
    assert {'entity': 'Door', 'state': 'open'} in event['outcomes']
    assert all(o['entity'] != 'Dust' for o in event['outcomes'])
    assert event['incidental_changes'] == [{'entity': 'Dust', 'state': 'briefly airborne'}]
    prompt = ' '.join(contract.prompt_parts)
    assert 'Door: open' in prompt and 'Dust: briefly airborne' in prompt
    assert 'Optional' in prompt


@pytest.mark.parametrize('change', [
    lambda s: s['action_beats'][0]['scene_changes'][0].update(entity='Unknown'),
    lambda s: s['action_beats'][0]['scene_changes'][0].update(entity=s['action_beats'][0]['target']),
    lambda s: s['action_beats'][0]['scene_changes'][0].update(importance='optional_maybe'),
    lambda s: s['action_beats'][0]['scene_changes'][0].update(state=' '),
    lambda s: s['production_slot'].update(choreography='exact'),
    lambda s: s['action_beats'][0].pop('outcomes'),
])
def test_invalid_scene_changes_cannot_enter_saved_or_direct_contracts(change):
    shot = scene_shot()
    change(shot)
    with pytest.raises(ValueError):
        ShotSpec.model_validate(shot)
    with pytest.raises(ValueError):
        compile_action_contract(shot)


def test_old_contract_fingerprint_survives_model_round_trip():
    shot = valid_outcome_shot()
    before = compile_action_contract(shot).fingerprint()
    assert compile_action_contract(ShotSpec.model_validate(shot).model_dump()).fingerprint() == before


@pytest.mark.parametrize('mutation', [
    lambda r: r.pop('target_state'),
    lambda r: r.update(target_state=' '),
    lambda r: r.update(target_state=True),
    lambda r: r.update(entity='someone_else'),
    lambda r: r.update(actor_state=12),
    lambda r: r['scene_changes'][0].update(importance=True),
    lambda r: r['scene_changes'][0].update(state='x' * 1601),
])
def test_role_bound_results_reject_malformed_or_conflicting_authoring(mutation):
    shot = annotated_story()['shots'][0]
    mutation(shot['action_plan'][1]['resolution'])
    with pytest.raises(ValueError):
        AuthoredShot.model_validate(shot)


def test_old_and_new_result_authorities_cannot_be_combined():
    shot = annotated_story()['shots'][0]
    unit = shot['action_plan'][1]
    unit['outcomes'] = [{'entity': unit['target'], 'state': 'contradictory state'}]
    with pytest.raises(ValueError, match='同时提供'):
        AuthoredShot.model_validate(shot)


def test_multiple_claims_are_preserved_instead_of_rejected_as_duplicate_owners():
    from pipeline.action_results import lower_authored_resolution
    raw = annotated_story()['shots'][0]
    unit = raw['action_plan'][1]
    unit['resolution']['scene_changes'].extend([
        deepcopy(unit['resolution']['scene_changes'][0]),
        {'entity': unit['target'], 'state': 'holds the parcel', 'importance': 'required'}])
    parsed = AuthoredShot.model_validate(raw).action_plan[1]
    lowered = lower_authored_resolution(parsed.actor, parsed.target, parsed.resolution)
    assert [c['state'] for c in lowered['outcomes']] == ['remains able to continue', 'holds the parcel']
    assert lowered['scene_changes'].count({'entity': 'Door', 'state': 'open', 'importance': 'required'}) == 2
    shot = scene_shot()
    shot['action_beats'][0]['scene_changes'].append(deepcopy(shot['action_beats'][0]['scene_changes'][0]))
    ShotSpec.model_validate(shot)
    assert compile_action_contract(shot).review_projection['action_events'][0]['outcomes'].count(
        {'entity': 'Door', 'state': 'open'}) == 2


def test_new_authoring_cannot_invent_an_unregistered_result_owner():
    story = annotated_story()
    story['shots'] = story['shots'][:1]
    story['shots'][0]['action_plan'][1]['resolution']['scene_changes'][0]['entity'] = 'Unknown'
    with pytest.raises(ValueError, match='未注册'):
        compile_authored_storyboard(story, {'slots': [{'shot_id': 1, 'duration': 5, 'narrative_function': 'progress'}]})


def test_explicit_new_protocol_annotation_preserves_every_state_from_real_failure():
    original = json.loads((Path(__file__).parent / 'fixtures/real_outcome_ownership_failure.json').read_text())['shot']
    annotated = deepcopy(original)
    annotated['shot_id'] = 1
    for unit in annotated['action_plan']:
        if unit['kind'] != 'interaction':
            continue
        states = {o['entity']: o['state'] for o in unit.pop('outcomes')}
        unit['resolution'] = {'target_state': states.pop(unit['target']),
            'actor_state': states.pop(unit['actor']), 'scene_changes': [
                {'entity': e, 'state': s, 'importance': 'required'} for e, s in states.items()]}
        # Explicit test annotation registers the formerly unregistered environment
        # name. Production never guesses aliases or adds registrations on failure.
        annotated['key_props'].extend(e for e in states if e not in annotated['key_props'])
    story = {'characters': [{'name': 'AzureSwordsman'}, {'name': 'CrimsonSwordsman'}], 'shots': [annotated]}
    result = compile_authored_storyboard(story, {'slots': [{'shot_id': 1, 'duration': 5, 'narrative_function': 'progress'}]})
    from pipeline.action_results import required_outcomes
    for source, beat in zip(original['action_plan'], result['shots'][0]['action_beats']):
        assert (beat['action'], beat['visible_result']) == (source['action'], source['result'])
        if source['kind'] == 'interaction':
            assert {o['entity']: o['state'] for o in required_outcomes(beat)} == {
                o['entity']: o['state'] for o in source['outcomes']}


def test_normal_and_safe_provider_prompts_share_required_and_optional_results():
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    shot = scene_shot()
    normal = compile_normal_provider_prompt(shot, {}, has_observed_start=False)
    safe = compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0)
    assert 'Door: open' in normal and 'Door: open' in safe
    assert 'Optional scene details (not required): Dust: briefly airborne' in normal
    # Moderation fallback deliberately omits optional prose; required states survive.
    assert 'Dust: briefly airborne' not in safe
