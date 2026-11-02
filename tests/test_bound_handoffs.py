"""Bind continuity to an existing predecessor, not another authored cause."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.action_authoring import AuthoredShot, authored_shot_response_schema, compile_action_units
from pipeline.causality import compile_action_contract
from pipeline.storyboard import generate_storyboard
from test_action_results import annotated_story
from test_action_authoring import plan_for
from test_director_intent import Responses, intent_reply


def linked_shot():
    shot = annotated_story()['shots'][0]
    for unit in shot['action_plan']:
        if unit['kind'] == 'interaction':
            unit.pop('carry', None)
            unit['link_previous'] = True
    return shot


def test_new_wire_contract_cannot_author_a_second_cause():
    schema = authored_shot_response_schema(['attacker', 'defender'], require_outcomes=True, bound_handoffs=True)
    for unit in schema['$defs']['InteractionUnit']['oneOf']:
        assert 'carry' not in unit['properties']
        assert unit['properties']['link_previous'] == {'type': 'boolean'}
        assert 'link_previous' in unit['required']
    first = schema['properties']['shot']['properties']['action_plan']['prefixItems'][0]
    for unit in first['oneOf'][:-1]:
        assert unit['properties']['link_previous']['const'] is False


@pytest.mark.parametrize('topic', ['sword exchange', 'partner dance', 'assembly transfer'])
def test_forward_link_preserves_actions_and_binds_only_predecessor(topic):
    raw = linked_shot()
    raw['action_plan'][1]['action'] = f'{topic}: first continuous movement'
    raw['action_plan'][2]['action'] = f'{topic}: use the preceding movement to change direction'
    original = deepcopy(raw)
    beats = compile_action_units(AuthoredShot.model_validate(raw), plan_for()['slots'][0])
    assert 'Beat 1 -> Beat 2' in beats[1]['flow_from_previous']
    assert 'Beat 2 -> Beat 3' in beats[2]['flow_from_previous']
    assert [b['action'] for b in beats] == [u['action'] for u in raw['action_plan']]
    assert raw == original


@pytest.mark.parametrize('mutation', ['both', 'first', 'overlap_without_link', 'wrong_type'])
def test_invalid_links_fail_without_inventing_a_cause(mutation):
    raw = linked_shot()
    if mutation == 'both':
        raw['action_plan'][1]['carry'] = 'after this action, the next move begins'
    elif mutation == 'first':
        raw['action_plan'].pop(0)
    elif mutation == 'overlap_without_link':
        raw['action_plan'][2].update(link_previous=False, overlap=True)
    else:
        raw['action_plan'][1]['link_previous'] = 'true'
    with pytest.raises(ValueError):
        compile_action_units(AuthoredShot.model_validate(raw), plan_for()['slots'][0])


def test_legacy_carry_is_preserved_without_reinterpretation():
    raw = annotated_story()['shots'][0]
    unit = raw['action_plan'][1]
    unit.pop('link_previous', None)
    unit['carry'] = 'after this move, the actor prepares the next move'
    beats = compile_action_units(AuthoredShot.model_validate(raw), plan_for()['slots'][0])
    assert beats[1]['flow_from_previous'] == unit['carry']


def test_typed_link_round_trip_does_not_invent_a_second_authority():
    shot = AuthoredShot.model_validate(linked_shot())
    assert AuthoredShot.model_validate(shot.model_dump()) == shot


def test_formal_new_plan_uses_links_through_provider_contract(monkeypatch):
    raw = annotated_story()
    for shot in raw['shots']:
        for index, unit in enumerate(shot['action_plan']):
            if unit['kind'] == 'interaction':
                unit.pop('carry', None)
                unit['link_previous'] = index > 0
    raw['scene_entities'] = ['Door', 'Dust']
    brief = '15秒连续动作展示'
    client = Responses({**intent_reply(brief), 'choreography': 'flexible'}, raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    story = generate_storyboard(brief, target_duration=15)
    assert len(client.calls) == 5
    for shot in story['shots']:
        contract = compile_action_contract(shot)
        assert 'Beat 2 -> Beat 3' in ' '.join(contract.prompt_parts)
    schema = client.calls[2]['response_format']['json_schema']['schema']
    assert 'link_previous' in json.dumps(schema)
    assert '"carry"' not in json.dumps(schema)


def test_captured_outgoing_carry_is_not_an_executable_new_protocol_cause(monkeypatch):
    from jsonschema import validate, ValidationError
    from pipeline.action_authoring import InteractionUnit
    record = json.loads((Path(__file__).parent / 'fixtures/real_outgoing_carry.json').read_text())
    original = deepcopy(record)
    schema = authored_shot_response_schema(['Lingfeng', 'Zhenwei'], require_outcomes=True, bound_handoffs=True)
    variant = next(v for v in schema['$defs']['InteractionUnit']['oneOf']
                   if v['properties']['response']['const'] == 'deflection')
    # Historic imports preserve their text; no keyword detector relabels their meaning.
    assert InteractionUnit.model_validate(record['unit']).carry == record['unit']['carry']
    with pytest.raises(ValidationError):
        validate(record['unit'], {**variant, '$defs': schema['$defs']})
    raw = annotated_story()
    raw['scene_entities'] = ['Door', 'Dust']
    raw['shots'][0]['action_plan'][1].pop('link_previous')
    raw['shots'][0]['action_plan'][1]['carry'] = record['unit']['carry']
    client = Responses({**intent_reply('offline'), 'choreography': 'flexible'}, raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    with pytest.raises(ValueError, match='link_previous'):
        generate_storyboard('offline', target_duration=15)
    assert len(client.calls) == 3  # Intent, identity, first shot; no retry or next shot.
    assert record == original


def test_independent_action_and_cross_shot_entry_keep_their_own_meaning():
    from pipeline.action_authoring import compile_authored_storyboard
    raw = annotated_story()
    for shot in raw['shots']:
        shot['action_plan'].pop(0)
        shot['start_state']['open_motion'] = 'already rotating from the preceding shot'
    result = compile_authored_storyboard(raw, plan_for())
    for shot in result['shots']:
        assert shot['start_state']['open_motion'] == 'already rotating from the preceding shot'
        assert all('flow_from_previous' not in beat for beat in shot['action_beats'])


def test_linked_overlap_keeps_time_union_and_forward_order():
    raw = linked_shot()
    raw['action_plan'][2]['overlap'] = True
    beats = compile_action_units(AuthoredShot.model_validate(raw), plan_for()['slots'][0])
    assert beats[2]['start_seconds'] < beats[1]['end_seconds']
    assert beats[2]['effect_at_seconds'] > beats[1]['effect_at_seconds']
    assert beats[-1]['end_seconds'] == 5
