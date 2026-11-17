"""Frozen names constrain both provider output and the local authoring boundary."""
from copy import deepcopy

import pytest

from pipeline.action_authoring import authored_shot_response_schema
from pipeline.storyboard import generate_storyboard
from test_action_results import annotated_story
from test_director_intent import Responses, intent_reply


def test_result_and_target_ids_are_constrained_to_the_same_frozen_registry():
    schema = authored_shot_response_schema(['A', 'B'], require_outcomes=True, scene_entities=['Door', 'Dust'])
    expected = ['A', 'B', 'Door', 'Dust']
    assert schema['$defs']['SceneChangeSpec']['properties']['entity']['enum'] == expected
    for variant in schema['$defs']['InteractionUnit']['oneOf']:
        assert variant['properties']['target']['enum'] == expected
    assert schema['properties']['shot']['properties']['key_props']['items']['enum'] == expected


@pytest.mark.parametrize('where', ['target', 'scene_change', 'prop'])
def test_provider_ignoring_registry_stops_before_next_shot_without_alias_guessing(monkeypatch, where):
    from pipeline.authoring_contract import AuthoringContractError
    story = annotated_story()
    story['scene_entities'] = ['Door', 'Dust']
    shot = story['shots'][0]
    if where == 'target':
        shot['action_plan'][1]['target'] = 'Unregistered'
    elif where == 'scene_change':
        shot['action_plan'][1]['resolution']['scene_changes'][0]['entity'] = 'Unregistered'
    else:
        shot['key_props'].append('Unregistered')
    original = deepcopy(story)
    client = Responses({**intent_reply('15秒连续交互'), 'choreography': 'flexible'}, story)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    with pytest.raises(AuthoringContractError) as caught:
        generate_storyboard('15秒连续交互', target_duration=15)
    path, constraint = {
        'target': ('/shot/action_plan/interaction_1/target', 'enum'),
        'scene_change': ('/shot/action_plan/interaction_1/resolution/states', 'additionalProperties'),
        'prop': (f'/shot/prop_roles/{len(shot["key_props"])-1}/entity', 'enum'),
    }[where]
    assert {'path': path, 'constraint': constraint} in caught.value.issues
    assert len(client.calls) == 3 and story == original


@pytest.mark.parametrize('entities', [None, ['Door', 'Door'], ['attacker'], [' '], ['x' * 129]])
def test_invalid_frozen_registry_stops_before_shot_authoring(monkeypatch, entities):
    story = annotated_story()
    identity = {k: v for k, v in story.items() if k != 'shots'}
    identity['scene_entities'] = entities
    client = Responses({**intent_reply('15秒连续交互'), 'choreography': 'flexible'}, identity)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    with pytest.raises(ValueError):
        generate_storyboard('15秒连续交互', target_duration=15)
    assert len(client.calls) == 2


def test_actual_http_schema_uses_frozen_ids_and_derives_per_shot_registrations(monkeypatch):
    import json
    from test_authoring_transport import http_client
    from authoring_cases import scheduled_shot_fixture
    from pipeline.production_plan import build_production_plan
    story = annotated_story()
    story['scene_entities'] = ['Door', 'Dust']
    for shot in story['shots']:
        shot.pop('key_props', None)
        shot.pop('continuity_props', None)
        shot['prop_roles'] = []
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit['weight'] = 3.
    original = deepcopy(story)
    values = [{**intent_reply('15秒连续交互'), 'choreography': 'flexible'},
              {k: v for k, v in story.items() if k != 'shots'}]
    plan = build_production_plan('action', 15, intent={**values[0], 'source_request': '15秒连续交互'})
    values += [{'shot': {k: v for k, v in scheduled_shot_fixture(shot, slot, execution_timing=True).items() if k != 'shot_id'}}
               for shot, slot in zip(story['shots'], plan['slots'])]
    sdk, calls = http_client(monkeypatch, values)
    with sdk:
        result = generate_storyboard('15秒连续交互', target_duration=15)
    assert len(calls) == 5 and story == original
    assert 'scene_entities' in calls[1]['response_format']['json_schema']['schema']['required']
    for call, shot in zip(calls[2:], result['shots']):
        schema = call['response_format']['json_schema']['schema']
        states = schema['$defs']['EntityResolution']['properties']['states']
        assert list(states['properties']) == ['attacker', 'defender', 'Door', 'Dust']
        assert states['additionalProperties'] is False
        assert json.loads(call['messages'][1]['content'])['scene_entities'] == ['Door', 'Dust']
        assert shot['key_props'] == ['Door', 'Dust']
