"""Story-owned locations and one explicit declaration per prop, across topics."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.action_authoring import AuthoredShot, authored_shot_response_schema
from pipeline.authoring_context import SceneLocation, bind_authored_context, contextual_shot_schema
from test_action_results import annotated_story


def context():
    return [SceneLocation(id='workroom', description='The shared workroom')], ['Door', 'Dust', 'Parcel']


def response():
    raw = deepcopy(annotated_story()['shots'][0])
    raw.pop('shot_id')
    raw.pop('key_props')
    raw.pop('continuity_props', None)
    raw['scene_id'] = 'workroom'
    raw['prop_roles'] = [
        {'entity': 'Door', 'role': 'visible'},
        {'entity': 'Parcel', 'role': 'visible_and_persistent'},
        {'entity': 'Dust', 'role': 'visible'},
    ]
    return raw


def test_program_projects_prop_roles_without_promoting_decorations():
    locations, entities = context()
    raw = response()
    original = deepcopy(raw)
    shot = bind_authored_context(raw, locations, entities)
    assert shot['key_props'] == ['Door', 'Parcel', 'Dust']
    assert shot['continuity_props'] == ['Parcel']
    assert 'prop_roles' not in shot
    assert shot['action_plan'] == raw['action_plan'] and raw == original
    AuthoredShot.model_validate({'shot_id': 1, **shot})


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(scene_id='workroom_shot2'),
    lambda s: s.update(prop_roles=[{'entity': 'unknown', 'role': 'visible'}]),
    lambda s: s['prop_roles'].append(s['prop_roles'][0]),
    lambda s: s.update(key_props=['Door']),
    lambda s: s.update(continuity_props=[]),
    lambda s: s['prop_roles'][0].update(role='maybe'),
    lambda s: s.pop('prop_roles'),
])
def test_unknown_ids_conflicting_authorities_and_invalid_roles_stop(mutate):
    raw = response()
    mutate(raw)
    original = deepcopy(raw)
    with pytest.raises(ValueError):
        bind_authored_context(raw, *context())
    assert raw == original


def test_new_schema_and_local_decoder_use_the_same_ids_and_roles():
    import jsonschema
    locations, entities = context()
    base = authored_shot_response_schema(['attacker', 'defender'], require_outcomes=True,
                                          scene_entities=entities, bound_handoffs=True)
    before = deepcopy(base)
    schema = contextual_shot_schema(base, locations, entities)
    props = schema['properties']['shot']['properties']
    assert props['scene_id']['enum'] == ['workroom']
    assert 'key_props' not in props and 'continuity_props' not in props
    assert 'prop_roles' in schema['properties']['shot']['required']
    assert 'VisibleOutcomeSpec' not in schema['$defs'] and 'Effect' not in schema['$defs']
    jsonschema.validate({'shot': response()}, schema)
    assert base == before


@pytest.mark.parametrize('topic', ['partner dance', 'assembly transfer', 'sports exchange'])
def test_location_identity_survives_camera_changes_and_real_location_changes(topic):
    locations, entities = context()
    locations.append(SceneLocation(id='outside', description='The courtyard outside'))
    first = response()
    first['visual_description'] = topic
    second = deepcopy(first)
    second['camera']['primary_movement'] = 'dolly in'
    third = deepcopy(second)
    third['scene_id'] = 'outside'
    result = [bind_authored_context(s, locations, entities) for s in (first, second, third)]
    assert [s['scene_id'] for s in result] == ['workroom', 'workroom', 'outside']
    assert all(s['visual_description'] == topic for s in result)


def test_actual_partial_duplicate_response_remains_rejected():
    from pipeline.structured_response import ObjectKeyGuard, StructuredResponseError
    raw = json.loads((Path(__file__).parent / 'fixtures/real_shot_context_failure.json').read_text())
    for size in (1, 7, len(raw['response']['content'])):
        guard = ObjectKeyGuard()
        with pytest.raises(StructuredResponseError, match='重复字段'):
            for start in range(0, len(raw['response']['content']), size):
                guard.feed(raw['response']['content'][start:start + size])


def test_valid_legacy_prop_sets_round_trip_without_merging_their_meaning():
    locations, entities = context()
    legacy = deepcopy(annotated_story()['shots'][0])
    legacy.update(scene_id='workroom', key_props=['Door', 'Dust'], continuity_props=['Dust', 'Parcel'])
    wire = {k: deepcopy(v) for k, v in legacy.items() if k not in ('shot_id', 'key_props', 'continuity_props')}
    wire['prop_roles'] = [{'entity': 'Door', 'role': 'visible'},
                          {'entity': 'Dust', 'role': 'visible_and_persistent'},
                          {'entity': 'Parcel', 'role': 'persistent'}]
    assert {'shot_id': legacy['shot_id'], **bind_authored_context(wire, locations, entities)} == legacy


def test_formal_author_uses_frozen_locations_and_single_prop_declaration(monkeypatch):
    from pipeline.storyboard import generate_storyboard
    from test_director_intent import Responses, intent_reply
    raw = annotated_story()
    raw['scene_entities'] = ['Door', 'Dust']
    client = Responses({**intent_reply('continuous work'), 'choreography': 'flexible'}, raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    result = generate_storyboard('continuous work', target_duration=15)
    assert len(client.calls) == 5
    for call, shot in zip(client.calls[2:], result['shots']):
        user = json.loads(call['messages'][1]['content'])
        assert user['locations'] == raw['locations']
        assert user['authoring_protocol'] == 'story-context-v6-candidate-results'
        schema = call['response_format']['json_schema']['schema']['properties']['shot']
        assert schema['properties']['scene_id']['enum'] == ['training_roof']
        assert 'prop_roles' in schema['required']
        assert 'key_props' not in schema['properties'] and 'continuity_props' not in schema['properties']
        assert '"$defs"' not in call['messages'][0]['content']
        assert shot['scene_id'] == 'training_roof' and shot['key_props'] == ['Door', 'Dust']
        assert shot['continuity_props'] == []


@pytest.mark.parametrize('locations', [None, [], [{'id': 'x', 'description': 'a'}, {'id': 'x', 'description': 'b'}],
                                     [{'id': ' ', 'description': 'a'}], [{'id': 'x', 'description': ' '}]] )
def test_invalid_story_locations_stop_before_requesting_shots(monkeypatch, locations):
    from pipeline.storyboard import generate_storyboard
    from test_director_intent import Responses, intent_reply
    raw = annotated_story()
    raw.update(scene_entities=['Door', 'Dust'], locations=locations)
    client = Responses({**intent_reply('continuous work'), 'choreography': 'flexible'}, raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    with pytest.raises(ValueError, match='地点|locations'):
        generate_storyboard('continuous work', target_duration=15)
    assert len(client.calls) == 2


def test_empty_prop_catalog_and_independent_legacy_contract():
    locations, _ = context()
    raw = response()
    raw['prop_roles'] = []
    assert bind_authored_context(raw, locations, [])['continuity_props'] == []
    raw['prop_roles'] = [{'entity': 'anything', 'role': 'visible'}]
    with pytest.raises(ValueError, match='已登记'):
        bind_authored_context(raw, locations, [])
    legacy = authored_shot_response_schema(['A', 'B'])['properties']['shot']['properties']
    assert 'key_props' in legacy and 'continuity_props' in legacy and 'prop_roles' not in legacy


def test_context_protocol_in_actual_sdk_request_and_response(monkeypatch):
    from pipeline.storyboard import generate_storyboard
    from test_director_intent import intent_reply
    from test_authoring_transport import http_client
    from authoring_cases import contextual_shot_fixture, scheduled_shot_fixture
    from pipeline.production_plan import build_production_plan
    raw = annotated_story()
    raw['scene_entities'] = ['Door', 'Dust']
    replies = [{**intent_reply('continuous movement'), 'choreography': 'flexible'},
               {k: v for k, v in raw.items() if k != 'shots'}]
    plan = build_production_plan('action', 15, intent={**replies[0], 'source_request': 'continuous movement'})
    replies += [{'shot': {k: v for k, v in scheduled_shot_fixture(contextual_shot_fixture(s), slot).items()
                          if k != 'shot_id'}} for s, slot in zip(raw['shots'], plan['slots'])]
    sdk, calls = http_client(monkeypatch, replies)
    with sdk:
        result = generate_storyboard('continuous movement', target_duration=15)
    assert len(calls) == 5 and len(result['shots']) == 3
    assert 'locations' in calls[1]['response_format']['json_schema']['schema']['required']
    assert all(c['stream'] is True for c in calls[1:])
    assert all(c['response_format']['json_schema']['strict'] is True for c in calls)
    assert 'prop_roles' in calls[2]['response_format']['json_schema']['schema']['properties']['shot']['required']


@pytest.mark.parametrize('role', ['visible', 'persistent', 'visible_and_persistent'])
def test_role_capacity_matches_both_existing_projections(role):
    import jsonschema
    locations, _ = context()
    entities = [f'prop_{i}' for i in range(40)]
    raw = response()
    raw['prop_roles'] = [{'entity': name, 'role': role} for name in entities]
    schema = contextual_shot_schema(authored_shot_response_schema(['attacker', 'defender'],
        require_outcomes=True, bound_handoffs=True), locations, entities)
    jsonschema.validate({'shot': raw}, schema)
    bound = bind_authored_context(raw, locations, entities)
    AuthoredShot.model_validate({'shot_id': 1, **bound})
    raw['prop_roles'].append({'entity': 'prop_40', 'role': role})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'shot': raw}, schema)
    with pytest.raises(ValueError):
        bind_authored_context(raw, locations, [*entities, 'prop_40'])


def test_exact_authoring_prompt_keeps_legacy_checkpoint_input(monkeypatch):
    from pipeline.storyboard import generate_storyboard
    from test_director_intent import Responses, intent_reply, draft
    from authoring_cases import authoring_from_legacy
    client = Responses(intent_reply('continuous movement'), authoring_from_legacy(draft()))
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    generate_storyboard('continuous movement', target_duration=15)
    shot_call = client.calls[2]
    system = shot_call['messages'][0]['content']
    assert '内容符合下方schema' in system and '"$defs"' in system
    assert 'prop_roles' not in system and 'locations' not in json.loads(client.calls[1]['messages'][1]['content'])
