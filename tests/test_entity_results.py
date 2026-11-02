"""结果归属跟随稳定实体；真实错误保留，独立样本验证新接口。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline audit attempted network'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline audit attempted network'))


def captured():
    record = json.loads((Path(__file__).parent / 'fixtures/real_role_result_failure.json').read_text())
    for source in record['sources'].values():
        assert hashlib.sha256(source['text'].encode()).hexdigest() == source['sha256']
    return {name: json.loads(source['text']) for name, source in record['sources'].items()}


def test_original_role_reversal_remains_visible_and_rejected_without_retry(tmp_path, monkeypatch):
    from pipeline.action_authoring import compile_authored_storyboard
    from pipeline.storyboard import _compile_storyboard_contract
    from pipeline.planning_review import ensure_planning_semantics
    from test_director_intent import Responses
    data = captured()
    raw, saved = data['planning_authoring.json'], data['storyboard.json']
    before = deepcopy(raw)
    plan = saved['production_plan']
    compiled = _compile_storyboard_contract(compile_authored_storyboard(raw, plan),
        saved['aspect_ratio'], saved['resolution'], saved['style'], production_plan=plan)
    bad = compiled['shots'][1]['action_beats'][2]
    original = raw['shots'][1]['action_plan'][2]
    assert bad['outcomes'][0] == {'entity': original['target'], 'state': original['resolution']['target_state']}
    assert bad['outcomes'][1] == {'entity': original['actor'], 'state': original['resolution']['actor_state']}
    assert bad['outcomes'] == saved['shots'][1]['action_beats'][2]['outcomes']
    assert raw == before
    client = Responses(data['planning_semantic_review.json']['verdict'])
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: client)
    def save(name, value):
        (tmp_path / name).write_text(json.dumps(value))
    for _ in range(2):
        with pytest.raises(ValueError, match='结果合同无效'):
            ensure_planning_semantics(saved, tmp_path, save)
    assert len(client.calls) == 1


def test_new_wire_uses_frozen_entity_keys_without_relative_result_slots():
    from test_authoring_schedule import schema_for, slot
    from pipeline.llm_options import ark_output_schema
    for schema in (schema_for(slot()), ark_output_schema(schema_for(slot()))):
        defs = schema['$defs']
        assert 'AuthoredResolution' not in defs
        states = defs['EntityResolution']['properties']['states']
        assert set(states['properties']) == {'attacker', 'defender', 'Door', 'Dust'}
        assert states['additionalProperties'] is False
        assert set(defs['EntityResolution']['properties']) == {'states'}


@pytest.mark.parametrize('actor,target', [('A', 'B'), ('B', 'A'), ('A', 'Panel')])
def test_owners_do_not_move_when_action_roles_or_state_order_change(actor, target):
    from pipeline.action_results import EntityResolution, lower_authored_resolution
    states = {
        'A': [{'state': 'holds the supporting handle', 'importance': 'required'}],
        'B': [{'state': 'stands with support', 'importance': 'required'},
              {'state': 'coat moves in the breeze', 'importance': 'incidental'}],
        'Panel': [{'state': 'indicator illuminated', 'importance': 'required'}],
    }
    expected = sorted((owner, c['state'], c['importance']) for owner, claims in states.items() for c in claims)
    for ordered in (states, dict(reversed(list(states.items())))):
        before = deepcopy(ordered)
        parsed = EntityResolution.model_validate({'states': ordered})
        lowered = lower_authored_resolution(actor, target, parsed)
        actual = [(c['entity'], c['state'], c.get('importance', 'required'))
                  for c in lowered['outcomes'] + lowered['scene_changes']]
        assert sorted(actual) == expected and ordered == before
        assert len(lowered['source_map']) == len(expected)


def test_source_paths_escape_entity_keys_and_preserve_repeated_conflicting_claims():
    from pipeline.action_results import EntityResolution, lower_authored_resolution
    raw = {'states': {'B/part~1': [
        {'state': 'open', 'importance': 'required'},
        {'state': 'closed', 'importance': 'required'},
        {'state': 'open', 'importance': 'required'}]}}
    lowered = lower_authored_resolution('A', 'B/part~1', EntityResolution.model_validate(raw))
    assert [c['state'] for c in lowered['outcomes']] == ['open', 'closed', 'open']
    assert [s['source'] for s in lowered['source_map']] == [
        f'resolution/states/B~1part~01/{i}' for i in range(3)]


@pytest.mark.parametrize('raw', [
    {'states': {}}, {'states': {'B': []}},
    {'states': {'B': [{'state': '', 'importance': 'required'}]}},
    {'states': {'B': [{'state': 'ready', 'importance': 'ignore'}]}},
    {'states': {'B': [{'state': 'ready', 'importance': 'required'}] * 17}},
    {'states': {'B': [{'state': 'ready', 'importance': 'required'}]}, 'actor_state': None},
])
def test_entity_result_boundaries_reject_invalid_shape_without_rewriting(raw):
    from pipeline.action_results import EntityResolution
    before = deepcopy(raw)
    with pytest.raises(ValueError):
        EntityResolution.model_validate(raw)
    assert raw == before


def test_existing_target_and_scene_capacity_requirements_still_apply():
    from pipeline.action_results import EntityResolution, lower_authored_resolution
    missing = EntityResolution.model_validate({'states': {'A': [{'state': 'ready', 'importance': 'required'}]}})
    with pytest.raises(ValueError, match='目标'):
        lower_authored_resolution('A', 'B', missing)
    too_many = EntityResolution.model_validate({'states': {
        'B': [{'state': 'ready', 'importance': 'required'}],
        'Object': [{'state': 'unchanged', 'importance': 'incidental'}] * 15}})
    with pytest.raises(ValueError, match='14'):
        lower_authored_resolution('A', 'B', too_many)


def test_wire_and_local_binding_reject_old_or_unknown_ownership():
    from test_authoring_schedule import schema_for, slot, raw_shot
    from pipeline.authoring_schedule import bind_authored_schedule
    from pipeline.storyboard_authoring import _bind_scene_entities
    raw = raw_shot()
    raw['action_plan']['interaction_1']['resolution'] = {'states': {
        'Unknown': [{'state': 'ready', 'importance': 'required'}],
        'defender': [{'state': 'balanced', 'importance': 'required'}]}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}, schema_for(slot()))
    bound = bind_authored_schedule(raw, slot())
    with pytest.raises(ValueError, match='未注册'):
        _bind_scene_entities(bound, ['attacker', 'defender'], ['Door', 'Dust'])
    raw['action_plan']['interaction_1']['resolution'] = {
        'target_state': 'ready', 'actor_state': None, 'scene_changes': []}
    with pytest.raises(ValueError):
        bind_authored_schedule(raw, slot())


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_whole_film_allocation_preserves_entity_results_and_all_actions(topic):
    from test_creation_schedule_flow import scenario_story
    from authoring_cases import entity_resolution_fixture
    from pipeline.action_authoring import compile_authored_storyboard
    from pipeline.film_rhythm import planned_interaction_seconds
    from pipeline.action_results import required_outcomes
    _, _, raw, plan = scenario_story(topic)
    for shot in raw['shots']:
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit['resolution'] = entity_resolution_fixture(unit)
                unit['weight'] = .01
    before = deepcopy(raw)
    result = compile_authored_storyboard(raw, plan)
    assert raw == before and planned_interaction_seconds(result['shots']) >= 10.5 - 1e-6
    for source, compiled in zip(raw['shots'], result['shots']):
        assert len(source['action_plan']) == len(compiled['action_beats'])
        for unit, beat in zip(source['action_plan'], compiled['action_beats']):
            assert (unit['action'], unit['result']) == (beat['action'], beat['visible_result'])
            if unit['kind'] == 'interaction':
                assert sorted((owner, c['state']) for owner, claims in unit['resolution']['states'].items()
                              for c in claims if c['importance'] == 'required') == sorted(
                    (c['entity'], c['state']) for c in required_outcomes(beat))


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_role_switching_through_real_sdk_keeps_entity_results_and_requests_once(monkeypatch, topic):
    from test_creation_schedule_flow import scenario_story
    from test_authoring_transport import http_client
    from authoring_cases import contextual_shot_fixture, scheduled_shot_fixture
    from pipeline.storyboard import generate_storyboard
    from pipeline.action_results import required_outcomes
    brief, intent, raw, plan = scenario_story(topic)
    expected = []
    for shot in raw['shots']:
        for i, unit in enumerate(shot['action_plan']):
            if unit['kind'] == 'interaction':
                unit['actor'], unit['target'] = ('Lead', 'Partner') if i % 2 else ('Partner', 'Lead')
                unit['resolution'] = {'states': {
                    'Partner': [{'state': 'keeps the shared connection', 'importance': 'required'}],
                    'Lead': [{'state': 'stands balanced on the supporting foot', 'importance': 'required'}]}}
                expected.append((unit['actor'], unit['target'], deepcopy(unit['resolution']['states'])))
    before = deepcopy(raw)
    values = [{k: v for k, v in intent.items() if k != 'source_request'},
              {k: v for k, v in raw.items() if k != 'shots'}]
    values += [{'shot': {k: v for k, v in scheduled_shot_fixture(contextual_shot_fixture(shot), slot).items()
                         if k != 'shot_id'}} for shot, slot in zip(raw['shots'], plan['slots'])]
    sdk, calls = http_client(monkeypatch, values)
    with sdk:
        story = generate_storyboard(brief, target_duration=15)
    actual = [beat for shot in story['shots'] for beat in shot['action_beats'] if beat['activity'] == 'interaction']
    for beat, (actor, target, states) in zip(actual, expected, strict=True):
        assert (beat['actor'], beat['target']) == (actor, target)
        assert [(c['entity'], c['state']) for c in required_outcomes(beat)] == [
            (entity, c['state']) for entity, claims in states.items() for c in claims]
    assert raw == before and len(calls) == 5
    for call in calls[2:]:
        context = json.loads(call['messages'][-1]['content'])
        assert context['authoring_protocol'] == 'story-context-v6-candidate-results'
        assert 'AuthoredResolution' not in call['response_format']['json_schema']['schema']['$defs']
