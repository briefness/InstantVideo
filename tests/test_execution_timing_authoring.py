"""创作只决定时间关系；不把剧情回应当作运动来源。真实稿原样保留。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from pipeline.action_authoring import AuthoredShot, compile_action_units
from pipeline.authoring_schedule import bind_authored_schedule
from pipeline.production_plan import build_production_plan
from test_authoring_schedule import raw_shot, slot, schema_for, interaction


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline test attempted network'))


def recorded():
    return json.loads((Path(__file__).parent / 'fixtures/real_narrative_handoff_failure.json').read_text())


def test_original_failure_remains_exact_and_is_not_silently_migrated():
    record = recorded()
    assert hashlib.sha256(record['source_text'].encode()).hexdigest() == '7bd653f04d7f2328a609137c7e918e150c662b49e9539b24e0491d2fa0da1efa'
    raw = json.loads(record['source_text'])['shot']
    plan = build_production_plan('balanced', 24, intent=record['intent'])
    bound = bind_authored_schedule({'shot_id': 1, **deepcopy(raw)}, plan['slots'][0])
    from pipeline.authoring_context import bind_authored_context, SceneLocation
    bound = bind_authored_context(bound,
        [SceneLocation.model_validate(s) for s in record['identity']['locations']], record['identity']['scene_entities'])
    with pytest.raises(ValueError, match='共同参与者'):
        compile_action_units(AuthoredShot.model_validate(bound), plan['slots'][0])
    assert raw['action_plan']['following'][0]['link_previous'] is True


def test_same_recorded_performance_has_an_unambiguous_sequential_representation():
    from pipeline.authoring_timing import execution_timing_schema, bind_execution_timing
    from pipeline.action_authoring import authored_shot_response_schema
    from pipeline.authoring_context import contextual_shot_schema, bind_authored_context, SceneLocation
    from pipeline.authoring_schedule import scheduled_shot_schema
    record = recorded()
    original = json.loads(record['source_text'])
    annotation = deepcopy(original)
    unit = annotation['shot']['action_plan']['following'][0]
    # 独立的新协议标注：原稿明确 overlap=false，保留全部表演文字，不自动迁移运行稿。
    assert unit.pop('overlap') is False
    unit.pop('link_previous')
    unit['timing'] = 'sequential'
    identity = record['identity']
    locations = [SceneLocation.model_validate(s) for s in identity['locations']]
    current = build_production_plan('balanced', 24, intent=record['intent'])['slots'][0]
    base = authored_shot_response_schema([c['name'] for c in identity['characters']], require_outcomes=True,
        scene_entities=identity['scene_entities'], bound_handoffs=True, allow_scene_subjects=True)
    schema = execution_timing_schema(scheduled_shot_schema(
        contextual_shot_schema(base, locations, identity['scene_entities']), current))
    jsonschema.validate(annotation, schema)
    bound = bind_authored_context(bind_authored_schedule(
        bind_execution_timing({'shot_id': 1, **annotation['shot']}), current), locations, identity['scene_entities'])
    beats = compile_action_units(AuthoredShot.model_validate(bound), current)
    source_units = [original['shot']['action_plan']['first'], *original['shot']['action_plan']['following']]
    assert [(b['actor'], b['action'], b['visible_result']) for b in beats] == [
        (u['actor'], u['action'], u['result']) for u in source_units]
    assert all('flow_from_previous' not in b for b in beats)
    assert beats[-1]['end_seconds'] == current['duration']
    assert json.loads(record['source_text']) == original
    with pytest.raises(ValueError, match='旧承接'):
        bind_execution_timing(original['shot'])


@pytest.mark.parametrize('previous', [('attacker', 'Door'), ('defender', 'Dust'), ('attacker', 'Dust'), ('defender', 'Door')])
@pytest.mark.parametrize('current', [('attacker', 'Door'), ('defender', 'Dust'), ('attacker', 'Dust'), ('defender', 'Door')])
@pytest.mark.parametrize('timing', ['sequential', 'overlap_previous'])
def test_actor_permutations_change_only_real_overlap_feasibility(previous, current, timing):
    from pipeline.authoring_timing import execution_timing_schema, bind_execution_timing
    def unit(pair, mode):
        result = interaction()
        result.pop('link_previous')
        result.update(actor=pair[0], target=pair[1], timing=mode,
                      resolution={'states': {pair[1]: [{'state': 'the registered object has shifted'}]}})
        return result
    raw = raw_shot({'interaction_1': unit(previous, 'sequential'), 'interaction_2': unit(current, timing)})
    jsonschema.validate({'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}, execution_timing_schema(schema_for(slot())))
    bound = bind_authored_schedule(bind_execution_timing(raw), slot())
    if timing == 'overlap_previous' and not set(previous) & set(current):
        with pytest.raises(ValueError, match='共同参与者'):
            compile_action_units(AuthoredShot.model_validate(bound), slot())
    else:
        beats = compile_action_units(AuthoredShot.model_validate(bound), slot())
        assert (beats[1]['start_seconds'] < beats[0]['end_seconds']) == (timing == 'overlap_previous')
        assert beats[-1]['end_seconds'] == 5
        assert [b['action'] for b in beats] == [raw['action_plan'][key]['action'] for key in ('interaction_1', 'interaction_2')]


@pytest.mark.parametrize('extra', [{'link_previous': False}, {'overlap': False}, {'carry': ''},
                                 {'timing': True}, {'timing': 'respond'}, {'timing': None}])
def test_ambiguous_or_mixed_timing_input_is_rejected(extra):
    from pipeline.authoring_timing import bind_execution_timing
    unit = interaction()
    unit.pop('link_previous')
    unit.update(timing='sequential', **extra) if 'timing' not in extra else unit.update(extra)
    with pytest.raises(ValueError):
        bind_execution_timing(raw_shot({'interaction_1': unit}))


@pytest.mark.parametrize('opening', [False, True])
def test_overlap_cannot_bind_first_event_or_passive_predecessor(opening):
    from pipeline.authoring_timing import execution_timing_schema, bind_execution_timing
    from test_authoring_schedule import passive
    unit = interaction()
    unit.pop('link_previous')
    unit['timing'] = 'overlap_previous'
    schedule = {'interaction_1': unit}
    if opening:
        schedule['opening'] = {'first': passive()}
    raw = raw_shot(schedule)
    schema = execution_timing_schema(schema_for(slot(1)))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}, schema)
    with pytest.raises(ValueError, match='搭接'):
        bind_authored_schedule(bind_execution_timing(raw), slot(1))


def test_formal_context_contains_creation_not_execution_controls(monkeypatch):
    from pipeline.storyboard import generate_storyboard
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    from test_director_intent import Responses, intent_reply
    from test_action_results import annotated_story
    raw = annotated_story()
    raw['scene_entities'] = ['Door', 'Dust']
    for shot in raw['shots']:
        shot['action_plan'][2].update(link_previous=True, overlap=True)
    client = Responses({**intent_reply('连续的双人协作表演'), 'choreography': 'flexible'}, raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    saved = {}
    story = generate_storyboard('连续的双人协作表演', target_duration=15, allow_repair=False,
        save_diagnostic=lambda name, data: saved.update({name: deepcopy(data)}))
    assert len(client.calls) == 5
    for index, (call, shot, source) in enumerate(zip(client.calls[2:], story['shots'], raw['shots'])):
        context = json.loads(call['messages'][-1]['content'])
        assert context['authoring_protocol'] == 'story-context-v9-temporal-relations'
        serialized = json.dumps({'schema': call['response_format'], 'predecessor': context['previous_shot']})
        assert '"link_previous"' not in serialized and '"overlap"' not in serialized and '"carry"' not in serialized
        if index:
            assert context['previous_shot'] == {'shot_id': index, **saved[f'planning_shot_{index:03d}.json']['shot']}
        normal = compile_normal_provider_prompt(shot, story, has_observed_start=False)
        safe = compile_policy_safe_prompt(shot, storyboard=story, has_state_reference=False,
                                         image_role=None, reference_count=0)
        assert all(unit['action'] in normal for unit in source['action_plan'])
        assert 'Beat 2 -> Beat 3' in normal
        assert 'continuous momentum handoff' in safe
        assert shot['action_beats'][2]['start_seconds'] < shot['action_beats'][1]['end_seconds']
