"""Source requirements and filming suggestions have distinct consumers; no live services."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def captured():
    record = json.loads((Path(__file__).parent / 'fixtures/real_second_shot_protocol_failure.json').read_text())
    for source in record['sources'].values():
        assert hashlib.sha256(source['text'].encode()).hexdigest() == source['sha256']
    return record


def test_raw_latest_failure_at_sdk_boundary_preserves_source_and_has_no_retry(monkeypatch):
    import httpx
    from test_authoring_transport import http_client
    from pipeline.storyboard import generate_storyboard
    from pipeline.completion_intent import source_goals
    record = captured()
    values = {k: json.loads(v['text']) for k, v in record['sources'].items()}
    raw = values['planning_shot_002_response.json']['content']
    assert raw.endswith('"weight":1.2,"weight":')
    event = {'id': 'offline', 'object': 'chat.completion.chunk', 'created': 0,
             'model': 'offline', 'choices': [{'index': 0, 'delta': {'content': raw}, 'finish_reason': None}]}
    response = httpx.Response(200, headers={'content-type': 'text/event-stream'},
                             content='data: ' + json.dumps(event) + '\n\n')
    intent = json.loads(values['planning_intent_response.json']['content'])
    sdk, calls = http_client(monkeypatch, [intent, values['planning_identity.json'],
        json.loads(values['planning_shot_001_response.json']['content']), response])
    saved = {}
    with sdk, pytest.raises(ValueError, match='重复字段'):
        generate_storyboard(record['request'], style=record['style'], target_duration=24,
                            allow_repair=False, save_diagnostic=saved.__setitem__)
    assert len(calls) == 4
    assert saved['planning_shot_002_response.json']['content'] == raw
    assert saved['planning_shot_002_response.json']['finish_reason'] == 'protocol_error'
    request = saved['planning_shot_002_request.json']
    assert request['system'] == calls[-1]['messages'][0]['content']
    assert request['user'] == calls[-1]['messages'][1]['content']
    assert request['response_schema'] == calls[-1]['response_format']['json_schema']['schema']
    expected = source_goals({'production_plan': {'director_intent': {
        **intent, 'source_request': record['request'], 'authoring_protocol': 'scene-units-v1'}}})
    for call in calls[1:]:
        assert json.loads(call['messages'][1]['content'])['completion_goals'] == expected


@pytest.mark.parametrize('topic', ['family', 'ensemble', 'product'])
def test_all_authors_share_the_frozen_review_goal_catalog(monkeypatch, topic):
    from pipeline.storyboard import generate_storyboard
    from pipeline.completion_intent import source_goals
    from test_ordinary_scene_contract import scene_case
    from test_director_intent import Responses
    brief, reply, identity, wires, _ = scene_case(topic)
    client = Responses(reply, identity, *wires)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    story = generate_storyboard(brief, target_duration=24, allow_repair=False)
    goals = source_goals(story)
    assert goals
    for call in client.calls[1:]:
        context = json.loads(call['messages'][1]['content'])
        assert context['completion_goals'] == goals
        assert context['request'] == brief
    assert len(client.calls) == 6  # No new interpreter or repair call.


@pytest.mark.parametrize('field', ['composition', 'screen_positions'])
def test_camera_performance_prose_changes_generation_not_completion_authority(field):
    from test_goal_authority import admitted_story
    from pipeline.semantic_review import _review_contract
    from pipeline.provider_prompt import compile_normal_provider_prompt
    story = admitted_story()
    shot = story['shots'][0]
    before = deepcopy(_review_contract(shot))
    marker = 'The performer rests two fingers on the collar.'
    if field == 'composition':
        shot['camera'][field] += ' ' + marker
    else:
        actor = shot['characters'][0]
        shot['camera'][field][actor] += ' ' + marker
    assert _review_contract(shot) == before
    assert marker in compile_normal_provider_prompt(shot, story, has_observed_start=False)


def test_camera_execution_and_explicit_gestures_still_have_authority():
    from test_goal_authority import admitted_story
    from pipeline.semantic_review import _review_contract
    from pipeline.result_observations import decode_observations
    from test_goal_authority import annotated_snapshot
    from test_functional_results import format_for
    story = admitted_story()
    shot = story['shots'][0]
    before = deepcopy(_review_contract(shot))
    shot['camera']['primary_movement'] = 'tracking'
    after = _review_contract(shot)
    assert before != after and after['camera']['primary_movement'] == 'tracking'
    goal = 'The performer must rest two fingers on the collar.'
    reference = shot['action_beats'][0]['completion_intent']['reference']
    shot['action_beats'][0]['completion_intent']['goal'] = goal
    shot['completion_goals'][reference]['goal'] = goal
    catalog = _review_contract(shot)['completion_contract']
    assert catalog['criteria']['beat:1/completion']['action'] == goal
    wire = annotated_snapshot(shot)
    wire['observations']['beat:1/completion']['status'] = 'violated'
    assert not decode_observations(wire, catalog, format_for(shot)['json_schema']['schema'], 25)['accepted']


def test_legacy_camera_review_keeps_its_saved_contract():
    from test_sequence_review import process_shot
    from pipeline.semantic_review import _review_contract
    shot = process_shot('direct_contact')
    assert _review_contract(shot)['camera'] == shot.get('camera')
