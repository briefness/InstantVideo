"""Captured omission plus independent service annotations; no model quality claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from pipeline.action_authoring import AuthoredShot, authored_shot_response_schema, compile_action_units
from pipeline.authoring_context import SceneLocation, contextual_shot_schema, bind_authored_context
from pipeline.production_plan import build_production_plan


@pytest.fixture(autouse=True)
def no_external_connections(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline replay attempted a connection'))


def captured_failure():
    record = json.loads((Path(__file__).parent / 'fixtures/real_action_allocation_failure.json').read_text())
    for source in record['sources'].values():
        assert hashlib.sha256(source['text'].encode()).hexdigest() == source['sha256']
    identity = json.loads(record['sources']['planning_identity.json']['text'])
    response = json.loads(record['sources']['planning_shot_001_response.json']['text'])
    assert response['finish_reason'] == 'stop'
    return record, identity, json.loads(response['content'])


def test_original_failure_preserves_invalid_target_and_missing_work():
    record, identity, response = captured_failure()
    before = deepcopy(response)
    plan = build_production_plan('action', 15, intent=record['intent'])
    assert [s['min_interaction_events'] for s in plan['slots']] == [2, 3, 1]
    locations = [SceneLocation(**s) for s in identity['locations']]
    schema = contextual_shot_schema(authored_shot_response_schema(
        [c['name'] for c in identity['characters']], require_outcomes=True,
        scene_entities=identity['scene_entities'], bound_handoffs=True), locations, identity['scene_entities'])
    # 当前接口也暴露同一原稿的非法实体引用；不再假设旧接口的接受代表有效。
    with pytest.raises(jsonschema.ValidationError) as rejected:
        jsonschema.validate(response, schema)
    assert 'blocking' in rejected.value.path and 'action_target' in rejected.value.path
    shot = AuthoredShot.model_validate(dict(shot_id=1, **bind_authored_context(
        response['shot'], locations, identity['scene_entities'])))
    with pytest.raises(ValueError, match='独立作用不足'):
        compile_action_units(shot, plan['slots'][0])
    assert all(u.kind == 'transition' for u in shot.action_plan)
    assert sum(s['duration'] - s['result_hold_seconds'] for s in plan['slots'][1:]) < plan['min_interaction_seconds']
    assert response == before


def test_formal_creator_binds_existing_work_into_each_response(monkeypatch):
    from pipeline.storyboard import generate_storyboard
    from test_action_results import annotated_story
    from test_director_intent import Responses, intent_reply
    raw = annotated_story()
    raw['scene_entities'] = ['Door', 'Dust']
    client = Responses({**intent_reply('continuous work'), 'choreography': 'flexible'}, raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    result = generate_storyboard('continuous work', target_duration=15)
    for call, slot in zip(client.calls[2:], result['production_plan']['slots']):
        schedule = call['response_format']['json_schema']['schema']['properties']['shot']['properties']['action_plan']
        assert schedule['type'] == 'object'
        for branch in schedule['oneOf']:
            assert {f'interaction_{i + 1}' for i in range(slot['min_interaction_events'])} <= set(branch['required'])
            assert ('result_hold' in branch['required']) == bool(slot['result_hold_seconds'])
    assert len(client.calls) == 5


@pytest.mark.asyncio
async def test_captured_omission_stops_actual_sdk_flow_without_retries_or_payment(tmp_path, monkeypatch):
    import config
    from pipeline.orchestrator import VideoPipeline
    from test_authoring_transport import http_client
    record, identity, response = captured_failure()
    original = deepcopy(response)
    intent = {k: v for k, v in record['intent'].items() if k != 'source_request'}
    sdk, calls = http_client(monkeypatch, [intent, identity, response])
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **_: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0, take_policy='single_pass')
    with sdk:
        with pytest.raises(ValueError, match='槽位绑定的必需作用'):
            await pipeline.run(record['intent']['source_request'])
        with pytest.raises(ValueError, match='槽位绑定的必需作用'):
            await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert len(calls) == 3
    schema = calls[-1]['response_format']['json_schema']['schema']
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(response, schema)
    assert json.loads((pipeline.workspace / 'planning_shot_001.json').read_text()) == original
    assert pipeline.run_workspace.manifest.paid_take_budget.reservations == []
    assert not (pipeline.workspace / 'planning_shot_002_response.json').exists()
    assert response == original


SCENARIOS = {
    '双人舞': [
        ('offers a moving hand connection', 'partner takes the hand without stopping'),
        ('guides the joined hands into a turn', 'partner rotates into the shared line'),
        ('redirects the turning momentum across the floor', 'partner travels sideways in balance'),
        ('supports the partner through a low arc', 'partner passes through the dip with support'),
        ('lifts the joined hands while stepping through', 'partner rises into the next rotation'),
        ('absorbs the final rotation through the joined arms', 'partner lands balanced alongside the lead'),
    ],
    '货物交接': [
        ('slides the parcel within the receiver reach', 'receiver supports its leading edge'),
        ('rolls the parcel along both supporting hands', 'receiver cradles the parcel base'),
        ('shifts support toward the receiver center', 'receiver bears most of the parcel weight'),
        ('guides the receiver hands under the corners', 'receiver has a stable two handed grip'),
        ('eases the load into the receiver arms', 'receiver carries the full load'),
        ('releases the last supporting hand after the transfer', 'receiver holds the parcel securely alone'),
    ],
    '绳索救援': [
        ('extends the secured line toward the stranded partner', 'partner grasps the line'),
        ('takes up slack while bracing against the anchor', 'partner is supported by the taut line'),
        ('pulls the line as the partner pushes from the ledge', 'partner rises above the lower ledge'),
        ('redirects the loaded line toward the stable platform', 'partner swings toward the safe approach'),
        ('catches the partner forearm during the approach', 'partner reaches the platform edge'),
        ('draws the partner fully over the edge', 'partner stands safely on the platform'),
    ],
}


def scenario_story(topic):
    """Explicit independent annotations, not corrections to any failed model draft."""
    from test_action_results import annotated_story
    from test_director_intent import intent_reply
    ending = f'最后完成{topic}并清楚展示结果。'
    brief = f'15秒，分三段，连续流畅地展示{topic}，动作密集但不要逐步停顿。' + ending
    reply = {**intent_reply(brief), 'choreography': 'flexible',
             'ending_requirement': ending, 'ending_evidence': ending,
             'requested_duration_seconds': 15, 'duration_evidence': '15秒',
             'requested_shot_count': 3, 'shot_count_evidence': '三段'}
    plan = build_production_plan('action', 15, intent={**reply, 'source_request': brief})
    story = annotated_story()
    story.update(title=topic, theme_elements=[topic], scene_entities=[],
                 locations=[{'id': 'shared_space', 'description': 'A stable shared working space'}],
                 characters=[{'name': 'Lead', 'description': 'Adult in blue practical clothing'},
                             {'name': 'Partner', 'description': 'Adult in cream practical clothing'}],
                 story_arc={'goal': topic, 'stakes': 'the cooperative task remains incomplete',
                            'turning_point': 'the partners redirect continuous momentum', 'resolution': ending})
    offset = 0
    previous = 'both participants ready in the shared space'
    for shot, slot in zip(story['shots'], plan['slots']):
        count = slot['min_interaction_events']
        pairs = SCENARIOS[topic][offset:offset + count]
        offset += count
        shot.update(scene_id='shared_space', scene_description='Two cooperating adults in a shared working space.',
                    visual_description='Clear full body composition with natural side lighting and a stable horizon.',
                    key_props=[], continuity_props=[], story_change=pairs[-1][1], story_state_after=pairs[-1][1])
        shot['blocking'] = dict(zip(('Lead', 'Partner'), shot['blocking'].values()))
        for name, blocking in shot['blocking'].items():
            for field in ('facing_target', 'eyeline_target', 'action_target'):
                blocking[field] = 'Partner' if name == 'Lead' else 'Lead'
        shot['camera']['screen_positions'] = dict(zip(('Lead', 'Partner'), shot['camera']['screen_positions'].values()))
        shot['start_state'] = {'location': 'shared space', 'subject': 'Lead and Partner',
                               'action_phase': previous, 'camera': 'wide shot'}
        previous = pairs[-1][1]
        shot['end_state'] = {**shot['start_state'], 'action_phase': previous}
        units = []
        if shot['shot_id'] == 1:
            units.append({'id': 'entry', 'kind': 'transition', 'actor': 'Lead', 'target': '',
                          'action': 'moves into reach while keeping the approach in motion',
                          'result': 'both participants are within reach', 'weight': 6.})
        for index, (action, result) in enumerate(pairs):
            units.append({'id': f'cause_{index}', 'kind': 'interaction', 'actor': 'Lead', 'target': 'Partner',
                          'action': action, 'result': result, 'weight': 1., 'response': 'impact',
                          'link_previous': index > 0, 'overlap': index > 0,
                          'effect': {'mode': 'direct_contact', 'motion': 'sweep', 'scope': 'single',
                                     'source': 'the shared physical connection', 'region': 'the partner contact area',
                                     'unaffected': 'the surrounding environment remains stable'},
                          'resolution': {'target_state': result, 'actor_state': None, 'scene_changes': []}})
        if slot['result_hold_seconds']:
            units.append({'id': 'ending', 'kind': 'hold', 'actor': 'Lead', 'target': '',
                          'action': 'maintains the completed cooperative position', 'result': previous, 'weight': .2})
        shot['action_plan'] = units
    return brief, reply, story, plan


@pytest.mark.parametrize('topic', SCENARIOS)
def test_distinct_performances_survive_actual_sdk_and_global_time_allocation(monkeypatch, topic):
    from authoring_cases import contextual_shot_fixture, scheduled_shot_fixture
    from pipeline.film_rhythm import planned_interaction_seconds, planned_film_timing_issues
    from pipeline.causality import compile_action_contract
    from pipeline.storyboard import generate_storyboard
    from test_authoring_transport import http_client
    brief, reply, source, plan = scenario_story(topic)
    original = deepcopy(source)
    responses = [reply, {k: v for k, v in source.items() if k != 'shots'}]
    responses += [{'shot': {k: v for k, v in scheduled_shot_fixture(contextual_shot_fixture(s), slot).items()
                            if k != 'shot_id'}} for s, slot in zip(source['shots'], plan['slots'])]
    sdk, calls = http_client(monkeypatch, responses)
    with sdk:
        result = generate_storyboard(brief, target_duration=15)
    assert len(calls) == 5 and [s['min_interaction_events'] for s in plan['slots']] == [2, 3, 1]
    assert planned_film_timing_issues(result) == []
    assert planned_interaction_seconds(result['shots']) >= 10.5 - 1e-6
    assert planned_interaction_seconds(result['shots'][:1]) < 3.5  # Global budget, not 70% per shot.
    for source_shot, shot, response, call in zip(source['shots'], result['shots'], responses[2:], calls[2:]):
        jsonschema.validate(response, call['response_format']['json_schema']['schema'])
        assert [u['action'] for u in source_shot['action_plan']] == [b['action'] for b in shot['action_beats']]
        assert [u['result'] for u in source_shot['action_plan']] == [b['visible_result'] for b in shot['action_beats']]
        assert [u['kind'] for u in source_shot['action_plan']] == [b['activity'] for b in shot['action_beats']]
        prompt = ' '.join(compile_action_contract(shot).prompt_parts)
        assert all(u['action'] in prompt for u in source_shot['action_plan'] if u['kind'] == 'interaction')
        assert json.loads(call['messages'][-1]['content'])['slot']['duration'] == shot['duration']
    assert source == original


@pytest.mark.asyncio
async def test_typed_work_is_not_semantic_approval_and_rejection_survives_resume(tmp_path, monkeypatch):
    import config
    from types import SimpleNamespace
    from pipeline.orchestrator import VideoPipeline
    from test_director_intent import Responses
    from test_action_result_flow import annotated_verdict
    brief, reply, source, _ = scenario_story('双人舞')
    # Deliberately false service annotation: a filled interaction is still only an entrance.
    source['shots'][0]['action_plan'][1].update(
        action='walks into the space alone without affecting the partner', result='stands alone')
    author = Responses(reply, source)
    reviewed = []

    def review(**kwargs):
        context = json.loads(kwargs['messages'][-1]['content'])
        story = context['storyboard']
        reviewed.append(story)
        verdict = annotated_verdict(story)
        verdict['units'][1].update(classification_matches_description=False,
                                   reason='offline annotation: an entrance is not an interaction')
        verdict.pop('requested_outcome_valid')
        verdict['requested_outcome'] = {'relation': 'entailed',
            'evidence': [next(p for p in context['outcome_review']['sources'] if p.endswith('/end_state/action_phase'))],
            'reason': 'synthetic ending annotation; the earlier classification still fails'}
        return Responses(verdict).create(**kwargs)

    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: author)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=review))))
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **_: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0, take_policy='single_pass')
    with pytest.raises(ValueError, match='分镜语义验收未通过'):
        await pipeline.run(brief)
    with pytest.raises(ValueError, match='分镜语义验收未通过'):
        await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert len(author.calls) == 5 and len(reviewed) == 1
    assert reviewed[0]['shots'][0]['action_beats'][1]['action'] == source['shots'][0]['action_plan'][1]['action']
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations
