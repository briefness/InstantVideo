"""真实原始失败保持失败；新协议用独立离线标注验证，不冒充新模型输出。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from pipeline import goal_allocation as allocation
from pipeline.completion_intent import source_goals, intent_evidence_sources, admitted_unit_intents, validate_goal_coverage
from pipeline.planning_review_protocol import review_targets, decode_review_response
from pipeline.result_admission import apply_result_admission
from pipeline.causality import compile_action_contract


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def captured():
    fixture = json.loads((Path(__file__).parent / 'fixtures/real_goal_support_cycle.json').read_text())
    for name, record in fixture['files'].items():
        assert hashlib.sha256(record['text'].encode()).hexdigest() == record['sha256']
    return (json.loads(fixture['files']['storyboard.json']['text']),
            json.loads(fixture['files']['planning_semantic_review.json']['text']))


def evidence(quote, *units):
    return {'quote': quote, 'units': list(units), 'relation': 'entailed',
            'reason': 'Independent offline annotation of source requirement and authored evidence.'}


def annotation(story):
    """标注本次原稿；不导入、修改或纠正那份真实失败审查。"""
    goals = source_goals(story)
    refs = list(goals)
    texts = [g['goal'] for g in goals.values()]
    return {refs[0]: {'parts': [evidence(texts[0], '1:0')]},
            refs[1]: {'parts': [evidence('母亲林阿姨看见小安，', '1:1'),
                               evidence('将一杯温水放到桌边空椅旁，以安静的方式示意她可以坐下。', '2:0')]},
            refs[2]: {'parts': [evidence(texts[2], '3:0', '3:1', '3:2', '3:3')]},
            refs[3]: {'parts': [evidence(texts[3], '4:0', '4:1', '4:2')]}}


def bind(story, judgments):
    targets = review_targets(story)
    units = [{'shot_id': t['shot_id'], 'beat_index': t['beat_index']} for t in targets.values()]
    result = allocation.decode_allocation({'units': units, 'goal_allocation': judgments}, source_goals(story), targets)
    return apply_result_admission(story, result['units'])


def test_real_response_replays_cycle_and_missing_goal_without_modification():
    story, record = captured()
    targets = review_targets(story, admit_intents=True)
    for t in targets.values():
        t['candidate_results'] = {}
    raw = record['verdict']
    units = decode_review_response(raw, targets, evidence_sources=intent_evidence_sources(story))['units']
    assert all(u['completion_intent']['kind'] == 'supporting' for u in units if u['shot_id'] == 3)
    assert raw['units']['3:3']['completion_intent']['supports'] == 5
    assert raw['units']['3:4']['completion_intent']['supports'] == 4
    with pytest.raises(ValueError, match='循环'):
        admitted_unit_intents(story, units)
    with pytest.raises(ValueError, match='目标覆盖'):
        allocation.decode_allocation({'units': units}, source_goals(story), targets)


def test_goal_evidence_reaches_generation_and_is_observed_once_for_combined_work():
    from pipeline.provider_prompt import compile_generation_preview
    import config
    story, _ = captured()
    original = deepcopy(story)
    admitted = bind(story, annotation(story))
    assert story == original
    shot = admitted['shots'][2]
    contract = compile_action_contract(shot)
    criteria = contract.review_projection['completion_contract']['criteria']
    required = [c for c in criteria.values() if c['kind'] == 'completion']
    assert len(required) == 1 and required[0]['beat_ids'] == [1, 2, 3, 4]
    assert required[0]['action'] == source_goals(story)['/intent/completion_goals/2']['goal']
    assert criteria['beat:5/completion']['supports'] == []
    assert [e['event_id'] for e in contract.review_projection['action_events']] == [3, 4]
    generated = '\n'.join(contract.prompt_parts)
    for beat in original['shots'][2]['action_beats']:
        assert beat['action'] in generated
    assert required[0]['action'] in generated
    # 看见与放杯跨镜，不要求两镜各重复完成完整复合事件。
    first = compile_action_contract(admitted['shots'][0]).review_projection['completion_contract']['criteria']
    second = compile_action_contract(admitted['shots'][1]).review_projection['completion_contract']['criteria']
    assert any(c.get('action') == '母亲林阿姨看见小安，' for c in first.values())
    assert not any(c.get('action') == '母亲林阿姨看见小安，' for c in second.values())
    preview = compile_generation_preview(admitted, model=config.SEEDANCE_MODEL)
    for row, s in zip(preview['shots'], admitted['shots']):
        functions = compile_action_contract(s).review_projection['completion_contract']['criteria']
        for profile in row['profiles'].values():
            assert not profile['preflight_issues']
            for c in functions.values():
                if c['kind'] == 'completion':
                    assert c['action'] in profile['prompt']


@pytest.mark.parametrize('mutation', ['missing_goal', 'missing_part', 'wrong_goal', 'cycle', 'cross_shot',
                                    'duplicate_unit', 'duplicate_part', 'unknown', 'contradicted', 'new_pose'])
def test_invalid_coverage_never_becomes_success(mutation):
    story, _ = captured()
    value = annotation(story)
    ref = '/intent/completion_goals/2'
    part = value[ref]['parts'][0]
    if mutation == 'missing_goal':
        value.pop(ref)
    elif mutation == 'missing_part':
        value[ref]['parts'] = []
    elif mutation == 'wrong_goal':
        part['quote'] = '不存在的目标'
    elif mutation == 'cycle':
        part['supports'] = '3:3'
    elif mutation == 'cross_shot':
        part['units'].append('4:0')
    elif mutation == 'duplicate_unit':
        part['units'].append('3:3')
    elif mutation == 'duplicate_part':
        value[ref]['parts'].append(deepcopy(part))
    elif mutation in ('unknown', 'contradicted'):
        part['relation'] = 'undetermined' if mutation == 'unknown' else mutation
    else:
        part['quote'] += '必须抬眼停留。'
    with pytest.raises(ValueError):
        bind(story, value)


@pytest.mark.parametrize('text', ['快递员递出包裹，收件人签收。', 'Dancers exchange partners and finish together.',
                                '修理人员断电后更换保险丝，设备重新启动。', '镜头停留在静止的山脉上。'])
def test_domain_independent_many_actions_one_goal_and_many_goals_one_action(text):
    from test_goal_authority import goal_story
    story, _ = goal_story()
    intent = story['production_plan']['director_intent']
    intent['source_request'] = text + ' 保持正常速度。'
    intent['completion_goals'] = [{'goal': text, 'basis': 'user', 'evidence': text},
                                 {'goal': '保持正常速度。', 'basis': 'user', 'evidence': '保持正常速度。'}]
    refs = list(source_goals(story))
    keys = list(review_targets(story))
    judgments = {refs[0]: {'parts': [evidence(text, *keys)]},
                 refs[1]: {'parts': [evidence('保持正常速度。', keys[0])]}}
    admitted = bind(story, judgments)
    criteria = compile_action_contract(admitted['shots'][0]).review_projection['completion_contract']['criteria']
    assert len([c for c in criteria.values() if c['kind'] == 'completion']) == 2
    assert len(criteria['beat:1/completion']['supports']) == 2
    # 这里检验关联基数与无题材分支，不宣称原家庭动作实现这些合成主题。


def test_formal_review_uses_new_schema_once_and_failed_cache_never_retries(tmp_path, monkeypatch):
    from pipeline.planning_review import ensure_planning_semantics
    from test_action_result_flow import annotated_verdict
    from planning_protocol_cases import wire_verdict, annotated_outcome
    story, _ = captured()
    saved, calls = {}, []
    def review(_client, _system, user, **kwargs):
        payload = json.loads(user)
        assert payload['version'] == allocation.VERSION
        assert all('completion_owners' not in t for t in payload['review_targets'].values())
        assert kwargs['repair_json'] is False
        v = annotated_verdict(story)
        for u in v['units']:
            u.pop('result_admission', None)
        v['goal_allocation'] = annotation(story)
        v = annotated_outcome(v, story)
        request = {'response_format': {'json_schema': {'schema': kwargs['response_schema']}}, 'messages': [{'content': user}]}
        result = wire_verdict(v, request)
        jsonschema.validate(result, kwargs['response_schema'])
        calls.append(result)
        return result
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: object())
    monkeypatch.setattr('pipeline.storyboard._call_llm_for_storyboard', review)
    def save(name, value):
        from pipeline.run_state import RunWorkspace
        RunWorkspace._atomic_write_json(tmp_path / name, value)
        saved[name] = value
    admitted = ensure_planning_semantics(story, tmp_path, save)
    assert len(calls) == 1
    assert ensure_planning_semantics(story, tmp_path, save) == admitted
    assert len(calls) == 1
    saved['planning_semantic_review.json']['verdict']['goal_allocation']['/intent/completion_goals/2']['parts'] = []
    save('planning_semantic_review.json', saved['planning_semantic_review.json'])
    for _ in range(2):
        with pytest.raises(ValueError, match='全片未承接'):
            ensure_planning_semantics(story, tmp_path, save)
    assert len(calls) == 1


def test_multibatch_scope_only_checks_goal_completeness_after_merge():
    story, _ = captured()
    judgments = annotation(story)
    admitted_shots = []
    for shot in story['shots']:
        scoped = {**story, 'shots': [shot], 'review_scope': {'is_final_batch': shot['shot_id'] == 4}}
        local = {ref: {'parts': [p for p in j['parts'] if p['units'][0].startswith(f"{shot['shot_id']}:")]}
                 for ref, j in judgments.items()}
        admitted_shots += bind(scoped, local)['shots']
    result = {**story, 'shots': admitted_shots}
    validate_goal_coverage(result)
    result['shots'][2]['action_beats'][3]['completion_intent']['requirements'] = []
    # 删除一个证据边不等于删除目标，另外三个动作仍绑定同一完整功能；
    # 语义是否足够是独立审查责任，不用关键词伪装语义保证。
    validate_goal_coverage(result)
    for b in result['shots'][2]['action_beats']:
        b['completion_intent']['requirements'] = []
    with pytest.raises(ValueError, match='全片未承接'):
        validate_goal_coverage(result)


@pytest.mark.parametrize('status,accepted', [('satisfied', True), ('violated', False), ('unknown', None)])
def test_observations_require_function_but_not_optional_performance(status, accepted):
    from test_goal_authority import annotated_snapshot
    from test_functional_results import format_for
    from pipeline.result_observations import decode_observations
    story, _ = captured()
    shot = bind(story, annotation(story))['shots'][2]
    catalog = compile_action_contract(shot).review_projection['completion_contract']
    schema = format_for(shot)['json_schema']['schema']
    value = annotated_snapshot(shot)
    key = next(k for k, c in catalog['criteria'].items() if c['kind'] == 'completion')
    value['observations'][key]['status'] = status
    for i in range(1, 6):
        assert f'beat:{i}/completion' not in value['observations']
    if accepted is None:
        with pytest.raises(ValueError, match='观察不足'):
            decode_observations(value, catalog, schema, 25)
    else:
        result = decode_observations(value, catalog, schema, 25)
        assert result['accepted'] is accepted
        assert all(e['result_visible'] is accepted for e in result['action_event_evidence'])


def test_continuation_uses_allocated_functions_and_preserves_source_performance(tmp_path):
    from pipeline.continuation import prepare_continuation
    from test_continuation import scenario
    from test_director_intent import Responses
    story, _ = captured()
    shot = bind(story, annotation(story))['shots'][2]
    _, previous, tail = scenario(tmp_path)
    previous['scene_id'] = shot['scene_id']
    shot['continuity_from_previous'] = 'seamless'
    before = deepcopy(shot)
    sdk = Responses({'compatible': True, 'entry_action': 'Continue from the actual previous state.',
                     'entry_visible_result': 'compatible', 'reason': 'offline annotation'})
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk)
    context = json.loads(sdk.calls[0]['messages'][-1]['content'])
    assert context['completion_contract']['intent_policy'] == allocation.POLICY
    assert 'Optional realization' in context['planned_shot']['action_beats'][4]['action']
    assert result['action_beats'] == before['action_beats']
    assert shot == before


@pytest.mark.parametrize('goal,actions', [
    ('Courier hands over the parcel; receiver signs the receipt.',
     ['Courier hands the parcel to receiver.', 'Receiver signs the receipt.', 'Courier adjusts a hat.']),
    ('Partners exchange places and finish side by side.',
     ['Partners exchange places in one connected turn.', 'Partners finish side by side.', 'One dancer smiles.']),
    ('Technician disconnects power, then replaces the fuse.',
     ['Technician disconnects power.', 'Technician replaces the fuse.', 'Technician wipes a sleeve.']),
    ('Lamp rotates to show its profile and translucent shade.',
     ['Lamp rotates to reveal its curved profile.', 'Light passes through the translucent shade.', 'A reflection moves on the table.']),
])
def test_other_scenes_keep_real_functions_and_replaceable_details_separate(goal, actions):
    from pipeline.result_observations import completion_contract
    story = {'production_plan': {'director_intent': {'authoring_protocol': 'scene-units-v1',
        'source_request': goal, 'completion_goals': [{'goal': goal, 'basis': 'user', 'evidence': goal}]}},
        'shots': [{'shot_id': 1, 'result_admission': 'review-required-v1',
            'production_slot': {'choreography': 'flexible'}, 'action_beats': [
                {'actor': 'subject', 'activity': 'transition', 'action': action, 'visible_result': action}
                for action in actions]}]}
    ref = '/intent/completion_goals/0'
    judgments = {ref: {'parts': [evidence(goal, '1:0', '1:1')]}}
    admitted = bind(story, judgments)
    criteria = completion_contract(admitted['shots'][0])['criteria']
    necessary = [c for c in criteria.values() if c['kind'] == 'completion']
    assert len(necessary) == 1 and necessary[0]['action'] == goal
    assert necessary[0]['beat_ids'] == [1, 2]
    assert criteria['beat:3/completion']['supports'] == []
    assert [b['action'] for b in admitted['shots'][0]['action_beats']] == actions
    with pytest.raises(ValueError, match='全片未承接'):
        bind(story, {ref: {'parts': []}})


def test_ongoing_goal_can_be_explicitly_allocated_across_shots_without_peer_links():
    from test_goal_authority import goal_story
    story, goal = goal_story()
    original = story['shots'][0]
    story['shots'] = [{**deepcopy(original), 'shot_id': i} for i in (1, 2)]
    ref = '/intent/completion_goals/0'
    judgments = {ref: {'parts': [evidence(goal['goal'], '1:0'), evidence(goal['goal'], '2:0')]}}
    result = bind(story, judgments)
    validate_goal_coverage(result)
    assert all(s['completion_policy'] == allocation.POLICY for s in result['shots'])


def test_allocated_contract_cannot_be_silently_read_as_legacy():
    from pipeline.completion_intent import admitted_shot_intents
    story, _ = captured()
    shot = bind(story, annotation(story))['shots'][0]
    for policy in ('admitted-intent-v1', 'source-goals-v1'):
        wrong = {**shot, 'completion_policy': policy}
        with pytest.raises(ValueError, match='历史逐动作合同不能混入'):
            admitted_shot_intents(wrong)
