"""真实失败原样回放；离线标注只验证职责边界，不冒充模型理解准确率。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.completion_intent import intent_evidence_sources, validate_intent
from pipeline.result_observations import decode_observations
from test_functional_results import format_for
from pipeline.causality import compile_action_contract


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def captured():
    data = json.loads((Path(__file__).parent / 'fixtures/real_intent_handoff_failure.json').read_text())
    assert hashlib.sha256(data['raw_response']['text'].encode()).hexdigest() == data['hashes'][data['raw_response']['path']]
    return data


def test_original_failure_is_temporal_evidence_not_video_api():
    data = captured()
    raw = json.loads(data['raw_response']['text'])
    assert len(raw['attempts']) == 1
    value = json.loads(raw['attempts'][0]['choices'][0]['content'])
    assert value['observations']['beat:1/completion']['sample_indices'][-1] == 18
    assert value['end_state_sources']['action_phase'] == ['beat:1/completion']
    assert all(v['status'] == 'satisfied' for v in value['observations'].values())
    # 使用原字段集重现真实解码器，而非更改响应使它通过。
    schema = {'properties': {k: {} for k in value}}
    schema['properties']['end_state_context'] = {'properties': {k: {} for k in value['end_state_context']}}
    with pytest.raises(ValueError, match='末帧'):
        decode_observations(value, compile_action_contract(data['shot']).review_projection['completion_contract'], schema, 25)


def goal_story():
    data = captured()
    shot = deepcopy(data['shot'])
    shot['result_admission'] = 'review-required-v1'
    shot.pop('completion_policy')
    for b in shot['action_beats']:
        b.pop('completion_intent', None)
    goal = {'goal': '母亲将一杯温水放到桌边空椅旁。', 'basis': 'user',
            'evidence': '将一杯温水放到桌边空椅旁'}
    return {'shots': [shot], 'production_plan': {'director_intent': {
        **data['intent'], 'completion_goals': [goal]}}}, goal


def test_new_goals_cannot_be_created_from_authored_gaze_or_story_prose():
    story, goal = goal_story()
    sources = intent_evidence_sources(story)
    intent = next(u['completion_intent'] for u in captured()['units'] if u['shot_id'] == 2 and u['beat_index'] == 1)
    with pytest.raises(ValueError):
        validate_intent(intent, sources, '2:1')
    reference = '/intent/completion_goals/0'
    bound = {'kind': 'required', 'goal': goal['goal'], 'basis': 'user', 'reference': reference, 'reason': 'Explicit requirement.'}
    assert validate_intent(bound, sources, '2:0') == bound
    with pytest.raises(ValueError):
        validate_intent({**bound, 'goal': goal['goal'] + ' Must lower shoulders.'}, sources, '2:0')


def test_source_schema_does_not_allow_free_goal_text():
    from pipeline.completion_intent import intent_schema
    import jsonschema
    story, goal = goal_story()
    sources = intent_evidence_sources(story)
    schema = intent_schema(sources, '2:0', [2, 3])
    forged = {'kind': 'required', 'goal': goal['goal'] + ' Close fingers exactly.', 'basis': 'user',
              'reference': '/intent/completion_goals/0', 'reason': 'Source exists.'}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(forged, schema)
    jsonschema.validate({'kind': 'required', 'reference': '/intent/completion_goals/0', 'reason': 'Source goal.'}, schema)


@pytest.mark.parametrize('new_goals', [True, False])
def test_review_instructions_match_the_goal_authority(new_goals, tmp_path, monkeypatch):
    from pipeline.planning_review import ensure_planning_semantics
    from test_result_admission import candidate
    story = candidate()
    intent = story['production_plan']['director_intent']
    if new_goals:
        intent.update(authoring_protocol='scene-units-v1', completion_goals=[{
            'goal': intent['source_request'], 'basis': 'user', 'evidence': intent['source_request']}])
    else:
        intent.pop('completion_goals', None)
    prompts = []
    def capture(_client, system, *_args, **_kwargs):
        prompts.append(system)
        raise RuntimeError('offline request boundary')
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_kw: object())
    monkeypatch.setattr('pipeline.storyboard._call_llm_for_storyboard', capture)
    with pytest.raises(RuntimeError, match='offline request boundary'):
        ensure_planning_semantics(story, tmp_path, lambda *_args: None)
    assert len(prompts) == 1
    assert ('required是必须可见的对象结果' in prompts[0]) is (not new_goals)
    if new_goals:
        assert '不从作者字段的required/incidental标签推导必要性' in prompts[0]


def test_new_handoff_has_a_terminal_snapshot_not_an_event_foreign_key():
    shot = deepcopy(captured()['shot'])
    shot['completion_policy'] = 'source-goals-v1'
    shot['completion_goals'] = {'/intent/completion_goals/0': {
        'goal': 'Mother places the water beside the empty chair.', 'basis': 'user', 'evidence': '放到桌边空椅旁'}}
    for index, b in enumerate(shot['action_beats']):
        b['completion_intent'] = ({'kind': 'required', 'goal': 'Mother places the water beside the empty chair.',
            'basis': 'user', 'reference': '/intent/completion_goals/0', 'reason': 'Explicit requirement.'} if index == 0 else
            {'kind': 'supporting', 'supports': 1, 'reason': 'Realization of the invitation.'})
    props = format_for(shot)['json_schema']['schema']['properties']
    assert 'end_state_sources' not in props
    assert props['terminal_observation']['properties']['sample_index']['const'] == 25


def admitted_story():
    from pipeline.result_admission import apply_result_admission
    story, goal = goal_story()
    units = deepcopy([u for u in captured()['units'] if u['shot_id'] == 2])
    for j, unit in enumerate(units):
        unit['result_admission'] = None
        unit['completion_intent'] = ({'kind': 'required', 'reference': '/intent/completion_goals/0',
            'goal': goal['goal'], 'basis': goal['basis'], 'reason': 'Offline annotation of original user requirement.'}
            if j == 0 else {'kind': 'supporting', 'supports': 1, 'reason': 'Replaceable realization of invitation.'})
    return apply_result_admission(story, units)


def annotated_snapshot(shot):
    """测试服务标注，不是对真实原片的补写或批准。"""
    from test_functional_results import annotated_wire
    schema = format_for(shot)['json_schema']['schema']
    legacy = deepcopy(schema)
    legacy['properties'].pop('terminal_observation')
    legacy['properties']['end_state_sources'] = {'type': 'object', 'properties': {}}
    wire = annotated_wire(legacy)
    wire.pop('end_state_sources')
    wire['terminal_observation'] = {'sample_index': 25, 'status': 'observed',
        'action_phase': 'The invitation is complete; both wait.', 'pose_and_gaze': 'Both remain visible.',
        'prop_state': 'The water is beside the empty chair.', 'open_motion': ''}
    return wire


@pytest.mark.parametrize('cover_final_goal', [True, False])
def test_goal_coverage_is_checked_after_all_batches(cover_final_goal, tmp_path, monkeypatch):
    from pipeline.planning_review_batches import ensure_batched_planning_review
    story = admitted_story()
    original = story['shots'][0]
    second = {'goal': '小安接住杯子', 'basis': 'user', 'evidence': '小安接住杯子'}
    story['production_plan']['director_intent']['completion_goals'].append(second)
    story['shots'] = [{**deepcopy(original), 'shot_id': i} for i in range(1, 26)]
    if cover_final_goal:
        last = story['shots'][-1]
        reference = '/intent/completion_goals/1'
        last['completion_goals'][reference] = second
        last['action_beats'][0]['completion_intent'].update(reference=reference, goal=second['goal'])
    scopes, saved = [], {}
    def accepted_batch(scoped, *_args):
        # 独立服务的离线标注；本用例检验聚合层，不宣称模型能正确判断目标。
        scopes.append(scoped['review_scope'])
        return scoped
    monkeypatch.setattr('pipeline.planning_review.ensure_planning_semantics', accepted_batch)
    save = lambda name, data: saved.update({name: data})
    if cover_final_goal:
        admitted = ensure_batched_planning_review(story, tmp_path, save)
        assert len(admitted['shots']) == 25
        assert saved['planning_semantic_review.json']['status'] == 'passed'
    else:
        with pytest.raises(ValueError, match='全片未承接必要目标'):
            ensure_batched_planning_review(story, tmp_path, save)
        assert not saved
    assert len(scopes) > 1 and scopes[-1]['is_final_batch'] is True
    assert not any(scope['is_final_batch'] for scope in scopes[:-1])


def test_authored_states_do_not_reintroduce_optional_poses_through_other_fields():
    from pipeline.semantic_review import _review_contract
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    story = admitted_story()
    shot = story['shots'][0]
    contract = _review_contract(shot)
    assert {v['kind'] for v in contract['completion_contract']['criteria'].values()} == {'completion', 'realization'}
    detail = shot['action_beats'][1]['action']
    assert detail not in json.dumps(contract, ensure_ascii=False)
    # 生成保留作者动作；验收不凭这些作者文字新增要求。
    assert detail in compile_normal_provider_prompt(shot, story, has_observed_start=False)
    assert '母亲将一杯温水放到桌边空椅旁' in compile_policy_safe_prompt(
        shot, has_state_reference=False, image_role=None, reference_count=0)
    assert [b['action'] for b in shot['action_beats']] == [b['action'] for b in captured()['shot']['action_beats']]


@pytest.mark.parametrize('status,expected', [('satisfied', True), ('violated', False)])
def test_early_event_and_later_snapshot_have_independent_time_scopes(status, expected):
    shot = admitted_story()['shots'][0]
    value = annotated_snapshot(shot)
    value['observations']['beat:1/completion'].update(status=status, sample_indices=[3, 18],
        observation='The event is complete.' if expected else 'The action is only attempted.')
    value['observations']['beat:2/completion'].update(status='unknown', sample_indices=[10])
    result = decode_observations(value, compile_action_contract(shot).review_projection['completion_contract'],
                                 format_for(shot)['json_schema']['schema'], 25)
    assert result['accepted'] is expected
    assert result['observed_end_state']['action_phase'] == value['terminal_observation']['action_phase']
    assert result['observed_end_state']['action_phase'] != value['observations']['beat:1/completion']['observation']


@pytest.mark.parametrize('change', [{'sample_index': 18}, {'sample_index': True}, {'status': 'unknown'},
                                   {'action_phase': ''}, {'prop_state': None}, {'invented': 'state'}])
def test_invalid_snapshot_cannot_be_used_for_handoff(change):
    shot = admitted_story()['shots'][0]
    value = annotated_snapshot(shot)
    value['terminal_observation'].update(change)
    with pytest.raises(ValueError, match='片尾快照'):
        decode_observations(value, compile_action_contract(shot).review_projection['completion_contract'],
                             format_for(shot)['json_schema']['schema'], 25)


@pytest.mark.parametrize('goal,detail', [
    ('The courier hands the parcel to the receiver.', 'The courier touches the collar.'),
    ('The dancers complete the lift.', 'The dancer looks toward the ceiling.'),
    ('The witness recognizes the suspect.', 'The witness remains still for a moment.'),
    ('The lamp illuminates its translucent shade.', 'A crescent reflection rests on the desk.'),
])
@pytest.mark.parametrize('explicit', [False, True])
def test_requirement_authority_is_independent_of_scene_and_action_size(goal, detail, explicit):
    from pipeline.result_admission import apply_result_admission
    story, _ = goal_story()
    intent = story['production_plan']['director_intent']
    intent['source_request'] = goal + (' ' + detail if explicit else '')
    intent['completion_goals'] = [{'goal': goal, 'basis': 'user', 'evidence': goal}]
    if explicit:
        intent['completion_goals'].append({'goal': detail, 'basis': 'user', 'evidence': detail})
    units = deepcopy([u for u in captured()['units'] if u['shot_id'] == 2])
    for j, unit in enumerate(units):
        unit['result_admission'] = None
        if j == 0 or (j == 1 and explicit):
            item = intent['completion_goals'][j]
            unit['completion_intent'] = {'kind': 'required', 'reference': f'/intent/completion_goals/{j}',
                'basis': item['basis'], 'goal': item['goal'], 'reason': 'Offline explicit requirement.'}
        else:
            unit['completion_intent'] = {'kind': 'supporting', 'supports': 1, 'reason': 'Optional realization.'}
    shot = apply_result_admission(story, units)['shots'][0]
    wire = annotated_snapshot(shot)
    wire['observations']['beat:2/completion']['status'] = 'violated'
    schema = format_for(shot)['json_schema']['schema']
    catalog = compile_action_contract(shot).review_projection['completion_contract']
    assert decode_observations(wire, catalog, schema, 25)['accepted'] is (not explicit)
    wire['observations']['beat:1/completion']['status'] = 'violated'
    assert decode_observations(wire, catalog, schema, 25)['accepted'] is False


def test_goal_coverage_and_saved_source_integrity():
    from pipeline.completion_intent import validate_goal_coverage
    story = admitted_story()
    validate_goal_coverage(story)
    forged = deepcopy(story)
    forged['shots'][0]['completion_goals']['/intent/completion_goals/0']['goal'] = 'Another goal.'
    with pytest.raises(ValueError, match='冻结目录'):
        validate_goal_coverage(forged)
    story['production_plan']['director_intent']['completion_goals'].append({
        'goal': '小安接住杯子', 'basis': 'user', 'evidence': '小安接住杯子'})
    with pytest.raises(ValueError, match='未承接'):
        validate_goal_coverage(story)


def test_authored_detail_mutation_changes_generation_but_not_required_review():
    from pipeline.semantic_review import _review_contract
    from pipeline.provider_prompt import compile_normal_provider_prompt
    story = admitted_story()
    shot = story['shots'][0]
    before = _review_contract(shot)
    original_prompt = compile_normal_provider_prompt(shot, story, has_observed_start=False)
    shot['action_beats'][0]['action'] += ' Then raises two fingers and looks toward the window.'
    shot['action_beats'][0]['outcomes'][0]['state'] += ' Keeps the elbow at a precise angle.'
    assert _review_contract(shot) == before
    prompt = compile_normal_provider_prompt(shot, story, has_observed_start=False)
    assert prompt != original_prompt and 'raises two fingers' in prompt and 'precise angle' in prompt


def test_missing_initial_goal_catalog_is_not_silently_inferred_from_future_actions():
    from pipeline.director_intent import resolve_director_intent
    from types import SimpleNamespace
    data = captured()
    raw = {k: v for k, v in data['intent'].items() if k not in ('source_request', 'style_input', 'authoring_protocol')}
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
            content=json.dumps(raw), refusal=None))])
    with pytest.raises(ValueError, match='未按关键词'):
        resolve_director_intent(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
                                data['intent']['source_request'], style_input=data['intent']['style_input'])
    assert len(calls) == 1
    assert json.loads(calls[0]['messages'][1]['content'])['request'] == data['intent']['source_request']
    assert 'storyboard' not in json.loads(calls[0]['messages'][1]['content'])


def test_original_media_remains_unapproved_and_invalid_review_is_not_retried(tmp_path):
    from pipeline.semantic_review import SemanticTakeReviewer, SemanticReviewUnavailableError
    from types import SimpleNamespace
    record = captured()
    video = Path(__file__).resolve().parents[1] / record['video']['path']
    if not video.is_file():
        pytest.skip('original media absent; raw response replay still tested')
    assert hashlib.sha256(video.read_bytes()).hexdigest() == record['video']['sha256']
    content = json.loads(record['raw_response']['text'])['attempts'][0]['choices'][0]['content']
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=content, refusal=None))])
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-captured',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    for _ in range(2):
        with pytest.raises(SemanticReviewUnavailableError):
            reviewer.review(str(video), record['shot'])
    assert len(calls) == 1
    evidence = json.loads(next(reviewer.cache_dir.glob('*.evidence.json')).read_text())
    assert evidence['status'] == 'unavailable'
    assert 'sample_timestamps' in evidence
    assert evidence['response'] == json.loads(content)
    assert hashlib.sha256(video.read_bytes()).hexdigest() == record['video']['sha256']
