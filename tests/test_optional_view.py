"""可选执行能力不能成为新业务义务；原始失败响应原样离线重放。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from test_goal_allocation import offline


def captured():
    record = json.loads((Path(__file__).parent / 'fixtures/real_optional_view_failure.json').read_text())
    for item in record['files'].values():
        assert hashlib.sha256(item['text'].encode()).hexdigest() == item['sha256']
    values = {k: json.loads(v['text']) for k, v in record['files'].items()}
    story = values['result_admission.json']['execution_storyboard']
    review = next(v for k, v in values.items() if k.startswith('semantic_reviews/'))
    response = next(v for k, v in values.items() if k.startswith('continuations/'))
    return record, values, story, review, json.loads(response['response']['choices'][0]['content'])


def case(tmp_path):
    import cv2
    import numpy as np
    _, _, story, review, response = captured()
    previous, shot = deepcopy(story['shots'][:2])
    previous['observed_end_state'] = review['review']['observed_end_state']
    tail = tmp_path / 'tail.jpg'
    assert cv2.imwrite(str(tail), np.zeros((496, 864, 3), dtype='uint8'))
    return shot, previous, tail, response


def test_real_compatible_reply_does_not_turn_optional_view_into_a_new_requirement(tmp_path):
    from pipeline.continuation import prepare_continuation
    from pipeline.generator import VideoGenerator
    from test_director_intent import Responses
    shot, previous, tail, response = case(tmp_path)
    original = deepcopy((shot, response))
    sdk = Responses(response)
    execution = prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk)
    assert (shot, response) == original
    assert execution['action_beats'] == shot['action_beats'] and execution['camera'] == shot['camera']
    assert 'reference_view' not in execution
    decision = execution['reference_view_decision']
    assert decision['mode'] == 'native_reference' and decision['reason'] == 'invalid_proposal'
    gen = VideoGenerator(output_dir=str(tmp_path), review_content=False)
    refs, role = gen._build_image_refs(execution, str(tail), previous)
    assert refs == [str(tail)] and role == 'reference_image'
    assert gen._generation_provenance(execution, refs, role)['reference_view_decision'] == decision
    cached = json.loads((tmp_path / 'continuations/shot_002.json').read_text())
    assert cached['plan'] == response and cached['view_decision'] == decision
    assert prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk, allow_create=False) == execution
    assert len(sdk.calls) == 1


@pytest.mark.parametrize('guidance', [None, '', {'unexpected': 'object'}])
def test_missing_entry_advice_cannot_veto_compatible_authored_actions(tmp_path, guidance):
    from pipeline.continuation import prepare_continuation
    from test_director_intent import Responses
    shot, previous, tail, response = case(tmp_path)
    response.update(entry_action=guidance, view_window=None)
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=Responses(response))
    assert result['action_beats'] == shot['action_beats']
    assert 'observed_entry_path' not in result


@pytest.mark.parametrize('mutation', ['incompatible', 'unknown', 'missing_core', 'bad_reason', 'foreign_field'])
def test_optional_failure_never_hides_invalid_required_continuity(tmp_path, mutation):
    from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
    from test_director_intent import Responses
    shot, previous, tail, response = case(tmp_path)
    if mutation == 'incompatible':
        response['compatible'] = False
    elif mutation == 'unknown':
        response['compatible'] = None
    elif mutation == 'missing_core':
        del response['compatible']
    elif mutation == 'bad_reason':
        response['reason'] = ''
    else:
        response['replace_action'] = 'Invent completion'
    sdk = Responses(response)
    with pytest.raises(ContinuationUnavailableError):
        prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk)
    assert len(sdk.calls) == 1 and not list((tmp_path / 'continuations').glob('shot_*.json'))


@pytest.mark.parametrize('mutation', ['source', 'decision', 'missing_decision', 'proposal', 'camera'])
def test_selected_route_cannot_change_during_recovery(tmp_path, mutation):
    from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
    from test_director_intent import Responses
    shot, previous, tail, response = case(tmp_path)
    sdk = Responses(response)
    prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk)
    path = tmp_path / 'continuations/shot_002.json'
    saved = json.loads(path.read_text())
    if mutation == 'source':
        tail.write_bytes(tail.read_bytes() + b'changed')
    elif mutation == 'camera':
        shot['camera']['primary_movement'] = 'pan left'
    elif mutation == 'missing_decision':
        saved.pop('view_decision')
    elif mutation == 'decision':
        saved['view_decision']['mode'] = 'derived_frame'
    else:
        saved['plan']['view_window'] = None
    if mutation not in ('source', 'camera'):
        path.write_text(json.dumps(saved))
    with pytest.raises(ContinuationUnavailableError):
        prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk, allow_create=False)
    assert len(sdk.calls) == 1


def test_historical_protocol_stays_strict_and_actual_failure_is_not_rewritten(tmp_path):
    from pipeline.continuation import _validate_plan, ContinuationUnavailableError
    from pipeline.reference_view import camera_view_context
    shot, previous, tail, response = case(tmp_path)
    with pytest.raises(ContinuationUnavailableError, match='取景'):
        _validate_plan(response, camera_view=camera_view_context(shot, previous, tail))
    assert response['view_window']['scale'] == 1


@pytest.mark.parametrize('proposal', [None, 'not a rectangle', {}, [], {'scale': -2}])
def test_view_capability_can_be_absent_without_changing_requirements(tmp_path, proposal):
    from pipeline.continuation import prepare_continuation
    from test_director_intent import Responses
    shot, previous, tail, response = case(tmp_path)
    response['view_window'] = proposal
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=Responses(response))
    assert result['reference_view_decision']['mode'] == 'native_reference'
    assert result['camera'] == shot['camera'] and result['action_beats'] == shot['action_beats']


def test_new_decision_is_part_of_paid_request_provenance(tmp_path):
    from pipeline.continuation import prepare_continuation
    from pipeline.generator import VideoGenerator
    from test_director_intent import Responses
    shot, previous, tail, response = case(tmp_path)
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=Responses(response))
    gen = VideoGenerator(output_dir=str(tmp_path), review_content=False)
    refs, role = gen._build_image_refs(result, str(tail), previous)
    descriptor = gen._generation_provenance(result, refs, role,
        prompt_profile='normal', prompt_fingerprint='a' * 64)
    gen._write_generation_provenance(2, descriptor, 'offline-task')
    descriptor['task_id'] = 'offline-task'
    assert gen._pending_descriptor_matches(descriptor, result, refs, role)
    changed = deepcopy(result)
    changed['reference_view_decision']['reason'] = 'not_proposed'
    assert not gen._pending_descriptor_matches(descriptor, changed, refs, role)


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_optional_entry_is_optional_for_other_scenes_even_without_view(tmp_path, topic):
    from test_creation_schedule_flow import scenario_story
    from test_staging_direction import compile_story
    from test_goal_allocation import bind, evidence
    from pipeline.continuation import prepare_continuation
    from test_director_intent import Responses
    _, _, raw, plan = scenario_story(topic)
    goals = ['; '.join(u['action'] for u in s['action_plan'] if u['kind'] == 'interaction')
             for s in raw['shots']]
    plan['director_intent'].update(authoring_protocol='scene-units-v1',
        source_request=plan['director_intent']['source_request'] + '; '.join(goals),
        completion_goals=[{'goal': g, 'basis': 'user', 'evidence': g} for g in goals])
    story = compile_story(raw, plan)
    for s in story['shots']:
        s['result_admission'] = 'review-required-v1'
    admitted = bind(story, {f'/intent/completion_goals/{i}': {'parts': [evidence(g,
        *(f"{s['shot_id']}:{j}" for j in range(len(s['action_beats']))))]}
        for i, (s, g) in enumerate(zip(story['shots'], goals))})
    previous, shot = deepcopy(admitted['shots'][:2])
    # 此专项隔离历史尾帧协议的可选入场字段；动态视频证据另有流程测试。
    shot.pop('continuity_evidence', None)
    shot.pop('spatial_control', None)
    previous['observed_end_state'] = deepcopy(previous['end_state'])
    shot['continuity_from_previous'] = 'seamless'
    _, _, tail, _ = case(tmp_path)
    sdk = Responses({'compatible': True, 'reason': 'Offline compatible annotation; no entry advice needed.'})
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk)
    assert result['action_beats'] == shot['action_beats'] and result['camera'] == shot['camera']
    assert 'observed_entry_path' not in result and 'reference_view' not in result
    assert len(sdk.calls) == 1
