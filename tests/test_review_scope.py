"""原始视频/响应只读回放；范围标注与真实观察不得混淆。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.causality import compile_action_contract
from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
from pipeline.semantic_review import SemanticTakeReviewer, SemanticReviewUnavailableError, _apply_event_evidence_verdict
from test_performance_outcomes import outcome_shot, outcome_response
from test_sequence_review import reviewer_for, offline_frames


def captured():
    record = json.loads((Path(__file__).parent / 'fixtures/real_review_scope_failure.json').read_text())
    values = {}
    for name, source in record['sources'].items():
        assert hashlib.sha256(source['text'].encode()).hexdigest() == source['sha256']
        values[name] = json.loads(source['text'])
    evidence = next(v for n, v in values.items() if n.endswith('.evidence.json'))
    return record, values['storyboard.json'], evidence


def ordinary_shot():
    shot = outcome_shot()
    shot['production_slot'].update(authoring_protocol='scene-units-v1', min_interaction_seconds=0,
                                   result_hold_seconds=0)
    return shot


def test_original_result_failure_is_not_hidden_by_irrelevant_precision():
    _, story, evidence = captured()
    shot = story['shots'][0]
    original = deepcopy(evidence)
    response = deepcopy(evidence['response'])
    contract = compile_action_contract(shot)
    _apply_event_evidence_verdict(response, contract, evidence['action_event_evidence'],
                                timestamps=evidence['sample_timestamps'], media_duration=evidence['media_duration'])
    assert response['accepted'] is False and response['action_sequence_valid'] is False
    assert response['timing_assessment']['status'] == 'not_required'
    assert 'result_visible' in response['failure_reason']
    assert '椅面' in response['failure_reason']
    assert evidence == original
    # 生成端没有漏掉结果；审查不能把新策略误当作把原片修好了。
    for prompt in (compile_normal_provider_prompt(shot, story, has_observed_start=False),
                   compile_policy_safe_prompt(shot, storyboard=story, has_state_reference=False,
                                              image_role=None, reference_count=0)):
        for event in contract.review_projection['action_events']:
            for result in event['outcomes']:
                assert result['state'] in prompt


@pytest.mark.parametrize('topic', ['家庭剧情', '设备搬运', '舞伴交接', '产品演示'])
def test_ordinary_schema_requests_causal_observations_not_invented_measurements(topic):
    shot = ordinary_shot()
    shot['production_slot']['creative_brief'] = topic
    contract = compile_action_contract(shot)
    fields = contract.event_evidence_schema()['items']['properties']
    assert 'observed_effect_seconds' not in fields and 'active_intervals' not in fields
    assert 'causal_order_valid' in fields and 'result_visible' in fields
    assert 'observed_effect_seconds' not in contract.event_review_instruction()
    response = outcome_response()
    for item in response['action_event_evidence']:
        for key in list(item):
            if key.startswith('observed_') or key == 'active_intervals':
                item.pop(key)
    _apply_event_evidence_verdict(response, contract, response['action_event_evidence'])
    assert response['action_sequence_valid']
    response['action_event_evidence'][0]['causal_order_valid'] = False
    _apply_event_evidence_verdict(response, contract, response['action_event_evidence'])
    assert response['action_sequence_valid'] is False


@pytest.mark.parametrize('mode', ['local_budget', 'result_hold', 'whole_film', 'legacy'])
def test_required_time_is_not_waived_and_preserves_independent_failure(mode):
    shot = ordinary_shot()
    if mode == 'local_budget':
        shot['production_slot']['min_interaction_seconds'] = 1
    elif mode == 'result_hold':
        shot['production_slot']['result_hold_seconds'] = .5
    elif mode == 'whole_film':
        shot['generation_rhythm'] = 'allocated-v1'
    else:
        shot['production_slot'].pop('authoring_protocol')
    contract = compile_action_contract(shot)
    assert 'observed_effect_seconds' in contract.event_evidence_schema()['items']['properties']
    response = outcome_response()
    response['action_event_evidence'][0].update(observed_reaction_seconds=.1, result_visible=False)
    with pytest.raises(SemanticReviewUnavailableError):
        _apply_event_evidence_verdict(response, contract, response['action_event_evidence'],
                                    timestamps=[i / 10 for i in range(51)])
    report = response['event_assessment']
    assert report['status'] == 'unavailable' and report['timing']['status'] == 'invalid'
    assert any('result_visible' in x for x in report['content_issues'])


def test_unavailable_evidence_is_reused_without_new_model_call(tmp_path):
    shot = outcome_shot()
    response = outcome_response()
    response['action_event_evidence'][0]['observed_reaction_seconds'] = .1
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    for _ in range(2):
        with pytest.raises(SemanticReviewUnavailableError):
            reviewer.review(str(video), shot)
    assert len(sdk.calls) == 1


def test_original_video_and_wire_response_through_formal_reviewer(tmp_path, monkeypatch):
    record, story, _ = captured()
    source = Path(__file__).resolve().parents[1] / record['video']['path']
    if not source.is_file():
        pytest.skip('真实原片仅在原验收工作区；固定JSON回放仍在所有环境执行')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record['video']['sha256']
    wire = next(json.loads(v['text']) for k, v in record['sources'].items() if '/responses/' in k)['attempts'][0]
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model=wire['model'], choices=[SimpleNamespace(
            finish_reason=c['finish_reason'], message=SimpleNamespace(content=c['content'],
            refusal='captured refusal' if c['refused'] else None)) for c in wire['choices']])
    # 上面的全局离线帧替身只供其他单元测试；此测试恢复本地真实抽帧和测时。
    from tools.frame_extractor import extract_frame
    from tools.ffmpeg_ops import get_video_duration
    monkeypatch.setattr('pipeline.semantic_review.extract_frame', extract_frame)
    monkeypatch.setattr('pipeline.semantic_review.get_video_duration', get_video_duration)
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-captured-response',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    review = reviewer.review(str(source), story['shots'][0])
    assert not review.accepted and not review.action_sequence_valid
    assert '椅面' in review.failure_reason and 'result_visible' in review.failure_reason
    assert not reviewer.review(str(source), story['shots'][0]).accepted
    assert len(calls) == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record['video']['sha256']


@pytest.mark.parametrize('mode', ['direct_contact', 'directed_path', 'area_effect', 'indirect_effect'])
def test_observation_scope_preserves_distinct_physical_modes(tmp_path, mode):
    from test_sequence_review import process_shot, event_response
    shot = process_shot(mode)
    shot['production_slot'].update(ordinary_shot()['production_slot'])
    for beat in shot['action_beats']:
        if beat['activity'] == 'interaction':
            beat['outcomes'] = [{'entity': beat['target'], 'state': beat['visible_result']}]
    response = event_response()
    response['creative_brief_valid'] = True
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    assert reviewer.review(str(video), shot).accepted
    assert mode in sdk.calls[0]['messages'][1]['content'][0]['text']
    fields = sdk.calls[0]['response_format']['json_schema']['schema']['properties']['action_event_evidence']['items']['properties']
    assert 'active_intervals' not in fields


def test_damaged_unavailable_cache_does_not_trigger_another_call(tmp_path):
    response = outcome_response()
    response['action_event_evidence'][0]['observed_reaction_seconds'] = .1
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    with pytest.raises(SemanticReviewUnavailableError):
        reviewer.review(str(video), outcome_shot())
    next((tmp_path / 'semantic_reviews').glob('*.evidence.json')).write_text('[]')
    with pytest.raises(SemanticReviewUnavailableError, match='损坏'):
        reviewer.review(str(video), outcome_shot())
    assert len(sdk.calls) == 1


def test_changed_contract_does_not_reuse_an_unavailable_observation(tmp_path):
    shot = outcome_shot()
    response = outcome_response()
    response['action_event_evidence'][0]['observed_reaction_seconds'] = .1
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    with pytest.raises(SemanticReviewUnavailableError):
        reviewer.review(str(video), shot)
    shot['action_beats'][0]['outcomes'][0]['state'] = 'stands upright beside the platform'
    sdk.response = outcome_response()
    assert reviewer.review(str(video), shot).accepted
    assert len(sdk.calls) == 2


@pytest.mark.parametrize('policy', [
    {'temporal_evidence': 'causal-order-v1', 'min_interaction_seconds': 1},
    {'temporal_evidence': 'causal-order-v1', 'result_hold_seconds': .5},
    {'temporal_evidence': 'invented-scope'},
])
def test_inconsistent_timing_scope_fails_closed(policy):
    from pipeline.performance import requires_measured_timing
    with pytest.raises(ValueError, match='范围'):
        requires_measured_timing(policy)


@pytest.mark.parametrize('bad_assessment', [[], {'timing': []}, {'timing': None}])
def test_malformed_measurement_cache_does_not_create_an_implicit_retry(tmp_path, bad_assessment):
    response = outcome_response()
    response['action_event_evidence'][0]['observed_reaction_seconds'] = .1
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    with pytest.raises(SemanticReviewUnavailableError):
        reviewer.review(str(video), outcome_shot())
    path = next((tmp_path / 'semantic_reviews').glob('*.evidence.json'))
    record = json.loads(path.read_text())
    record['event_assessment'] = bad_assessment
    path.write_text(json.dumps(record))
    with pytest.raises(SemanticReviewUnavailableError, match='损坏'):
        reviewer.review(str(video), outcome_shot())
    assert len(sdk.calls) == 1
