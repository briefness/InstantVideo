"""真实记录只读回放；合成场景验证权限、采样与执行投影，不证明视频质量。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.causality import compile_action_contract
from pipeline.event_assessment import assess_event_evidence
from pipeline.performance import measured_performance
from pipeline.result_observations import completion_contract
from pipeline.assessment_scope import local_assessment
from pipeline.observation_support import support_input
from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt


RUN = Path(__file__).parents[1] / 'output/20261117_122159_440065'
VERSION = 'intent-bounded-v1'
FIXTURE = json.loads((Path(__file__).parent / 'fixtures/real_sampling_authority_failure.json').read_text())


def captured():
    return deepcopy((FIXTURE['execution_storyboard'], FIXTURE['evidence'], FIXTURE['observation']))


def bounded(shot):
    return {**deepcopy(shot), 'execution_semantics': VERSION}


def test_real_record_stays_failed_but_sample_alias_is_not_malformed_response():
    story, evidence, _ = captured()
    old = compile_action_contract(story['shots'][0])
    args = dict(timestamps=evidence['sample_timestamps'], media_duration=evidence['media_duration'])
    assert assess_event_evidence(old, evidence['action_event_evidence'], **args)['timing']['status'] == 'invalid'
    new = compile_action_contract(bounded(story['shots'][0]))
    report = assess_event_evidence(new, evidence['action_event_evidence'], **args)
    assert report['status'] == 'rejected'  # 不用时间歧义覆盖已记录的内容否定，也不改成通过。
    assert report['timing']['causal_order']['status'] == 'uncertain'
    assert report['content_issues'] == evidence['event_assessment']['content_issues']


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_observation_authority_excludes_authored_paths_but_generation_keeps_them(topic):
    from test_creation_schedule_flow import scenario_story
    from test_staging_direction import compile_story
    # 编译真实模块；目标目录是独立的合成标注，不冒充模型审查。
    _, _, raw, plan = scenario_story(topic)
    story = compile_story(raw, plan)
    for shot in story['shots']:
        assert shot['execution_semantics'] == VERSION
        from pipeline.goal_allocation import evidence_unit
        catalog = {'execution_semantics': VERSION, 'criteria': {'goal': {
            'kind': 'completion', 'action': f'完成{topic}',
            'beat_ids': list(range(1, len(shot['action_beats']) + 1)),
            'execution_scope': {'shot_id': shot['shot_id'], 'units': [
                evidence_unit(s['shot_id'], i, b) for s in story['shots']
                for i, b in enumerate(s['action_beats'])]}}}}
        view = local_assessment(catalog)
        for criterion in view['criteria'].values():
            for unit in criterion.get('execution_scope', {}).get('units', []):
                assert 'action' not in unit
        # 拍摄仍得到所有作者动作；审查排除的是表演授权，并不是删动作。
        prompt = compile_normal_provider_prompt(shot, story, has_observed_start=False)
        for beat in shot['action_beats']:
            assert beat['action'] in prompt


def test_real_review_receives_goals_and_local_owners_not_left_hand_choreography():
    story, _, raw = captured()
    shot = bounded(story['shots'][0])
    catalog = completion_contract(shot)
    before = deepcopy(catalog)
    view = local_assessment(catalog)
    payload = support_input(catalog, raw)
    assert '左手拔出' not in json.dumps(view, ensure_ascii=False)
    assert '左手拔出' not in json.dumps(payload, ensure_ascii=False)
    assert '左手拔出' in json.dumps(catalog, ensure_ascii=False)
    assert catalog == before
    for key, criterion in view['criteria'].items():
        if criterion.get('kind') == 'contribution':
            original = catalog['criteria'][key]
            assert criterion['goal_context']['action'] == original['action']
            assert criterion['actors'] == original['actors']
            assert criterion['targets'] == original['targets']
            assert criterion['beat_ids'] == original['beat_ids']


def test_flexible_request_has_no_per_gesture_clock_but_preserves_paths_and_dependencies():
    story, _, _ = captured()
    old = story['shots'][0]
    actual = FIXTURE['prompt']
    assert compile_normal_provider_prompt(old, story, has_observed_start=False) == actual
    shot = bounded(old)
    normal = compile_normal_provider_prompt(shot, story, has_observed_start=False)
    for beat in shot['action_beats']:
        assert beat['action'] in normal
    assert 'begin after Beat 3 finishes' in normal  # 不擅自把串行动作改成并行。
    for prompt in (
        normal,
        compile_policy_safe_prompt(shot, storyboard=story, has_state_reference=False,
                                   image_role=None, reference_count=0),
    ):
        assert '2-3.3s' not in prompt and '3.3-4.6s' not in prompt
        assert 'performer_tempo=fast' in prompt and 'playback=real_time' in prompt


def evidence(effect=1., reaction=1.):
    return [{'event_id': 1, 'observed_start_seconds': 0., 'observed_effect_seconds': effect,
             'observed_reaction_seconds': reaction, 'observed_end_seconds': 2.,
             'active_intervals': [{'start_seconds': 0., 'end_seconds': 2.}]}]


@pytest.mark.parametrize('effect,reaction,status', [(1., 1., 'uncertain'),
    (1., 1.25, 'passed'), (1.25, 1., 'failed'), (1.1, 1.15, 'uncertain')])
def test_sample_resolution_defines_order_without_epsilon_or_invented_precision(effect, reaction, status):
    policy = {'duration': 3., 'min_interaction_seconds': 0., 'result_hold_seconds': 0.,
              'time_basis': 'sample-bounds-v1'}
    report = measured_performance(evidence(effect, reaction), policy, [i/4 for i in range(13)])
    assert report['causal_order']['status'] == status
    assert report['status'] == status


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), True, '1', -1, 9])
def test_invalid_time_still_fails_closed(invalid):
    with pytest.raises((ValueError, TypeError)):
        measured_performance(evidence(invalid), {'duration': 3., 'min_interaction_seconds': 0.,
            'result_hold_seconds': 0., 'time_basis': 'sample-bounds-v1'}, [0., 1., 2., 3.])


def test_evidence_and_original_video_unchanged():
    if not RUN.exists():
        pytest.skip('原片仅保留在本地；固定 JSON 证据回放仍在所有环境执行')
    for name, digest in FIXTURE['source_sha256'].items():
        assert hashlib.sha256((RUN/name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((RUN/'shots/shot_001.mp4').read_bytes()).hexdigest() == \
        '9cb3fa9c854f3ffd8748890c0089143a0e74ab3f0ff6e0944dcb7cfaebaf9048'


@pytest.mark.parametrize('explicit', ['必须用左手完成交接', '必须放在指定桌面', '必须等待信号后启动'])
def test_explicit_user_constraints_survive_scope_projection(explicit):
    story, _, _ = captured()
    catalog = completion_contract(bounded(story['shots'][0]))
    key = next(k for k, c in catalog['criteria'].items() if 'execution_scope' in c)
    catalog['criteria'][key]['action'] = explicit
    catalog['criteria']['required_surface'] = {'kind': 'state', 'beat_id': 3,
        'entity': 'Parcel', 'state': 'rests on the workbench, not the seat'}
    view = local_assessment(catalog)
    assert view['criteria'][key]['goal_context']['action'] == explicit
    assert view['criteria']['required_surface'] == catalog['criteria']['required_surface']


def test_sampling_refinement_resolves_order_only_with_new_distinguishable_observations():
    policy = {'duration': 3., 'min_interaction_seconds': 0., 'result_hold_seconds': 0.,
              'time_basis': 'sample-bounds-v1'}
    coarse = measured_performance(evidence(1.1, 1.15), policy, [0, 1, 2, 3])
    fine = measured_performance(evidence(1.1, 1.15), policy, [0, 1, 1.1, 1.15, 2, 3])
    assert coarse['status'] == 'uncertain' and fine['status'] == 'passed'
    assert measured_performance(evidence(1.1, 1.1), policy, [0, 1, 1.1, 1.15, 2, 3])['status'] == 'uncertain'


@pytest.mark.parametrize('dependency,expected', [('after_effects', 'uncertain'), ('after_ends', 'failed'), (None, 'passed')])
def test_partial_order_and_overlapping_independent_events(dependency, expected):
    first = evidence(1, 1.25)[0]
    second = {**evidence(1, 1.25)[0], 'event_id': 2}
    policy = {'duration': 3., 'min_interaction_seconds': 0., 'result_hold_seconds': 0.,
        'time_basis': 'sample-bounds-v1', 'event_order': [
            {'event_id': 1, 'after_ends': [], 'after_effects': []},
            {'event_id': 2, 'after_ends': [], 'after_effects': []}]}
    if dependency:
        policy['event_order'][1][dependency] = [1]
    result = measured_performance([first, second], policy, [i/4 for i in range(13)])
    assert result['status'] == expected
    assert result['interaction_seconds'][1] <= 2.25  # 重叠不能重复计时。


@pytest.mark.parametrize('content_failure', [True, False])
def test_original_pixels_replay_through_reviewer_and_resume_without_new_calls(tmp_path, content_failure):
    if not (RUN/'shots/shot_001.mp4').exists():
        pytest.skip('原片仅保留在本地；固定 JSON 证据回放仍在所有环境执行')
    from types import SimpleNamespace as NS
    from pipeline.semantic_review import SemanticTakeReviewer, SemanticReviewUnavailableError
    from test_full_production_flow import completion
    story, _, raw = captured()
    original = deepcopy(raw)
    shot = bounded(story['shots'][0])
    support = deepcopy(FIXTURE['support'])
    if not content_failure:
        # 独立合成标注只隔离时间分辨率路径，不是给真实片段改判。
        for row in support['decisions'].values():
            row.update(requirement='fulfilled', issue='none', finding='Offline annotation of a visible functional result.')
    calls = []
    def create(**request):
        calls.append(request)
        assert '左手拔出' not in json.dumps(request['messages'], ensure_ascii=False)
        return completion(raw if len(calls) == 1 else support)
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-bounded',
        client=NS(chat=NS(completions=NS(create=create))))
    for _ in range(2):
        if content_failure:
            result = reviewer.review(str(RUN/'shots/shot_001.mp4'), shot, identity_crop_entities=shot['characters'])
            assert not result.accepted
        else:
            with pytest.raises(SemanticReviewUnavailableError, match='语义不确定'):
                reviewer.review(str(RUN/'shots/shot_001.mp4'), shot, identity_crop_entities=shot['characters'])
    assert len(calls) == 2 and raw == original
    if not content_failure:
        record = json.loads(next(reviewer.cache_dir.glob('*.evidence.json')).read_text())
        assert record['failure_kind'] == 'semantic_uncertainty'
        assert record['event_assessment']['timing']['causal_order']['status'] == 'uncertain'


@pytest.mark.parametrize('relation,expected', [('same_sample', 'uncertain'), ('reversed', 'failed')])
def test_film_delivery_keeps_sample_order_reason_and_never_trusts_cached_pass(tmp_path, relation, expected):
    from dataclasses import asdict
    from pipeline.run_state import RunWorkspace
    from pipeline.generator import ShotResult
    from pipeline.film_rhythm import ensure_observed_film_rhythm
    from pipeline.semantic_review import SemanticReviewUnavailableError
    from test_performance_outcomes import outcome_shot, outcome_response
    shot = bounded(outcome_shot())
    shot['production_slot'].update(min_interaction_seconds=0, result_hold_seconds=0)
    contract = compile_action_contract(shot)
    events = outcome_response()['action_event_evidence']
    events[0]['observed_reaction_seconds'] = events[0]['observed_effect_seconds'] if relation == 'same_sample' else .1
    video = tmp_path / 'synthetic.mp4'
    video.write_bytes(b'Independent offline ledger fixture, not a generated-video observation.')
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    cache = tmp_path/'semantic_reviews'/f'{digest}.json'
    cache.parent.mkdir()
    RunWorkspace._atomic_write_json(cache, {'video_hash': digest, 'action_contract': asdict(contract),
        'review': {'accepted': True}, 'action_event_evidence': events,
        'media_duration': 5, 'sample_timestamps': [i/4 for i in range(21)]})
    result = ShotResult(shot_id=shot['shot_id'], status='success', local_path=str(video),
        semantic_accepted=True, accepted_contract_fingerprint=contract.fingerprint())
    error = SemanticReviewUnavailableError if expected == 'uncertain' else RuntimeError
    with pytest.raises(error, match='不自动重拍'):
        ensure_observed_film_rhythm({'shots': [shot], 'production_plan': {'min_interaction_seconds': 0}}, [result], tmp_path)
    report = json.loads((tmp_path/'film_rhythm_review.json').read_text())
    assert report['status'] == expected
    assert report['clips'][0]['measurement']['causal_order']['status'] == expected


@pytest.mark.parametrize('semantics,expected', [(None, 'unavailable'), (VERSION, 'rejected')])
def test_recorded_protocol_controls_diagnostic_priority_without_migrating_history(semantics, expected):
    from test_performance_outcomes import outcome_shot, outcome_response
    shot = outcome_shot()
    if semantics:
        shot['execution_semantics'] = semantics
    shot['production_slot'].update(min_interaction_seconds=4.4, result_hold_seconds=0)
    events = outcome_response()['action_event_evidence']
    events[0]['result_visible'] = False
    report = assess_event_evidence(compile_action_contract(shot), events,
                                  timestamps=[i / 4 for i in range(21)])
    assert report['timing']['status'] == 'uncertain'
    assert report['status'] == expected
    assert report['content_issues']
