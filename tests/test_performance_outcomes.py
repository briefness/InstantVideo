"""Raw failures stay failures; new protocol examples are offline test annotations."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.causality import compile_action_contract
from pipeline.semantic_review import _apply_event_evidence_verdict, SemanticReviewUnavailableError
from test_performance_contract import performance_shot, observed_response
from test_sequence_review import reviewer_for, offline_frames
from pipeline.performance import measured_performance


def outcome_shot():
    shot = performance_shot()
    for beat in shot['action_beats']:
        if beat.get('activity') == 'interaction':
            beat['outcomes'] = [{'entity': beat['target'], 'state': 'remains uninjured and able to continue'}]
    return shot


def test_results_are_not_choreography_in_the_review_contract():
    shot = outcome_shot()
    for beat in shot['action_beats'][:2]:
        beat['action'] = 'suggested wrist thrust'
        beat['visible_result'] = 'suggested wrist thrust then sideways step'
        beat['interaction_geometry']['reaction_scope'] = beat['visible_result']
    contract = compile_action_contract(shot)
    assert contract.review_projection['performance_policy']['version'] == 'performance-v2-outcomes'
    event = contract.review_projection['action_events'][0]
    assert event['outcomes'] == shot['action_beats'][0]['outcomes']
    assert 'suggested wrist' not in json.dumps(contract.review_projection)
    assert 'suggested wrist thrust' in ' '.join(contract.prompt_parts)


def test_sparse_near_threshold_measurement_is_unavailable_not_a_retake_verdict():
    shot = outcome_shot()
    shot['production_slot'].update(min_interaction_seconds=3.5, result_hold_seconds=0)
    response = observed_response()
    for item, times in zip(response['action_event_evidence'], [(1.5, 3.1, 3.8, 3.8), (3., 3.8, 4.4, 4.988)]):
        item.update(zip(('observed_start_seconds', 'observed_effect_seconds', 'observed_reaction_seconds', 'observed_end_seconds'), times))
        item['active_intervals'] = [{'start_seconds': times[0], 'end_seconds': times[-1]}]
    with pytest.raises(SemanticReviewUnavailableError, match='时间证据'):
        _apply_event_evidence_verdict(response, compile_action_contract(shot), response['action_event_evidence'],
                                      timestamps=[0, 1, 1.5, 2, 3, 3.1, 3.8, 4.4, 4.988])


def test_actual_failed_records_are_preserved_and_still_fail_legacy_contract():
    data = json.loads((Path(__file__).parent / 'fixtures/real_flexible_performance_failure.json').read_text())
    contract = compile_action_contract(data['storyboard']['shots'][0])
    assert contract.review_projection['performance_policy']['version'] == 'performance-v1'
    for report in data['reports']:
        assert report['review']['accepted'] is False
        assert contract.event_evidence_issues(report['evidence'])


def outcome_response():
    response = observed_response()
    for event in response['action_event_evidence']:
        event['active_intervals'] = [{'start_seconds': event['observed_start_seconds'],
                                      'end_seconds': event['observed_end_seconds']}]
    return response


def test_new_protocol_through_reviewer_cache_and_safe_prompt(tmp_path):
    from pipeline.provider_prompt import compile_policy_safe_prompt
    shot = outcome_shot()
    video, reviewer, sdk = reviewer_for(tmp_path, outcome_response())
    assert reviewer.review(str(video), shot).accepted
    assert reviewer.review(str(video), shot).accepted
    assert len(sdk.calls) == 1
    schema = sdk.calls[0]['response_format']['json_schema']['schema']
    assert 'active_intervals' in schema['properties']['action_event_evidence']['items']['required']
    safe = compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0)
    assert 'remains uninjured and able to continue' in safe
    cache = next((tmp_path / 'semantic_reviews').glob('*.json'))
    stored = json.loads(cache.read_text())
    assert len(stored['sample_timestamps']) == 21
    stored.pop('sample_timestamps')
    cache.write_text(json.dumps(stored))
    with pytest.raises(SemanticReviewUnavailableError, match='采样来源'):
        reviewer.review(str(video), shot)
    assert len(sdk.calls) == 1


def test_unavailable_review_keeps_evidence_and_does_not_relabel_video(tmp_path):
    shot = outcome_shot()
    shot['production_slot']['min_interaction_seconds'] = 4.4
    video, reviewer, sdk = reviewer_for(tmp_path, outcome_response())
    with pytest.raises(SemanticReviewUnavailableError, match='时间证据'):
        reviewer.review(str(video), shot)
    files = list((tmp_path / 'semantic_reviews').glob('*.json'))
    assert len(files) == 1 and files[0].name.endswith('.evidence.json')
    evidence = json.loads(files[0].read_text())
    assert evidence['status'] == 'unavailable'
    assert evidence['sample_timestamps'] and evidence['action_event_evidence']
    assert video.is_file() and len(sdk.calls) == 1


def test_pause_inside_an_event_does_not_count_as_interaction():
    evidence = outcome_response()['action_event_evidence']
    evidence[0]['active_intervals'] = [{'start_seconds': .1, 'end_seconds': .3}]
    evidence[1]['active_intervals'] = [{'start_seconds': 3., 'end_seconds': 3.2}]
    policy = compile_action_contract(outcome_shot()).review_projection['performance_policy']
    measurement = measured_performance(evidence, policy, [i / 10 for i in range(51)])
    assert measurement['status'] == 'failed'
    assert measurement['interaction_seconds'][1] < 1.


@pytest.mark.parametrize('change', [
    lambda e: e[0].update(active_intervals=[{'start_seconds': True, 'end_seconds': 2}]),
    lambda e: e[0].update(active_intervals=[{'start_seconds': -.1, 'end_seconds': 2}]),
    lambda e: e[0].update(active_intervals=[{'start_seconds': 0, 'end_seconds': 6}]),
    lambda e: e[0].update(active_intervals=[{'start_seconds': 1, 'end_seconds': 2}, {'start_seconds': .5, 'end_seconds': 1.5}]),
    lambda e: e[0].update(observed_reaction_seconds=.1),
])
def test_invalid_observations_never_trigger_a_paid_retake(change):
    response = outcome_response()
    change(response['action_event_evidence'])
    with pytest.raises(SemanticReviewUnavailableError):
        _apply_event_evidence_verdict(response, compile_action_contract(outcome_shot()), response['action_event_evidence'],
                                      timestamps=[i / 10 for i in range(51)])


def test_outcome_review_cannot_ignore_identity_or_requested_kill(tmp_path):
    response = outcome_response()
    response.update(identity_continuity_valid=False, creative_brief_valid=False, requested_outcome_valid=False)
    shot = outcome_shot()
    shot['production_slot']['required_outcome'] = 'the opponent loses the ability to fight permanently'
    video, reviewer, _ = reviewer_for(tmp_path, response)
    assert not reviewer.review(str(video), shot).accepted


def test_planning_requires_explicit_review_of_outcomes():
    from pipeline.planning_review import validate_planning_verdict
    from test_planning_semantics import verdict_for
    story = {'shots': [outcome_shot()]}
    verdict = verdict_for(story)
    with pytest.raises(ValueError, match='独立结果'):
        validate_planning_verdict(verdict, story)
    for v, b in zip(verdict['units'], story['shots'][0]['action_beats']):
        if b.get('outcomes'):
            v.update(outcome_contract_valid=True, evidence_fields=['outcomes'])
    assert validate_planning_verdict(verdict, story) == []
    verdict['units'][0].update(outcome_contract_valid=False, reason='state repeats the suggested wrist path')
    assert validate_planning_verdict(verdict, story)


def test_new_continuation_preserves_outcomes_without_locking_old_maneuver(tmp_path):
    from test_continuation import scenario, Planner
    from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
    from pipeline.performance import outcome_text
    _, previous, tail = scenario(tmp_path)
    shot = outcome_shot()
    shot.update(shot_id=2, scene_id='roof', continuity_from_previous='seamless')
    expected = outcome_text(shot['action_beats'][0]['outcomes'])
    planner = Planner(entry_action='continues from the existing guard with a different approach', entry_visible_result=expected)
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=planner)
    assert result['action_beats'][0]['outcomes'] == shot['action_beats'][0]['outcomes']
    assert result['action_beats'][0]['visible_result'] == expected
    assert prepare_continuation(shot, previous, str(tail), tmp_path, client=planner, allow_create=False) == result
    assert len(planner.calls) == 1


def test_actual_media_duration_accounts_for_provider_frame_padding(tmp_path, monkeypatch):
    monkeypatch.setattr('pipeline.semantic_review.get_video_duration', lambda _: 5.088)
    shot = outcome_shot()
    shot['production_slot'].update(result_hold_seconds=0, min_interaction_seconds=3)
    response = outcome_response()
    response['action_event_evidence'][-1]['observed_end_seconds'] = 5.038
    response['action_event_evidence'][-1]['active_intervals'][-1]['end_seconds'] = 5.038
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    assert reviewer.review(str(video), shot).accepted
    assert reviewer.review(str(video), shot).accepted
    assert len(sdk.calls) == 1


@pytest.mark.parametrize('mode', ['direct_contact', 'directed_path', 'area_effect', 'indirect_effect'])
def test_outcome_projection_applies_to_non_combat_interactions(mode):
    from test_sequence_review import process_shot
    shot = process_shot(mode)
    shot['production_slot'].update(choreography='flexible')
    for beat in shot['action_beats']:
        if beat.get('activity') == 'interaction':
            beat['outcomes'] = [{'entity': beat['target'], 'state': beat['visible_result']}]
    contract = compile_action_contract(shot)
    assert contract.review_projection['performance_policy']['version'] == 'performance-v2-outcomes'
    assert all(e['interaction_geometry']['interaction_mode'] == mode for e in contract.review_projection['action_events'])


@pytest.mark.parametrize('outcomes', [
    [{'entity': 'unregistered', 'state': 'changed'}],
    [{'entity': 'attacker', 'state': 'ready'}],
    [{'entity': 'defender', 'state': 'ready'}, {'entity': 'defender', 'state': 'fallen'}],
    [{'entity': 'defender', 'state': ' '}],
])
def test_outcomes_must_have_real_unique_owners(outcomes):
    from pipeline.models import ActionBeatSpec
    beat = outcome_shot()['action_beats'][0]
    beat['outcomes'] = outcomes
    with pytest.raises(ValueError):
        ActionBeatSpec.model_validate(beat)


def test_changing_outcomes_cannot_reuse_accepted_review(tmp_path):
    shot = outcome_shot()
    video, reviewer, sdk = reviewer_for(tmp_path, outcome_response())
    assert reviewer.review(str(video), shot).accepted
    shot['action_beats'][0]['outcomes'][0]['state'] = 'unable to continue'
    sdk.response['action_event_evidence'][0]['result_visible'] = False
    assert not reviewer.review(str(video), shot).accepted
    assert len(sdk.calls) == 2


@pytest.mark.asyncio
async def test_formal_generator_preserves_submission_when_timing_is_unavailable(tmp_path, monkeypatch):
    from dataclasses import asdict
    from pipeline.generator import VideoGenerator
    from pipeline.run_state import RunWorkspace
    from pipeline.models import RunOptions
    from test_action_sequence import sequence_story
    story = sequence_story()
    shot = outcome_shot()
    shot['production_slot']['min_interaction_seconds'] = 4.4
    shot['production_slot'].update(shot_id=1, duration=5, allowed_effect_phases=['active'],
        requires_visible_result=True, coverage_roles=['interaction'], framing_family='wide',
        reference_policy='independent')
    story['shots'] = [shot]
    workspace = RunWorkspace.create(tmp_path, RunOptions(request='offline outcomes', paid_take_budget=2))
    workspace.save_storyboard(story)
    def progress(result):
        fields = asdict(result)
        fields.pop('character_ref_path')
        workspace.record_shot(**fields)
    generator = VideoGenerator(str(workspace.path), on_progress=progress,
        reserve_paid_take=workspace.reserve_paid_take,
        confirm_paid_take_submission=workspace.confirm_paid_take_submission,
        reconcile_paid_take=workspace.reconcile_paid_take)
    class API:
        supports_last_frame = True
        calls = 0
        async def generate(self, **kwargs):
            self.calls += 1
            assert self.calls == 1, 'uncertain evidence triggered another paid submission'
            kwargs['on_submitted']('offline-original')
            return {'status': 'succeeded', 'provider_task_id': 'offline-original', 'video_url': 'offline'}
        async def download_video(self, _url, destination):
            Path(destination).write_bytes(b'offline media')
    generator.api = API()
    _, generator.semantic_reviewer, sdk = reviewer_for(tmp_path, outcome_response())
    monkeypatch.setattr('pipeline.generator.check_video_quality',
        lambda _: {'pass': True, 'quality_score': 100, 'issues': []})
    with pytest.raises(SemanticReviewUnavailableError, match='时间证据'):
        await generator._generate_single_shot(shot, None, None, story)
    assert generator.api.calls == 1 and len(sdk.calls) == 1
    assert len(workspace.manifest.paid_take_budget.reservations) == 1
    restored = RunWorkspace.resume(workspace.path)
    assert restored.manifest.shots['1'].provider_task_id == 'offline-original'
    assert (workspace.path / 'shots/shot_001.mp4').is_file()
    assert not list((workspace.path / 'shots').glob('*rejected*'))
