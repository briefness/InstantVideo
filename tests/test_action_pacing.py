"""Offline regressions for intent, time occupancy and observed outcome handoff."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
from pipeline.production_plan import build_production_plan
from pipeline.readiness import storyboard_readiness_issues


@pytest.fixture(autouse=True)
def no_external_http(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx.Client, 'send', lambda *_a, **_k: pytest.fail('offline test attempted HTTP'))
    async def reject(*_a, **_k):
        pytest.fail('offline test attempted async HTTP')
    monkeypatch.setattr(httpx.AsyncClient, 'send', reject)


def timed_shot():
    return {'shot_id': 1, 'duration': 5, 'prompt_en': 'A coherent exchange',
        'scene_id': 'roof', 'primary_action': 'White counters the incoming blade',
        'interaction_geometry': {'effect_phase': 'active'},
        'production_slot': {'min_interaction_seconds': 3.5, 'result_hold_seconds': .5},
        'action_beats': [
            {'phase': 'trigger', 'actor': 'White', 'target': 'Black', 'action': 'approaches',
             'visible_result': 'in range', 'start_seconds': 0., 'end_seconds': 1., 'activity': 'transition'},
            {'phase': 'peak', 'actor': 'White', 'target': 'Black', 'action': 'counters',
             'visible_result': 'Black loses resistance', 'start_seconds': 1., 'end_seconds': 4.5, 'activity': 'interaction'},
            {'phase': 'aftermath', 'actor': 'White', 'target': 'Black', 'action': 'holds guard',
             'visible_result': 'Black stays down', 'start_seconds': 4.5, 'end_seconds': 5., 'activity': 'hold'}]}


def test_tiny_action_cannot_count_as_five_seconds_of_fighting():
    shot = timed_shot()
    shot['action_beats'][0]['end_seconds'] = 3.
    shot['action_beats'][1]['start_seconds'] = 3.
    issues = storyboard_readiness_issues({'shots': [shot]})
    assert any('有效交互时长' in issue for issue in issues)


@pytest.mark.parametrize('phase', ['peak', 'aftermath'])
def test_non_trigger_handoff_cannot_skip_incompatible_observation(tmp_path, phase):
    tail = tmp_path / 'tail.jpg'
    tail.write_bytes(b'actual-accepted-tail')
    shot = timed_shot()
    shot['continuity_from_previous'] = 'seamless'
    shot['interaction_geometry']['effect_phase'] = 'aftermath' if phase == 'aftermath' else 'active'
    shot['action_beats'] = [shot['action_beats'][1 if phase == 'peak' else 2]]
    previous = {'scene_id': 'roof', 'observed_end_state': {'action_phase': 'Black is still defending'}}
    response = {'compatible': False, 'entry_action': 'unchanged',
                'entry_visible_result': 'unchanged', 'reason': 'The required collapse has not happened'}
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: SimpleNamespace(
        choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps(response), refusal=None))]))))
    with pytest.raises(ContinuationUnavailableError):
        prepare_continuation(shot, previous, str(tail), tmp_path, client=client)


@pytest.mark.parametrize('field,value', [
    ('start_seconds', float('nan')), ('start_seconds', True), ('start_seconds', -.1),
    ('end_seconds', float('inf')), ('end_seconds', '5'), ('end_seconds', 6),
    ('end_seconds', 0), ('activity', 'camera'), ('activity', None),
])
def test_invalid_time_axis_fails_before_execution(field, value):
    from pipeline.action_timing import action_timing_issues
    shot = timed_shot()
    shot['action_beats'][1][field] = value
    assert action_timing_issues(shot)


def test_time_and_order_survive_normal_safe_review_and_fingerprint():
    from pipeline.causality import compile_action_contract
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    from pipeline.generator import VideoGenerator
    shot = timed_shot()
    contract = compile_action_contract(shot)
    assert '1-4.5s' in compile_normal_provider_prompt(shot, {}, has_observed_start=False)
    assert '1-4.5s' in compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0)
    assert [b['visible_result'] for b in contract.review_projection['action_beats']] == [
        b['visible_result'] for b in shot['action_beats']]
    changed = deepcopy(shot)
    changed['action_beats'][0]['end_seconds'] = .5
    changed['action_beats'][1]['start_seconds'] = .5
    assert VideoGenerator._compiled_contract_fingerprint(shot) != VideoGenerator._compiled_contract_fingerprint(changed)


def test_bad_timing_verdict_cannot_be_overridden_by_accepted_true():
    from pipeline.semantic_review import SemanticReview
    value = {'accepted': True, 'required_entities_visible': {}, 'action_geometry_valid': True,
             'primary_action_completed': True, 'observed_end_state': {}, 'action_timing_valid': False}
    result = SemanticReview.from_dict(value, require_timing=True)
    assert not result.accepted and '时间' in result.failure_reason
    del value['action_timing_valid']
    assert not SemanticReview.from_dict(value, require_timing=True).accepted


@pytest.mark.parametrize('changes', [
    {'action_beats': [None]}, {'action_beats': 'invalid'}, {'production_slot': [1]},
])
def test_malformed_time_axis_reports_issue_instead_of_crashing(changes):
    from pipeline.action_timing import action_timing_issues
    assert action_timing_issues({**timed_shot(), **changes})


def test_missing_phase_or_non_scalar_activity_cannot_enter_time_projection():
    from pipeline.action_timing import action_timing_issues, timing_projection
    for field, value in [('phase', None), ('activity', [])]:
        shot = timed_shot()
        shot['action_beats'][0][field] = value
        assert action_timing_issues(shot)
        assert timing_projection(shot) == {}


@pytest.mark.parametrize('failed_field', ['action_timing_valid', 'requested_outcome_valid'])
def test_reviewer_samples_each_window_and_persists_rejection(tmp_path, monkeypatch, failed_field):
    from pathlib import Path
    from pipeline.semantic_review import SemanticTakeReviewer
    video = tmp_path / 'sample.mp4'
    video.write_bytes(b'offline-recording-placeholder')
    samples, calls = [], []
    monkeypatch.setattr('pipeline.semantic_review.get_video_duration', lambda _: 5.)
    def frame(_video, path, timestamp=None):
        samples.append(timestamp)
        Path(path).write_bytes(b'offline-frame')
    monkeypatch.setattr('pipeline.semantic_review.extract_frame', frame)
    def create(**kwargs):
        calls.append(kwargs)
        evidence_schema = kwargs['response_format']['json_schema']['schema']['properties']['causal_sample_evidence']
        result = {'accepted': True, 'required_entities_visible': [],
                  'identity_continuity_valid': True, 'boundary_continuity_valid': True,
                  'action_geometry_valid': True, 'primary_action_completed': True,
                  'action_timing_valid': True, 'requested_outcome_valid': True,
                  'visual_invariants_valid': True,
                  'causal_sample_evidence': [
                      dict.fromkeys(evidence_schema['items']['properties'], False)
                      for _ in range(evidence_schema['minItems'])],
                  'observed_end_state': {'location': 'roof', 'subject': 'White and Black',
                                         'action_phase': 'mostly waiting'}, 'failure_reason': ''}
        result[failed_field] = False
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps(result), refusal=None))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    reviewer = SemanticTakeReviewer(tmp_path, client=client)
    shot = timed_shot()
    shot['production_slot']['required_outcome'] = 'Black stays down and does not rise again'
    review = reviewer.review(str(video), shot)
    assert not review.accepted and not getattr(review, failed_field)
    assert {0.5, 2.75, 4.75}.issubset(samples)
    schema = calls[0]['response_format']['json_schema']['schema']
    assert 'action_timing_valid' in schema['required']
    assert 'requested_outcome_valid' in schema['required']
    assert 'timestamped samples' in calls[0]['messages'][0]['content']
    assert not reviewer.review(str(video), shot).accepted
    assert len(calls) == 1


def test_completed_causal_storyboard_does_not_override_missing_user_outcome():
    from pipeline.semantic_review import SemanticReview
    value = {'accepted': True, 'required_entities_visible': {}, 'action_geometry_valid': True,
             'primary_action_completed': True, 'effect_path_valid': True,
             'reaction_causality_valid': True, 'narrative_state_change_valid': True,
             'observed_end_state': {}, 'requested_outcome_valid': False}
    assert not SemanticReview.from_dict(value, require_outcome=True).accepted
    del value['requested_outcome_valid']
    assert not SemanticReview.from_dict(value, require_outcome=True).accepted


@pytest.mark.asyncio
async def test_real_generator_blocks_peak_handoff_before_paid_reservation(tmp_path, monkeypatch):
    from pathlib import Path
    from pipeline.generator import VideoGenerator
    base = json.loads((Path(__file__).resolve().parents[1] / 'examples/contact_previs/storyboard.json').read_text())
    shot = base['shots'][0]
    shot.update(shot_id=2, scene_id='roof', continuity_from_previous='seamless')
    previous = {'shot_id': 1, 'scene_id': 'roof', 'observed_end_state': {
        'location': 'roof', 'subject': 'attacker and defender',
        'action_phase': 'attacker has lost the ability to strike'}}
    tail = tmp_path / 'accepted-tail.jpg'
    tail.write_bytes(b'offline-accepted-tail')
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        value = {'compatible': False, 'entry_action': 'unchanged',
                 'entry_visible_result': 'unchanged', 'reason': 'the actor cannot perform the planned peak'}
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps(value), refusal=None))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr('pipeline.continuation.OpenAI', lambda **_: client)
    generator = VideoGenerator(str(tmp_path), reserve_paid_take=lambda _: pytest.fail('reserved before compatibility'))
    with pytest.raises(ContinuationUnavailableError):
        await generator._generate_single_shot(shot, str(tail), previous, base)
    assert len(calls) == 1
    context = json.loads(calls[0]['messages'][1]['content'])
    assert context['entry_editable'] is False


@pytest.mark.parametrize('phase', ['peak', 'aftermath'])
def test_compatible_non_trigger_never_rewrites_action_or_timing(tmp_path, phase):
    shot = timed_shot()
    shot['continuity_from_previous'] = 'seamless'
    shot['action_beats'] = shot['action_beats'][1:] if phase == 'peak' else shot['action_beats'][2:]
    shot['interaction_geometry']['effect_phase'] = 'active' if phase == 'peak' else 'aftermath'
    previous = {'scene_id': 'roof', 'observed_end_state': {'action_phase': 'compatible actual endpoint'}}
    tail = tmp_path / 'tail.jpg'
    tail.write_bytes(b'accepted-tail')
    response = {'compatible': True, 'entry_action': 'must not replace existing action',
                'entry_visible_result': 'must not replace result', 'reason': 'already compatible'}
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps(response), refusal=None))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    changed = prepare_continuation(shot, previous, str(tail), tmp_path, client=client)
    assert changed['action_beats'] == shot['action_beats']
    assert changed['primary_action'] == shot['primary_action']
    assert changed['start_state']['action_phase'] == 'compatible actual endpoint'
    assert prepare_continuation(shot, previous, str(tail), tmp_path, client=client, allow_create=False) == changed
    assert len(calls) == 1
