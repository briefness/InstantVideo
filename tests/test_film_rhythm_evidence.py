"""Offline raw observations, never claims about real generated picture quality."""
import hashlib
import json
import pytest

from pipeline.causality import compile_action_contract
from pipeline.film_rhythm import ensure_observed_film_rhythm
from pipeline.generator import ShotResult
from pipeline.performance import measured_performance
from pipeline.semantic_review import SemanticReviewUnavailableError
from test_performance_outcomes import outcome_shot, outcome_response
from test_sequence_review import reviewer_for, offline_frames


def reviewed_film(tmp_path, *, minimum=7, spans=None):
    shots, results = [], []
    for i in (1, 2, 3):
        shot = outcome_shot()
        shot['shot_id'] = i
        shot['production_slot']['min_interaction_seconds'] = 0
        response = outcome_response()
        if spans:
            for event, span in zip(response['action_event_evidence'], spans[i - 1]):
                event['active_intervals'] = [dict(start_seconds=span[0], end_seconds=span[1])]
        video, reviewer, sdk = reviewer_for(tmp_path, response)
        video = video.rename(tmp_path / f'shot_{i}.mp4')
        video.write_bytes(f'offline video {i}'.encode())
        assert reviewer.review(str(video), shot).accepted
        assert len(sdk.calls) == 1
        results.append(ShotResult(shot_id=i, status='success', local_path=str(video),
            semantic_accepted=True, accepted_contract_fingerprint=compile_action_contract(shot).fingerprint()))
        shots.append(shot)
    return {'shots': shots, 'production_plan': {'min_interaction_seconds': minimum}}, results


def report(path):
    return json.loads((path / 'film_rhythm_review.json').read_text())


def test_unequal_shot_density_passes_at_film_level(tmp_path):
    # The entry has less than 70% interaction; the other two carry the pace.
    story, results = reviewed_film(tmp_path, spans=[[(.5, 1.), (3., 3.5)],
                                                  [(0., 2.8), (2.8, 4.4)],
                                                  [(0., 2.8), (2.8, 4.4)]])
    ensure_observed_film_rhythm(story, results, tmp_path)
    saved = report(tmp_path)
    assert saved['status'] == 'passed'
    assert saved['clips'][0]['interaction_seconds'][1] < 3.5
    assert saved['interaction_seconds'][0] >= 7


@pytest.mark.parametrize('minimum,status,error', [(13.6, 'failed', RuntimeError),
                                                 (12., 'uncertain', SemanticReviewUnavailableError)])
def test_film_admission_stops_on_insufficient_or_uncertain_actual_time(tmp_path, minimum, status, error):
    story, results = reviewed_film(tmp_path, minimum=minimum)
    before = [open(r.local_path, 'rb').read() for r in results]
    with pytest.raises(error, match='不自动重拍'):
        ensure_observed_film_rhythm(story, results, tmp_path)
    assert report(tmp_path)['status'] == status
    assert before == [open(r.local_path, 'rb').read() for r in results]


@pytest.mark.parametrize('mutation', ['missing', 'video_hash', 'contract', 'evidence', 'samples', 'hold'])
def test_saved_acceptance_cannot_hide_missing_or_changed_raw_evidence(tmp_path, mutation):
    story, results = reviewed_film(tmp_path)
    digest = hashlib.sha256(open(results[0].local_path, 'rb').read()).hexdigest()
    cache = tmp_path / 'semantic_reviews' / f'{digest}.json'
    saved = json.loads(cache.read_text())
    if mutation == 'missing':
        cache.unlink()
    else:
        if mutation == 'video_hash':
            saved['video_hash'] = 'different'
        elif mutation == 'contract':
            saved['action_contract']['phase'] = 'different'
        elif mutation == 'evidence':
            saved['action_event_evidence'][0]['effect_visible'] = False
        elif mutation == 'samples':
            saved['sample_timestamps'] = [True, 5.]
        else:
            saved['action_event_evidence'][-1]['observed_end_seconds'] = 5.
        cache.write_text(json.dumps(saved))
    with pytest.raises(SemanticReviewUnavailableError):
        ensure_observed_film_rhythm(story, results, tmp_path)
    assert report(tmp_path)['status'] == 'unavailable'


def test_saved_aggregate_is_not_an_authority(tmp_path):
    story, results = reviewed_film(tmp_path, minimum=13.6)
    for path in (tmp_path / 'semantic_reviews').glob('*.json'):
        saved = json.loads(path.read_text())
        saved['timing_assessment'] = {'status': 'passed', 'interaction_seconds': [5, 5]}
        path.write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match='不足'):
        ensure_observed_film_rhythm(story, results, tmp_path)


@pytest.mark.parametrize('change', [lambda r: r.reverse(), lambda r: r.pop(),
    lambda r: setattr(r[1], 'accepted_contract_fingerprint', ''),
    lambda r: setattr(r[1], 'local_path', r[0].local_path)])
def test_incomplete_or_reused_clips_cannot_count_twice(tmp_path, change):
    story, results = reviewed_film(tmp_path)
    change(results)
    with pytest.raises(SemanticReviewUnavailableError):
        ensure_observed_film_rhythm(story, results, tmp_path)


def test_trimmed_tail_is_not_counted_and_cannot_supply_result_hold():
    policy = dict(duration=6, min_interaction_seconds=0, result_hold_seconds=.5)
    evidence = outcome_response()['action_event_evidence']
    evidence[-1].update(observed_end_seconds=5.4,
        active_intervals=[{'start_seconds': 2.8, 'end_seconds': 5.4}])
    samples = [i / 10 for i in range(61)]
    raw = measured_performance(evidence, policy, samples)
    assert raw['status'] == 'passed'
    edited = measured_performance(evidence, policy, samples, edit_duration=5.)
    assert edited['status'] == 'failed'
    assert edited['interaction_seconds'][1] <= 5
    assert edited['interaction_seconds'][0] < raw['interaction_seconds'][0]


@pytest.mark.parametrize('duration', [0, -1, True, '5', float('nan'), float('inf'), 6])
def test_invalid_edit_window_is_rejected(duration):
    policy = compile_action_contract(outcome_shot()).review_projection['performance_policy']
    with pytest.raises(ValueError):
        measured_performance(outcome_response()['action_event_evidence'], policy,
                             [i / 10 for i in range(51)], edit_duration=duration)
