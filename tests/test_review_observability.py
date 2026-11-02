"""Unknown visual evidence pauses the original take, not a new paid generation."""
import json

import pytest

from pipeline.semantic_review import SemanticReview, SemanticReviewUnavailableError
from test_performance_outcomes import outcome_shot, outcome_response
from test_sequence_review import reviewer_for, offline_frames


@pytest.mark.parametrize('field', ['identity_continuity_valid', 'boundary_continuity_valid',
    'blocking_valid', 'visual_invariants_valid', 'primary_action_completed', 'creative_brief_valid'])
def test_unobservable_dimension_is_not_retried_or_declared_a_failed_video(tmp_path, field):
    response = outcome_response()
    response[field] = None
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    original = video.read_bytes()
    kwargs = {}
    if field == 'boundary_continuity_valid':
        tail = tmp_path / 'previous.jpg'
        tail.write_bytes(b'accepted previous frame')
        kwargs['previous_frame_path'] = str(tail)
    with pytest.raises(SemanticReviewUnavailableError, match='看不清|观察不足'):
        reviewer.review(str(video), outcome_shot(), **kwargs)
    assert len(sdk.calls) == 1
    assert video.read_bytes() == original
    files = list((tmp_path / 'semantic_reviews').glob('*.json'))
    assert len(files) == 1 and files[0].name.endswith('.evidence.json')
    saved = json.loads(files[0].read_text())
    assert saved['status'] == 'unavailable'
    assert field in saved['unobservable_fields']


@pytest.mark.parametrize('kind', ['event', 'visibility'])
def test_unknown_nested_evidence_stops_without_reinterpreting_null_as_false(tmp_path, kind):
    response = outcome_response()
    if kind == 'event':
        response['action_event_evidence'][0]['effect_visible'] = None
    else:
        response['required_entities_visible'][0] = None
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    with pytest.raises(SemanticReviewUnavailableError, match='看不清|观察不足'):
        reviewer.review(str(video), outcome_shot())
    assert len(sdk.calls) == 1


def test_protocol_can_express_unknown_without_weakening_known_failure(tmp_path):
    response = outcome_response()
    response['identity_continuity_valid'] = False
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    assert not reviewer.review(str(video), outcome_shot()).accepted
    schema = sdk.calls[0]['response_format']['json_schema']['schema']['properties']
    assert schema['identity_continuity_valid']['type'] == ['boolean', 'null']
    assert schema['action_event_evidence']['items']['properties']['result_visible']['type'] == ['boolean', 'null']
    assert 'null' in sdk.calls[0]['messages'][0]['content']


def test_cached_unknown_identity_is_never_defaulted_to_true(tmp_path):
    video, reviewer, sdk = reviewer_for(tmp_path, outcome_response())
    assert reviewer.review(str(video), outcome_shot()).accepted
    cache = next((tmp_path / 'semantic_reviews').glob('*.json'))
    saved = json.loads(cache.read_text())
    saved['review']['identity_continuity_valid'] = None
    cache.write_text(json.dumps(saved))
    with pytest.raises(SemanticReviewUnavailableError):
        reviewer.review(str(video), outcome_shot())
    assert len(sdk.calls) == 1


def test_explicit_unknown_is_not_the_legacy_missing_field_default():
    response = outcome_response()
    response['required_entities_visible'] = {'a': True, 'b': True}
    response['identity_continuity_valid'] = None
    with pytest.raises(SemanticReviewUnavailableError):
        SemanticReview.from_dict(response)


@pytest.mark.parametrize('kind', ['previous', 'identity'])
def test_supplied_but_missing_reference_cannot_be_silently_dropped(tmp_path, kind):
    video, reviewer, sdk = reviewer_for(tmp_path, outcome_response())
    missing = str(tmp_path / 'missing.jpg')
    args = {'previous_frame_path': missing} if kind == 'previous' else {'identity_reference_paths': [missing]}
    with pytest.raises(SemanticReviewUnavailableError, match='参考'):
        reviewer.review(str(video), outcome_shot(), **args)
    assert not sdk.calls
