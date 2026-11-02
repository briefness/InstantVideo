"""Raw provider responses keep the same integrity rules across model entrypoints."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.continuation import _request_plan, REVIEWED_VERSION, ContinuationUnavailableError
from pipeline.director_intent import resolve_director_intent
from pipeline.semantic_review import SemanticTakeReviewer, SemanticReviewUnavailableError
from test_semantic_review import _accepted_review_json


class RawCompletions:
    def __init__(self, content, finish='stop', refusal=None):
        self.content, self.finish, self.refusal = content, finish, refusal
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert len(self.calls) == 1, 'integrity failure must not request another model response'
        choice = SimpleNamespace(message=SimpleNamespace(content=self.content, refusal=self.refusal))
        if self.finish is not None:
            choice.finish_reason = self.finish
        return SimpleNamespace(choices=[choice])


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_a, **_k: pytest.fail('unexpected network call'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_a, **_k: pytest.fail('unexpected network call'))


def test_intent_cannot_overwrite_a_prior_classification():
    raw = ('{"content_focus":"action","content_focus":"balanced","pacing":"balanced",'
           '"intent_evidence":"测试需求","ending_requirement":"","ending_evidence":""}')
    client = RawCompletions(raw)
    diagnostics = {}
    with pytest.raises(ValueError, match='无法可靠解析'):
        resolve_director_intent(client, '测试需求', save_diagnostic=diagnostics.__setitem__)
    assert len(client.calls) == 1
    assert diagnostics['planning_intent_response.json']['content'] == raw


@pytest.mark.parametrize('last_key', ['compatible', r'compati\u0062le'])
def test_continuation_cannot_overwrite_incompatible_with_compatible(last_key):
    raw = ('{"compatible":false,"entry_action":"unchanged",'
           '"reason":"PRIVATE_UNTRUSTED_REASON","' + last_key + '":true}')
    client = RawCompletions(raw)
    with pytest.raises(ContinuationUnavailableError) as error:
        _request_plan({'version': REVIEWED_VERSION, 'entry_result_bound': True}, client)
    assert len(client.calls) == 1 and 'PRIVATE_UNTRUSTED_REASON' not in str(error.value)


@pytest.fixture
def visual_review(tmp_path, monkeypatch):
    video = tmp_path / 'take.mp4'
    video.write_bytes(b'unchanged raw take')
    monkeypatch.setattr('pipeline.semantic_review.extract_frame',
                        lambda _video, output, timestamp=None: Path(output).write_bytes(b'frame'))
    monkeypatch.setattr('pipeline.semantic_review.get_video_duration', lambda _path: 5.0)
    def run(client):
        return SemanticTakeReviewer(tmp_path, client=client, model='offline-test').review(
            str(video), {'shot_id': 1, 'required_visible_entities': ['combat_robot']})
    return run


@pytest.mark.parametrize('finish', ['length', None, 'tool_calls', 'PRIVATE_UNTRUSTED_REASON'])
def test_complete_json_is_not_a_complete_visual_response(tmp_path, visual_review, finish):
    client = RawCompletions(_accepted_review_json(), finish=finish)
    with pytest.raises(SemanticReviewUnavailableError) as error:
        visual_review(client)
    assert len(client.calls) == 1 and 'PRIVATE_UNTRUSTED_REASON' not in str(error.value)
    assert not list((tmp_path / 'semantic_reviews').glob('*.json'))
    assert (tmp_path / 'take.mp4').read_bytes() == b'unchanged raw take'


@pytest.mark.parametrize('last_key', ['identity_continuity_valid', r'identity_continuity_vali\u0064'])
def test_visual_response_cannot_overwrite_negative_judgment(tmp_path, visual_review, last_key):
    raw = _accepted_review_json().replace('"identity_continuity_valid": true',
          '"identity_continuity_valid": false, "' + last_key + '": true')
    client = RawCompletions(raw)
    with pytest.raises(SemanticReviewUnavailableError, match='重复'):
        visual_review(client)
    assert len(client.calls) == 1
    assert not list((tmp_path / 'semantic_reviews').glob('*.json'))


def test_nested_observed_state_cannot_overwrite_physical_fact(visual_review):
    raw = _accepted_review_json().replace('"action_phase": "target locked"',
          '"action_phase": "not completed", "action_phase": "completed"')
    with pytest.raises(SemanticReviewUnavailableError, match='重复'):
        visual_review(RawCompletions(raw))


@pytest.mark.parametrize('count', [0, 2])
def test_visual_response_requires_one_unambiguous_completion(visual_review, count):
    client = RawCompletions(_accepted_review_json())
    def receive(**kwargs):
        response = client.create(**kwargs)
        response.choices *= count
        return response
    client.chat.completions = SimpleNamespace(create=receive)
    with pytest.raises(SemanticReviewUnavailableError, match='一个完整判断'):
        visual_review(client)
    assert len(client.calls) == 1


def test_untrusted_usage_metadata_does_not_enter_diagnostics(visual_review):
    client = RawCompletions(_accepted_review_json(), finish='length')
    def receive(**kwargs):
        response = client.create(**kwargs)
        response.usage = SimpleNamespace(completion_tokens='PRIVATE_UNTRUSTED_REASON')
        return response
    client.chat.completions = SimpleNamespace(create=receive)
    with pytest.raises(SemanticReviewUnavailableError) as error:
        visual_review(client)
    assert 'PRIVATE_UNTRUSTED_REASON' not in str(error.value)


@pytest.mark.parametrize('accepted', [True, False])
def test_unambiguous_completed_visual_response_keeps_business_verdict(visual_review, accepted):
    value = json.loads(_accepted_review_json())
    value['identity_continuity_valid'] = accepted
    review = visual_review(RawCompletions(json.dumps(value)))
    assert review.accepted is accepted


@pytest.mark.parametrize('failure', ['duplicate', 'incomplete'])
def test_visual_integrity_failure_preserves_raw_diagnostic_only(tmp_path, visual_review, failure):
    raw = (_accepted_review_json().replace('"identity_continuity_valid": true',
           '"identity_continuity_valid": false, "identity_continuity_valid": true'))
    finish = 'stop' if failure == 'duplicate' else 'length'
    if failure == 'incomplete':
        raw = _accepted_review_json()
    client = RawCompletions(raw, finish=finish)
    with pytest.raises(SemanticReviewUnavailableError):
        visual_review(client)
    files = list((tmp_path / 'semantic_reviews' / 'responses').glob('*/*.json'))
    assert len(files) == 1
    diagnostic = json.loads(files[0].read_text())
    assert diagnostic['attempts'][0]['choices'] == [
        {'content': raw, 'finish_reason': finish, 'refused': False}]
    assert len(client.calls) == 1 and not list((tmp_path / 'semantic_reviews').glob('*.json'))
    assert (tmp_path / 'take.mp4').read_bytes() == b'unchanged raw take'
    first = files[0].read_bytes()
    with pytest.raises(SemanticReviewUnavailableError):
        visual_review(RawCompletions(raw, finish=finish))
    assert files[0].read_bytes() == first
    assert len(list((tmp_path / 'semantic_reviews' / 'responses').glob('*/*.json'))) == 2


def test_existing_visual_format_retry_retains_both_received_attempts(tmp_path, visual_review):
    from test_semantic_review import SequencedCompletions, _completion
    completions = SequencedCompletions([_completion(''), _completion(_accepted_review_json())])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    assert visual_review(client).accepted is True
    files = list((tmp_path / 'semantic_reviews' / 'responses').glob('*/*.json'))
    assert len(files) == 1 and len(completions.calls) == 2
    attempts = json.loads(files[0].read_text())['attempts']
    assert [attempt['choices'][0]['content'] for attempt in attempts] == ['', _accepted_review_json()]


@pytest.mark.parametrize('failure', ['duplicate', 'incomplete'])
def test_continuation_integrity_failure_preserves_raw_without_plan(tmp_path, failure):
    from pipeline.continuation import prepare_continuation
    from test_continuation import scenario
    shot, previous, tail = scenario(tmp_path)
    raw = ('{"compatible":false,"entry_action":"unchanged","entry_visible_result":"unchanged",'
           '"reason":"raw diagnostic 原文", "compatible":true}')
    finish = 'stop' if failure == 'duplicate' else 'length'
    client = RawCompletions(raw, finish=finish)
    with pytest.raises(ContinuationUnavailableError):
        prepare_continuation(shot, previous, str(tail), tmp_path, client=client)
    files = list((tmp_path / 'continuations' / 'responses').glob('shot_002/*.json'))
    assert len(files) == 1
    diagnostic = json.loads(files[0].read_text())
    assert diagnostic['response']['choices'] == [
        {'content': raw, 'finish_reason': finish, 'refused': False}]
    assert not (tmp_path / 'continuations' / 'shot_002.json').exists()
    assert len(client.calls) == 1 and tail.read_bytes() == b'accepted-frame'


def test_response_record_excludes_provider_request_and_exception_objects():
    from pipeline.structured_response import completion_response_record
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
        message=SimpleNamespace(content='model content', refusal='private refusal text'))],
        request={'authorization': 'DO_NOT_PERSIST'}, exception=RuntimeError('DO_NOT_PERSIST'))
    record = completion_response_record(response, model='test-model')
    assert record == {'model': 'test-model', 'choices': [
        {'content': 'model content', 'finish_reason': 'stop', 'refused': True}]}
