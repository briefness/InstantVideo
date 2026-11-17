"""客户端、响应头与错误正文的追踪标识不能互相冒充或覆盖。"""
import json
from types import SimpleNamespace

import pytest

from pipeline.models import SubmissionDiagnosticState
from tools.submission_evidence import creation_diagnostic, record_submission_result


def test_distinct_correlation_sources_survive_persistence(tmp_path):
    exc = RuntimeError('do not persist the exception text')
    exc.request_id = 'sdk-client-123'
    exc.status_code = 400
    exc.response = SimpleNamespace(headers={'x-request-id': 'response-456'})
    exc.body = {'error': {'code': 'InputVideoSensitiveContentDetected.PrivacyInformation',
                         'message': "input video 'content[1]' may contain real person. Request id: body-789"}}
    diagnostic = creation_diagnostic(exc, phase='create_request')
    assert diagnostic['request_id'] == 'response-456'
    assert diagnostic['sdk_request_id'] == 'sdk-client-123'
    assert diagnostic['response_request_id'] == 'response-456'
    assert diagnostic['body_request_id'] == 'body-789'
    record_submission_result(tmp_path, 3, 1, {'status': 'failed', 'error_type': 'privacy',
                                           'submission_diagnostic': diagnostic})
    saved = json.loads((tmp_path / 'submission_results/shot_003_attempt_001.json').read_text())
    assert saved['submission_diagnostic']['body_request_id'] == 'body-789'
    assert saved['submission_diagnostic']['sdk_request_id'] == 'sdk-client-123'
    assert 'exception text' not in json.dumps(saved)


@pytest.mark.parametrize('body', [
    {'request_id': 'body-789'},
    {'RequestId': 'body-789'},
    {'error': {'request_id': 'body-789'}},
    {'message': 'Request id: body-789'},
])
def test_server_body_id_is_preferred_to_sdk_id_without_headers(body):
    exc = RuntimeError('offline')
    exc.request_id = 'sdk-client-123'
    exc.body = body
    diagnostic = creation_diagnostic(exc, phase='create_request')
    assert diagnostic['request_id'] == diagnostic['body_request_id'] == 'body-789'
    assert diagnostic['response_request_id'] is None
    assert diagnostic['sdk_request_id'] == 'sdk-client-123'


@pytest.mark.parametrize('bad_id', ['SECRET', 'https://private.example/?token=SECRET',
                                  'x' * 201, 'bad\nheader', '', 123, None])
def test_correlation_does_not_leak_secrets_or_partial_untrusted_tokens(monkeypatch, bad_id):
    import config
    monkeypatch.setattr(config, 'ARK_API_KEY_SEEDANCE', 'SECRET')
    exc = RuntimeError('SECRET')
    exc.request_id = bad_id
    exc.response = SimpleNamespace(headers={'x-request-id': bad_id})
    exc.body = {'error': {'request_id': bad_id}}
    diagnostic = creation_diagnostic(exc, phase='create_request')
    assert all(diagnostic[key] is None for key in (
        'request_id', 'sdk_request_id', 'response_request_id', 'body_request_id'))
    assert 'SECRET' not in json.dumps(diagnostic)


@pytest.mark.parametrize('message', ['Request id: SECRET',
    'Request id: https://private.example/?token=SECRET', 'Request id: ' + 'x' * 201, None, ['request']])
def test_message_only_correlation_does_not_capture_secret_or_url_prefix(monkeypatch, message):
    import config
    monkeypatch.setattr(config, 'ARK_API_KEY_SEEDANCE', 'SECRET')
    exc = RuntimeError('SECRET')
    exc.body = {'error': {'message': message}}
    diagnostic = creation_diagnostic(exc, phase='create_request')
    assert diagnostic['body_request_id'] is None
    assert diagnostic['request_id'] is None
    assert 'SECRET' not in json.dumps(diagnostic)


def test_legacy_diagnostic_has_no_invented_correlation_source():
    old = {'version': 'submission-evidence-v1', 'phase': 'create_request',
           'observed_at': '2026-11-09T09:21:11+00:00', 'request_id': 'legacy-123'}
    parsed = SubmissionDiagnosticState.model_validate(old)
    assert parsed.request_id == 'legacy-123'
    assert parsed.sdk_request_id is None
    assert parsed.response_request_id is None
    assert parsed.body_request_id is None
