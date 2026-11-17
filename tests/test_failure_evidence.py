"""跨边界故障保真；所有传输均为本地替身，不触发付费请求。"""
import json
import socket
from types import SimpleNamespace as NS

import httpx
from openai import APIConnectionError, OpenAI
import pytest

from pipeline.planning_checkpoint import PlanningCheckpoint
from pipeline.models import RunOptions
from pipeline.run_state import RunWorkspace
from pipeline.structured_response import collect_structured_response


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *a: pytest.fail('network forbidden'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *a: pytest.fail('network forbidden'))


def workspace(tmp_path):
    return RunWorkspace.create(tmp_path, RunOptions(request='offline', paid_take_budget=0))


def test_real_sdk_connection_cause_survives_checkpoint_without_retry(tmp_path):
    calls = []
    def receive(request):
        calls.append(request)
        try:
            raise socket.gaierror(-2, 'secret-host.invalid')
        except OSError as cause:
            raise httpx.ConnectError('secret URL and key', request=request) from cause
    with OpenAI(api_key='secret-key', base_url='https://offline.invalid/v1', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(receive))) as sdk:
        w = workspace(tmp_path)
        checkpoint = PlanningCheckpoint(w, resume=False)
        with pytest.raises(APIConnectionError):
            checkpoint.step('shot_004', {}, lambda: sdk.chat.completions.create(
                model='offline', messages=[{'role': 'user', 'content': 'offline'}]))
        entry = json.loads(checkpoint.path.read_text())['steps'][0]
        evidence = entry['failure']
        assert evidence['category'] == 'transport'
        assert evidence['remote_outcome'] == 'unknown'
        assert [c['type'] for c in evidence['chain']] == ['APIConnectionError', 'ConnectError', 'gaierror']
        assert evidence['chain'][-1]['errno'] == -2
        assert 'secret' not in json.dumps(evidence)
        with pytest.raises(ValueError, match='失败或请求结果不明'):
            PlanningCheckpoint(w, resume=True).step('shot_004', {}, lambda: pytest.fail('reissued'))
        assert len(calls) == 1 and not w.manifest.paid_take_budget.reservations


@pytest.mark.parametrize('secondary', ['save', 'close', 'both'])
def test_stream_cleanup_and_diagnostic_failure_never_replace_primary(secondary):
    primary = httpx.ReadError('original private message')
    class Stream:
        closed = False
        def __iter__(self):
            raise primary
        def close(self):
            self.closed = True
            if secondary in ('close', 'both'):
                raise OSError('private cleanup path')
    def save(record):
        if secondary in ('save', 'both'):
            raise OSError('private disk path')
    stream = Stream()
    with pytest.raises(httpx.ReadError) as caught:
        collect_structured_response(stream, on_failure=save)
    assert caught.value is primary and stream.closed
    from pipeline.operation_failure import failure_evidence
    record = failure_evidence(primary)
    assert record['phase'] == 'receive_response'
    assert len(record['secondary']) == (2 if secondary == 'both' else 1)
    assert 'private' not in json.dumps(record)


def test_failed_checkpoint_write_preserves_operation_error_and_unknown_state(tmp_path, monkeypatch):
    w = workspace(tmp_path)
    checkpoint = PlanningCheckpoint(w, resume=False)
    original = checkpoint._write
    def write(value):
        if value['steps'][-1]['state'] == 'failed':
            raise OSError('disk full')
        original(value)
    monkeypatch.setattr(checkpoint, '_write', write)
    primary = httpx.ReadTimeout('unknown remote result')
    def operation():
        raise primary
    with pytest.raises(httpx.ReadTimeout) as caught:
        checkpoint.step('review', {}, operation)
    assert caught.value is primary
    assert json.loads(checkpoint.path.read_text())['steps'][0]['state'] == 'started'
    with pytest.raises(ValueError, match='结果不明'):
        PlanningCheckpoint(w, resume=True).step('review', {}, lambda: pytest.fail('reissued'))


def test_successful_stream_with_failed_close_is_not_silently_accepted():
    class Stream:
        def __iter__(self):
            yield NS(choices=[NS(delta=NS(content='{}', refusal=None), finish_reason='stop')])
        def close(self):
            raise OSError('close failed')
    with pytest.raises(OSError):
        collect_structured_response(Stream())


def test_http_failure_is_not_conflated_with_transport_and_never_persists_body():
    from openai import BadRequestError
    from pipeline.operation_failure import failure_evidence, failure_message
    response = httpx.Response(400, request=httpx.Request('POST', 'https://secret.invalid?key=secret'),
                              headers={'x-request-id': 'test-request-123'})
    exc = BadRequestError('sensitive provider message', response=response,
                          body={'error': {'message': 'sensitive body'}})
    record = failure_evidence(exc)
    assert record['category'] == 'http' and record['http_status'] == 400
    assert record['remote_outcome'] == 'response_received'
    assert record['request_id'] == 'test-request-123'
    assert 'sensitive' not in json.dumps(record) + failure_message(exc)
    assert 'secret' not in json.dumps(record)


def test_cyclic_and_deep_exception_chains_are_bounded():
    from pipeline.operation_failure import failure_evidence
    first = RuntimeError('first')
    first.__cause__ = first
    assert failure_evidence(first)['chain_truncated'] is True
    current = first = RuntimeError('first')
    for _ in range(30):
        current.__cause__ = RuntimeError('private')
        current = current.__cause__
    record = failure_evidence(first)
    assert len(record['chain']) == 8 and record['chain_truncated']


def test_cancellation_survives_stream_cleanup():
    import asyncio
    class Stream:
        def __iter__(self):
            raise asyncio.CancelledError()
        def close(self):
            raise OSError('close failed')
    with pytest.raises(asyncio.CancelledError):
        collect_structured_response(Stream())


def test_run_record_keeps_primary_even_when_diagnostics_and_status_cannot_save(tmp_path, monkeypatch):
    from pipeline.operation_failure import record_run_failure, failure_evidence
    w = workspace(tmp_path)
    def fail(*a):
        raise OSError('disk unavailable')
    monkeypatch.setattr(w, '_atomic_write_json', fail)
    monkeypatch.setattr(w, 'mark_failed', fail)
    primary = httpx.ReadError('primary')
    record_run_failure(w, primary)
    assert [v['phase'] for v in failure_evidence(primary)['secondary']] == ['save_run_failure', 'save_run_status']


def test_resume_refusal_cannot_overwrite_original_run_failure(tmp_path):
    from pipeline.operation_failure import record_run_failure
    w = workspace(tmp_path)
    record_run_failure(w, httpx.ReadTimeout('original'))
    before = (w.path / 'run_failure.json').read_bytes()
    record_run_failure(w, ValueError('resume refused'))
    assert (w.path / 'run_failure.json').read_bytes() == before


@pytest.mark.asyncio
async def test_real_three_returned_shots_and_fourth_connection_failure_stop_formal_flow(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    import config
    from pipeline.orchestrator import VideoPipeline
    data = json.loads((Path(__file__).parent / 'fixtures/real_planning_connection_failure.json').read_text())
    assert data['failure']['key'] == 'shot_004' and data['failure']['error_type'] == 'APIConnectionError'
    raw_responses = []
    for file in data['files'].values():
        assert hashlib.sha256(file['text'].encode()).hexdigest() == file['sha256']
        raw_responses.append(json.loads(file['text']))
    calls = []
    class Client:
        def __init__(self):
            self.chat = self.completions = self
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) > len(raw_responses):
                # 真实旧记录未保存原因链，只重放已知APIConnectionError，不虚构DNS等原因。
                raise APIConnectionError(request=httpx.Request('POST', 'https://offline.invalid'))
            raw = raw_responses[len(calls)-1]
            return NS(choices=[NS(message=NS(content=raw['content'], refusal=None),
                                 finish_reason=raw['finish_reason'])])
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: Client())
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('video reached'))
    pipeline = VideoPipeline(style=data['style_input'], paid_take_budget=0,
                             take_policy='single_pass', planning_repair=False)
    with pytest.raises(APIConnectionError):
        await pipeline.run(data['request'])
    checkpoint_path = pipeline.workspace / 'planning_checkpoint.json'
    before = checkpoint_path.read_bytes()
    steps = json.loads(before)['steps']
    assert [s['state'] for s in steps] == ['returned']*5 + ['failed']
    assert steps[-1]['failure']['chain'] == [{'type': 'APIConnectionError', 'errno': None}]
    assert steps[-1]['failure']['phase'] == 'request'
    assert len(calls) == 6 and not pipeline.run_workspace.manifest.paid_take_budget.reservations
    failure = json.loads((pipeline.workspace / 'run_failure.json').read_text())
    assert failure['failure'] == steps[-1]['failure']
    assert '远端处理结果未知' in pipeline.run_workspace.manifest.error
    with pytest.raises(ValueError, match='失败或请求结果不明'):
        await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert len(calls) == 6 and checkpoint_path.read_bytes() == before


@pytest.mark.parametrize('brief', ['机器人装配零件', '演员双人舞蹈', '母女重聚'])
@pytest.mark.asyncio
async def test_formal_pipeline_faults_keep_identical_boundaries_across_topics(tmp_path, monkeypatch, brief):
    import config
    from pipeline.orchestrator import VideoPipeline
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    primary = httpx.ReadTimeout('sensitive read error')
    async def fail(self, request):
        raise primary
    monkeypatch.setattr(VideoPipeline, '_run', fail)
    pipeline = VideoPipeline(paid_take_budget=0, take_policy='single_pass')
    with pytest.raises(httpx.ReadTimeout) as caught:
        await pipeline.run(brief)
    assert caught.value is primary
    record = json.loads((pipeline.workspace / 'run_failure.json').read_text())
    assert record['failure']['category'] == 'transport'
    assert record['failure']['remote_outcome'] == 'unknown'
    assert pipeline.run_workspace.manifest.status.value == 'failed'
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations
