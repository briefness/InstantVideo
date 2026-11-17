"""真实 length 失败与请求执行契约；禁止联网，替身不证明模型语义准确率。"""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from pipeline.observation_support import verify_support


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))


def failed_record():
    root = Path(__file__).parents[1] / 'output/20261104_152506_584689/semantic_reviews/support'
    return json.loads(next(root.glob('*.json')).read_text())


def inputs(record):
    cases = record['input']['payload']['cases']
    return ({'criteria': {k: c['requirement_context'] for k, c in cases.items()}},
            {'observations': {k: {'status': 'satisfied', 'observation': c['observation'],
                                 'sample_indices': c['sample_indices']} for k, c in cases.items()}})


def test_original_empty_length_response_is_preserved_and_never_retried(tmp_path):
    record = failed_record()
    catalog, raw = inputs(record)
    calls = []
    def create(**request):
        calls.append(request)
        choice = record['response']['choices'][0]
        assert choice['content'] == '' and choice['finish_reason'] == 'length'
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=choice['finish_reason'],
            message=SimpleNamespace(content=choice['content'], refusal=None))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    for _ in range(2):
        with pytest.raises(ValueError):
            verify_support(catalog, raw, client=client, model='doubao-seed-2.1-lite',
                           cache_dir=tmp_path, video_hash='captured')
    assert len(calls) == 1
    saved = json.loads(next((tmp_path / 'support').glob('*.json')).read_text())
    assert saved['response']['choices'][0]['finish_reason'] == 'length'


def test_support_is_a_finite_decision_task_without_hidden_reasoning(tmp_path):
    record = failed_record()
    catalog, raw = inputs(record)
    def create(**request):
        assert request['extra_body']['thinking']['type'] == 'disabled'
        fields = request['response_format']['json_schema']['schema']['properties']['decisions']['properties']
        assert all(v == {'type': 'string', 'enum': ['supported', 'contradicted', 'unknown']} for v in fields.values())
        assert request['max_completion_tokens'] <= 8192
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps({'decisions': {k: 'supported' for k in fields}}), refusal=None))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result, _ = verify_support(catalog, raw, client=client, model='doubao-seed-2.1-lite',
                              cache_dir=tmp_path, video_hash='captured')
    assert result == raw


@pytest.mark.asyncio
async def test_original_interrupted_run_resume_does_not_call_any_provider(tmp_path, monkeypatch):
    import shutil
    import hashlib
    from pipeline.orchestrator import VideoPipeline
    from pipeline.semantic_review import SemanticReviewUnavailableError
    source = Path(__file__).parents[1] / 'output/20261104_152506_584689'
    workspace = tmp_path / 'offline-copy'
    shutil.copytree(source, workspace)
    before = hashlib.sha256((source / 'shots/shot_001.mp4').read_bytes()).hexdigest()
    def forbidden(*_, **__):
        pytest.fail('resume must not call a model, generate or download')
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=forbidden)))
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: client)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: SimpleNamespace(
        generate=forbidden, query_task=forbidden, download_video=forbidden))
    with pytest.raises(SemanticReviewUnavailableError, match='已有未完成或失败核对'):
        await VideoPipeline.from_workspace(str(workspace)).run()
    after = json.loads((workspace / 'run_manifest.json').read_text())
    assert len(after['paid_take_budget']['reservations']) == 1
    assert not (workspace / 'final.mp4').exists()
    assert hashlib.sha256((workspace / 'shots/shot_001.mp4').read_bytes()).hexdigest() == before


@pytest.mark.parametrize('fault', ['empty', 'extra', 'invalid', 'unknown', 'truncated', 'refused'])
def test_finite_boundary_fails_closed_once(tmp_path, fault):
    from test_director_intent import Responses
    catalog = {'criteria': {'g': {'kind': 'completion', 'action': 'Deliver parcel.'}}}
    raw = {'observations': {'g': {'status': 'satisfied', 'observation': 'Recipient holds parcel.', 'sample_indices': [1]}}}
    calls = []
    def create(**request):
        calls.append(request)
        body = {'decisions': {'g': 'supported'}}
        if fault == 'empty': body['decisions'] = {}
        if fault == 'extra': body['decisions']['foreign'] = 'supported'
        if fault == 'invalid': body['decisions']['g'] = {'verdict': 'supported'}
        if fault == 'unknown': body['decisions']['g'] = 'unknown'
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='length' if fault == 'truncated' else 'stop',
            message=SimpleNamespace(content=json.dumps(body), refusal='no' if fault == 'refused' else None))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    for model in ('offline-v1', 'offline-v2'):
        with pytest.raises(ValueError):
            verify_support(catalog, raw, client=client, model=model, cache_dir=tmp_path, video_hash='same-film')
    assert len(calls) == 1


def test_large_source_names_and_catalog_fail_before_network(tmp_path):
    catalog = {'criteria': {'x' * 10000: {'kind': 'completion', 'action': 'Deliver parcel.'}}}
    raw = {'observations': {'x' * 10000: {'status': 'satisfied', 'observation': 'Delivered.', 'sample_indices': [1]}}}
    with pytest.raises(ValueError, match='容量'):
        verify_support(catalog, raw, client=None, model='offline', cache_dir=tmp_path, video_hash='offline')


@pytest.mark.parametrize('saved', [{}, {'input': []}, {'input': {'video_hash': 'x'}, 'state': 'invented'}])
def test_corrupt_history_cannot_enable_new_calls(tmp_path, saved):
    from pipeline.observation_support import assert_support_available
    directory = tmp_path / 'support'
    directory.mkdir()
    (directory / 'history.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='历史记录格式'):
        assert_support_available(tmp_path, 'x')
