"""Replay a real runaway response without contacting a model or video service."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import httpx
from openai import OpenAI
import pytest

from pipeline.storyboard import _call_llm_for_storyboard, _parse_json_response


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def deny(*args, **kwargs):
        pytest.fail('offline stream test attempted network access')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)


def real_content():
    record = json.loads((Path(__file__).parent / 'fixtures/real_repeated_field_failure.json').read_text())
    return record['sources']['planning_shot_001_response.json']['value']['content']


class Chunks:
    def __init__(self, text, *, width=64, finish='stop', fail=False):
        self.text, self.width, self.finish, self.fail = text, width, finish, fail
        self.consumed = 0
        self.closed = False
    def __iter__(self):
        for start in range(0, len(self.text), self.width):
            content = self.text[start:start + self.width]
            self.consumed += len(content)
            yield NS(choices=[NS(delta=NS(content=content, refusal=None), finish_reason=None)])
        if self.fail:
            raise OSError('offline interrupted transport')
        if self.finish is not None:
            yield NS(choices=[NS(delta=NS(content=None, refusal=None), finish_reason=self.finish)])
    def close(self):
        self.closed = True


def client_for(response):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return response
    return NS(chat=NS(completions=NS(create=create))), calls


@pytest.mark.parametrize('wrapper', ['{}', '```json\n{}\n```', 'result follows: {}'])
@pytest.mark.parametrize('payload', ['{"accepted":false,"accepted":true}',
    '{"a":{"target":"x","target":"y"}}', '{"target":1,"tar\\u0067et":2}'])
def test_duplicate_fields_never_silently_overwrite(wrapper, payload):
    with pytest.raises(ValueError, match='重复'):
        _parse_json_response(wrapper.format(payload))


def test_real_runaway_is_stopped_at_second_key_without_retry():
    text = real_content()
    assert len(text) == 50185 and text.count('"target":""') == 4013
    stream = Chunks(text)
    client, calls = client_for(stream)
    saved = []
    with pytest.raises(ValueError, match='重复'):
        _call_llm_for_storyboard(client, 'schema', 'request', response_schema={}, save_response=saved.append)
    assert len(calls) == 1 and calls[0]['stream'] is True
    assert stream.closed and stream.consumed < 2200
    assert len(saved) == 1 and saved[0]['content'] == text[:stream.consumed]
    assert saved[0]['finish_reason'] == 'protocol_error'


@pytest.mark.parametrize('width', [1, 2, 7, 64, 1024])
def test_chunk_boundaries_escaped_keys_nested_objects_and_repeated_values(width):
    text = '{"items":[{"a":"}, \\\"target\\\": 1"},{"a":"same"}],"内容":"same","object":{"\\u4e2d":"文"}}'
    stream = Chunks(text, width=width)
    client, calls = client_for(stream)
    assert _call_llm_for_storyboard(client, 'schema', 'request', response_schema={}) == json.loads(text)
    assert stream.closed and len(calls) == 1


@pytest.mark.parametrize('finish', ['length', 'content_filter', None])
def test_unfinished_stream_cannot_be_accepted_even_with_complete_json(finish):
    stream = Chunks('{"ok":true}', finish=finish)
    client, calls = client_for(stream)
    saved = []
    with pytest.raises(ValueError):
        _call_llm_for_storyboard(client, 'schema', 'request', response_schema={}, save_response=saved.append)
    assert stream.closed and len(calls) == 1
    assert len(saved) == 1 and saved[0]['content'] == stream.text


def test_transport_failure_closes_and_preserves_partial_response():
    stream = Chunks('{"partial":', fail=True)
    client, calls = client_for(stream)
    saved = []
    with pytest.raises(OSError, match='interrupted'):
        _call_llm_for_storyboard(client, 'schema', 'request', response_schema={}, save_response=saved.append)
    assert stream.closed and len(calls) == 1
    assert saved[0]['finish_reason'] == 'transport_error' and saved[0]['content'] == stream.text


def test_duplicate_completed_response_never_enters_legacy_json_repair():
    response = NS(choices=[NS(message=NS(content='{"ok":false,"ok":true}'), finish_reason='stop')])
    client, calls = client_for(response)
    with pytest.raises(ValueError, match='重复'):
        _call_llm_for_storyboard(client, 'json', 'request')
    assert len(calls) == 1


def test_real_sdk_sse_cancellation_closes_http_body_and_stops_reading():
    raw = real_content()
    calls = []
    class Body(httpx.SyncByteStream):
        consumed = 0
        closed = False
        def __iter__(self):
            for start in range(0, len(raw), 64):
                part = raw[start:start + 64]
                self.consumed += len(part)
                event = {'id': 'offline', 'object': 'chat.completion.chunk', 'created': 0,
                    'model': 'offline', 'choices': [{'index': 0, 'delta': {'content': part}, 'finish_reason': None}]}
                yield ('data: ' + json.dumps(event) + '\n\n').encode()
        def close(self):
            self.closed = True
    body = Body()
    def receive(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=body)
    with OpenAI(api_key='offline', base_url='https://offline.invalid/v1', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(receive))) as sdk:
        with pytest.raises(ValueError, match='重复'):
            _call_llm_for_storyboard(sdk, 'schema', 'request', response_schema={})
    assert calls[0]['stream'] is True and calls[0]['max_completion_tokens'] == 16384
    assert len(calls) == 1 and body.closed and body.consumed < 2200


@pytest.mark.parametrize('payload', [
    '{"target":1,"tar\\u0067et":2}',
    '{"items":[{"value":0},{"nested":{"a\\\"b":1,"a\\\"b":2}}]}',
])
def test_duplicate_detection_at_every_possible_chunk_split(payload):
    from pipeline.structured_response import ObjectKeyGuard
    for split in range(len(payload) + 1):
        guard = ObjectKeyGuard()
        with pytest.raises(ValueError, match='重复'):
            guard.feed(payload[:split])
            guard.feed(payload[split:])


def test_refusal_is_not_accepted_even_when_json_looks_valid():
    from pipeline.structured_response import collect_structured_response
    chunks = iter([NS(choices=[NS(delta=NS(content='{"ok":true}', refusal='declined'),
                                finish_reason='stop')])])
    saved = []
    with pytest.raises(ValueError, match='拒绝'):
        collect_structured_response(chunks, on_failure=saved.append)
    assert saved == [{'content': '{"ok":true}', 'finish_reason': 'protocol_error'}]


def test_usage_only_chunks_and_reasoning_do_not_pollute_content():
    from pipeline.structured_response import collect_structured_response
    usage = NS(completion_tokens=42)
    response = collect_structured_response(iter([
        NS(choices=[NS(delta=NS(content=None, reasoning_content='private reasoning'), finish_reason=None)]),
        NS(choices=[NS(delta=NS(content='{"ok":true}'), finish_reason='stop')]),
        NS(choices=[], usage=usage),
    ]))
    assert response.choices[0].message.content == '{"ok":true}'
    assert response.usage is usage


@pytest.mark.parametrize('text', ['', '{"partial":', '{"key":"value"'])
def test_incomplete_json_never_triggers_a_second_request(text):
    stream = Chunks(text)
    client, calls = client_for(stream)
    with pytest.raises(ValueError):
        _call_llm_for_storyboard(client, 'schema', 'request', response_schema={})
    assert len(calls) == 1 and stream.closed


@pytest.mark.asyncio
async def test_real_failure_replays_formal_plan_only_without_video_or_extra_model_calls(tmp_path, monkeypatch):
    import config
    from pipeline.orchestrator import VideoPipeline
    fixture = json.loads((Path(__file__).parent / 'fixtures/real_repeated_field_failure.json').read_text())
    sources = fixture['sources']
    # 只标注新版上游目标，不修改用来复现失败的身份和分镜响应。
    intent = json.loads(sources['planning_intent_response.json']['value']['content'])
    intent['completion_goals'] = [{'goal': intent['ending_requirement'], 'basis': 'user', 'evidence': intent['ending_evidence']}]
    responses = iter([
        NS(choices=[NS(message=NS(content=json.dumps(intent)),
                       finish_reason='stop')]),
        Chunks(sources['planning_identity_response.json']['value']['content']),
        Chunks(real_content()),
    ])
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return next(responses)
    sdk = NS(chat=NS(completions=NS(create=create)))
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: sdk)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: pytest.fail('incomplete draft reached review'))
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0, take_policy='single_pass')
    # The original identity is already inconsistent with the source requirement;
    # duplicate-key transport behavior remains independently covered above.
    with pytest.raises(ValueError, match=r'创作响应契约.*story_arc/resolution \(const\)'):
        await pipeline.run(sources['run_manifest.json']['value']['options']['request'])
    assert len(calls) == 2  # Intent and rejected identity; no retries or shot calls.
    assert pipeline.run_workspace.manifest.status.value == 'failed'
    assert pipeline.run_workspace.manifest.paid_take_budget.reservations == []
    saved = json.loads((pipeline.workspace / 'planning_identity_response.json').read_text())
    assert saved['content'] == sources['planning_identity_response.json']['value']['content']
    assert not (pipeline.workspace / 'planning_shot_001_response.json').exists()
    assert not (pipeline.workspace / 'storyboard.json').exists()


@pytest.mark.parametrize('status', [429, 500])
def test_formal_authoring_client_does_not_retry_service_errors(monkeypatch, status):
    import config
    from openai import APIStatusError
    from pipeline.storyboard import generate_storyboard
    calls = []
    def receive(request):
        calls.append(request)
        return httpx.Response(status, headers={'retry-after-ms': '1'},
                              json={'error': {'message': 'offline service failure'}})
    def client(**kwargs):
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(receive)))
    monkeypatch.setattr(config, 'ARK_API_KEY', 'offline')
    monkeypatch.setattr(config, 'ARK_BASE_URL', 'https://offline.invalid/v1')
    monkeypatch.setattr('pipeline.storyboard.OpenAI', client)
    with pytest.raises(ValueError, match='无法可靠解析') as failure:
        generate_storyboard('15秒，三段连续操作展示')
    assert isinstance(failure.value.__cause__, APIStatusError)
    assert len(calls) == 1
