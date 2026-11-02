"""Obtained scene judgments cannot silently become new model calls on recovery."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from pipeline.scene_binding import resolve_scene_bindings
from test_scene_binding import locations, review


class Client:
    def __init__(self, raw, error=None):
        self.raw, self.error, self.calls = raw, error, []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=self.raw, refusal=None))])


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_a, **_k: pytest.fail('unexpected network call'))


def save(directory):
    from pipeline.run_state import RunWorkspace
    return lambda name, value: RunWorkspace._atomic_write_json(directory / name, value)


@pytest.mark.parametrize('invalid', ['uncertain', 'invalid_anchor'])
def test_negative_or_invalid_binding_is_retained_without_rejudging(tmp_path, monkeypatch, invalid):
    raw = review([1, 1, 1])
    if invalid == 'uncertain':
        raw['bindings'][1]['certain'] = False
    else:
        raw['bindings'][1]['anchor_shot_id'] = 99
    client = Client(json.dumps(raw))
    monkeypatch.setattr('pipeline.scene_binding.OpenAI', lambda **kw: client)
    for _ in range(2):
        with pytest.raises(ValueError):
            resolve_scene_bindings(locations(), tmp_path, save(tmp_path))
    assert len(client.calls) == 1
    stored = json.loads((tmp_path / 'scene_binding_review.json').read_text())
    assert stored['verdict'] == raw and stored['application_status'] == 'pending'
    assert not (tmp_path / 'execution_storyboard.json').exists()


def test_pending_candidate_cannot_bypass_locked_shot_guard_on_recovery(tmp_path, monkeypatch):
    client = Client(json.dumps(review([1, 1, 1])))
    monkeypatch.setattr('pipeline.scene_binding.OpenAI', lambda **kw: client)
    for _ in range(2):
        with pytest.raises(ValueError, match='已提交镜头'):
            resolve_scene_bindings(locations(), tmp_path, save(tmp_path), locked_shot_ids={2})
    assert len(client.calls) == 1
    stored = json.loads((tmp_path / 'scene_binding_review.json').read_text())
    assert stored['application_status'] == 'pending'
    assert not (tmp_path / 'execution_storyboard.json').exists()


@pytest.mark.parametrize('legacy', [False, True])
def test_applied_binding_can_resume_existing_media(tmp_path, monkeypatch, legacy):
    client = Client(json.dumps(review([1, 1, 1])))
    monkeypatch.setattr('pipeline.scene_binding.OpenAI', lambda **kw: client)
    story = locations()
    bound = resolve_scene_bindings(story, tmp_path, save(tmp_path))
    path = tmp_path / 'scene_binding_review.json'
    stored = json.loads(path.read_text())
    if legacy:
        stored.pop('application_status', None)
        save(tmp_path)(path.name, stored)
    before = path.read_bytes()
    assert resolve_scene_bindings(story, tmp_path, save(tmp_path), locked_shot_ids={2, 3}) == bound
    assert path.read_bytes() == before and len(client.calls) == 1


@pytest.mark.parametrize('failure', ['malformed_json', 'duplicate_key', 'transport'])
def test_raw_failed_response_prevents_implicit_repeat(tmp_path, monkeypatch, failure):
    raw = '{broken' if failure == 'malformed_json' else '{"bindings": [], "bindings": []}'
    client = Client(raw, RuntimeError('offline transport failure') if failure == 'transport' else None)
    monkeypatch.setattr('pipeline.scene_binding.OpenAI', lambda **kw: client)
    with pytest.raises((ValueError, RuntimeError)):
        resolve_scene_bindings(locations(), tmp_path, save(tmp_path))
    raw_path = tmp_path / 'scene_binding_response.json'
    before = raw_path.read_bytes()
    if failure != 'transport':
        assert json.loads(before)['content'] == raw
    with pytest.raises(ValueError, match='重试'):
        resolve_scene_bindings(locations(), tmp_path, save(tmp_path))
    assert len(client.calls) == 1 and raw_path.read_bytes() == before
    assert not (tmp_path / 'execution_storyboard.json').exists()


def test_stale_candidate_does_not_trigger_model_or_mutate_original(tmp_path, monkeypatch):
    client = Client(json.dumps(review([1, 1, 1])))
    monkeypatch.setattr('pipeline.scene_binding.OpenAI', lambda **kw: client)
    original = locations()
    resolve_scene_bindings(original, tmp_path, save(tmp_path))
    changed = deepcopy(original)
    changed['shots'][1]['scene_description'] += ' changed location'
    with pytest.raises(ValueError, match='不一致'):
        resolve_scene_bindings(changed, tmp_path, save(tmp_path))
    assert len(client.calls) == 1
