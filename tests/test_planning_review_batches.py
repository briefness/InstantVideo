"""Long plans use bounded reviews, with every batch boundary and unit accounted for."""
from copy import deepcopy
import json

import pytest

from pipeline.planning_review import ensure_planning_semantics
from test_planning_semantics import compiled, verdict_for


def long_story():
    story = compiled()
    original = story['shots'][0]
    story['production_plan']['version'] = 'production-plan-v8-request-shape'
    story['shots'] = [{**deepcopy(original), 'shot_id': i} for i in range(1, 41)]
    return story


class Reviewer:
    def __init__(self, *, fail_batch=None):
        self.chat = self.completions = self
        self.calls = []
        self.fail_batch = fail_batch

    def create(self, **kwargs):
        from types import SimpleNamespace
        from test_action_result_flow import annotated_verdict
        from planning_protocol_cases import wire_verdict
        payload = json.loads(kwargs['messages'][-1]['content'])
        self.calls.append(payload)
        if len(self.calls) == self.fail_batch:
            raise RuntimeError('offline interruption')
        verdict = annotated_verdict(payload['storyboard'])
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps(wire_verdict(verdict, kwargs)), refusal=None))])


def test_large_plan_review_covers_all_units_and_boundaries_without_oversized_response(tmp_path, monkeypatch):
    story = long_story()
    client = Reviewer()
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: client)
    save = lambda name, data: (tmp_path / name).write_text(json.dumps(data))
    ensure_planning_semantics(story, tmp_path, save)
    assert len(client.calls) > 1
    actual = []
    for index, payload in enumerate(client.calls):
        batch = payload['storyboard']
        assert sum(len(s['action_beats']) for s in batch['shots']) <= 24
        actual.extend(s['shot_id'] for s in batch['shots'])
        scope = batch['review_scope']
        assert scope['is_final_batch'] == (index == len(client.calls) - 1)
        if index:
            assert scope['previous_shot']['shot_id'] == batch['shots'][0]['shot_id'] - 1
        else:
            assert scope['previous_shot'] is None
    assert actual == list(range(1, 41))
    count = len(client.calls)
    ensure_planning_semantics(story, tmp_path, save)
    assert len(client.calls) == count
    changed = deepcopy(story)
    changed['shots'][-1]['primary_action'] += ' changed'
    with pytest.raises(ValueError, match='不一致'):
        ensure_planning_semantics(changed, tmp_path, save)
    assert len(client.calls) == count


def test_interrupted_review_resumes_without_repeating_accepted_batches(tmp_path, monkeypatch):
    story = long_story()
    client = Reviewer(fail_batch=2)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: client)
    save = lambda name, data: (tmp_path / name).write_text(json.dumps(data))
    with pytest.raises(Exception, match='offline interruption|调用失败|LLM'):
        ensure_planning_semantics(story, tmp_path, save)
    first_ids = [s['shot_id'] for s in client.calls[0]['storyboard']['shots']]
    client.fail_batch = None
    ensure_planning_semantics(story, tmp_path, save)
    assert sum([s['shot_id'] for s in p['storyboard']['shots']] == first_ids for p in client.calls) == 1


def test_oversized_single_shot_does_not_cause_recursive_review(tmp_path, monkeypatch):
    story = long_story()
    story['shots'][0]['action_beats'] *= 10
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: pytest.fail('must stop before model call'))
    with pytest.raises(ValueError, match='动作单元'):
        ensure_planning_semantics(story, tmp_path, lambda *args: None)


def test_failed_boundary_batch_stops_later_reviews_and_remains_failed_on_resume(tmp_path, monkeypatch):
    from types import SimpleNamespace
    calls = []
    def create(**kwargs):
        batch = json.loads(kwargs['messages'][-1]['content'])['storyboard']
        calls.append(batch)
        verdict = verdict_for(batch)
        if len(calls) == 2:
            verdict.update(continuity_valid=False, reason='boundary state contradicts previous_shot')
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps(verdict), refusal=None))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: client)
    save = lambda name, data: (tmp_path / name).write_text(json.dumps(data))
    for _ in range(2):
        with pytest.raises(ValueError, match='boundary state contradicts'):
            ensure_planning_semantics(long_story(), tmp_path, save)
    assert len(calls) == 2
    assert not (tmp_path / 'planning_semantic_review.json').exists()


@pytest.mark.asyncio
async def test_formal_long_plan_preserves_duration_and_resumes_all_admission_batches(tmp_path, monkeypatch):
    import config
    from pathlib import Path
    from pipeline.orchestrator import VideoPipeline
    from test_director_intent import Responses, draft
    from test_request_shape import intent
    from authoring_cases import scene_authoring_from_legacy

    authored = scene_authoring_from_legacy(draft())
    original = authored['shots'][0]
    transition, interaction, _, hold = original['action_plan']
    original['action_plan'] = [transition, *[
        {**deepcopy(interaction), 'id': f'cause_{i}'} for i in range(6)], hold]
    authored['shots'] = [{**deepcopy(original), 'shot_id': i,
                         'story_state_after': f'training exchange {i} completed'} for i in range(1, 41)]
    value = intent(600, 40)
    brief = value.pop('source_request')
    author, reviewer = Responses(value, authored), Reviewer()
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: author)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: reviewer)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0)
    result = await pipeline.run(brief)
    story = json.loads(Path(result).read_text())
    assert len(story['shots']) == 40 and sum(s['duration'] for s in story['shots']) == 600
    assert len(author.calls) == 42
    assert len(reviewer.calls) == 14  # Three complete eight-unit shots per review.
    resumed = VideoPipeline.from_workspace(pipeline.workspace)
    resumed.plan_only = True
    assert await resumed.run() == result
    assert len(author.calls) == 42 and len(reviewer.calls) == 14
    assert not resumed.run_workspace.manifest.paid_take_budget.reservations
