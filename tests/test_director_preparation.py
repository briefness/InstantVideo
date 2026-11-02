"""A reviewable production plan must not imply permission to spend on video."""
import json
from pathlib import Path

import pytest

from pipeline.llm_options import structured_output_options
from pipeline.storyboard import generate_storyboard
from pipeline.orchestrator import VideoPipeline
from authoring_cases import authoring_from_legacy
from test_authoring_transport import http_client
from test_director_intent import draft, intent_reply, Responses
from test_planning_semantics import compiled, verdict_for


def test_invalid_authoring_response_is_preserved_before_parsing(monkeypatch):
    from types import SimpleNamespace
    from pipeline.storyboard import _call_llm_for_storyboard
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='{broken'))])
    monkeypatch.setattr('pipeline.storyboard._create_storyboard_completion', lambda *a, **kw: response)
    saved = []
    with pytest.raises(ValueError, match='不是合法 JSON'):
        _call_llm_for_storyboard(None, 'system', 'user', repair_json=False, save_response=saved.append)
    assert len(saved) == 1 and saved[0]['content'] == '{broken'


def test_real_intent_mode_failure_remains_rejected_and_is_excluded_by_request_contract():
    from pipeline.director_intent import resolve_director_intent, director_intent_response_schema
    record = json.loads((Path(__file__).parent / 'fixtures/real_intent_mode_failure.json').read_text())
    raw = {k: v for k, v in record.items() if k not in ('source', 'sha256', 'model')}
    brief = raw['intent_evidence'] + raw['motion_control_evidence'] + raw['ending_evidence']
    client = Responses(raw)
    with pytest.raises(ValueError, match='未按关键词'):
        resolve_director_intent(client, brief)
    schema = client.calls[0]['response_format']['json_schema']['schema']
    assert schema == director_intent_response_schema(brief)
    for variant in schema['oneOf']:
        assert 'source_request' not in variant['properties']
        assert set(variant['required']) == set(variant['properties'])
        assert variant['additionalProperties'] is False
    text_mode, reference_mode = schema['oneOf']
    assert text_mode['properties']['motion_control']['const'] == 'text_guided'
    assert text_mode['properties']['motion_control_evidence']['const'] == ''
    assert reference_mode['properties']['motion_control']['const'] == 'reference_motion'
    assert raw['motion_control_evidence']  # Raw evidence is never cleared to force acceptance.


@pytest.mark.parametrize('brief', ['先入场。密集交锋！最后倒下。', 'No fighting. Show a quiet room.', '第一句\n第二句', '只保留一句'])
def test_intent_evidence_can_only_select_contiguous_original_sentences(brief):
    from pipeline.director_intent import director_intent_response_schema
    schema = director_intent_response_schema(brief)
    for variant in schema['oneOf']:
        choices = variant['properties']['intent_evidence']['enum']
        assert choices and all(choice and choice in brief for choice in choices)
        assert variant['properties']['ending_evidence']['enum'] == ['', *choices]


def test_intent_diagnostics_preserve_failure_without_provider_exception_secrets():
    from pipeline.director_intent import resolve_director_intent
    saved = {}
    bad = intent_reply('invented quote')
    with pytest.raises(ValueError):
        resolve_director_intent(Responses(bad), 'actual brief', save_diagnostic=saved.__setitem__)
    assert json.loads(saved['planning_intent_response.json']['content']) == bad
    assert saved['planning_intent_error.json']['type'] == 'ValidationError'
    assert 'input' not in saved['planning_intent_error.json']['fields'][0]
    class Failing:
        def __init__(self):
            self.chat = self.completions = self
        def create(self, **kwargs):
            raise RuntimeError('secret-test-token')
    saved.clear()
    with pytest.raises(ValueError):
        resolve_director_intent(Failing(), 'actual brief', save_diagnostic=saved.__setitem__)
    assert 'secret-test-token' not in json.dumps(saved)


def test_relative_action_weights_apply_to_the_entire_schedule():
    from pipeline.action_authoring import AuthoredShot, compile_action_units
    raw = authoring_from_legacy(draft())['shots'][0]
    raw['action_plan'].pop()
    for unit, weight in zip(raw['action_plan'], [1., 1.7, 2.3]):
        unit['weight'] = weight
    beats = compile_action_units(AuthoredShot.model_validate(raw), {
        'duration': 5, 'min_interaction_events': 2, 'min_interaction_seconds': 3.5,
    })
    assert [b['end_seconds'] - b['start_seconds'] for b in beats] == pytest.approx([1., 1.7, 2.3])


def test_reasoning_is_explicit_and_does_not_change_extraction_defaults():
    model = 'doubao-seed-2.1-lite'
    assert structured_output_options(model)['extra_body']['thinking']['type'] == 'disabled'
    assert structured_output_options(model, reasoning=True)['extra_body']['thinking']['type'] == 'enabled'
    assert structured_output_options('unknown-endpoint', reasoning=True) == {}


def test_creative_calls_enable_thinking_in_actual_http_request(monkeypatch):
    raw = authoring_from_legacy(draft())
    values = [intent_reply('continuous action'), {k: v for k, v in raw.items() if k != 'shots'}]
    values += [{'shot': {k: v for k, v in s.items() if k != 'shot_id'}} for s in raw['shots']]
    sdk, calls = http_client(monkeypatch, values)
    with sdk:
        generate_storyboard('continuous action', target_duration=15)
    assert calls[0]['thinking']['type'] == 'disabled'
    assert all(c['thinking']['type'] == 'enabled' for c in calls[1:])
    assert all(c['max_completion_tokens'] == 16384 for c in calls[1:])


@pytest.mark.asyncio
async def test_plan_only_validates_and_resumes_same_plan_without_video(tmp_path, monkeypatch):
    import config
    story = compiled()
    planning_calls = []
    def plan(**kwargs):
        planning_calls.append(kwargs)
        return story
    reviewer = Responses(verdict_for(story))
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.orchestrator.generate_storyboard', plan)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: reviewer)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('unexpected video generation'))
    pipeline = VideoPipeline(paid_take_budget=3, plan_only=True)
    path = await pipeline.run('15秒连续交互')
    assert path == str(pipeline.workspace / 'storyboard.json')
    assert pipeline.run_workspace.manifest.stage == 'planning_ready'
    assert pipeline.run_workspace.manifest.status.value == 'interrupted'
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations
    saved = json.loads((pipeline.workspace / 'storyboard.json').read_text())
    resumed = VideoPipeline.from_workspace(pipeline.workspace)
    resumed.plan_only = True
    assert await resumed.run() == path
    assert len(planning_calls) == 1 and len(reviewer.calls) == 1
    assert json.loads((pipeline.workspace / 'storyboard.json').read_text()) == saved
    assert not resumed.run_workspace.manifest.paid_take_budget.reservations
