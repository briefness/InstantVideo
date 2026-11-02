import json
from copy import deepcopy

import pytest

from authoring_cases import authoring_from_legacy
from test_director_intent import draft, intent_reply, Responses
from test_planning_semantics import verdict_for
from tools.prepare_authored_run import prepare_authored_run
from pipeline.orchestrator import VideoPipeline


def source():
    brief = '15秒持续动作交锋'
    return {'authored_by': 'offline test author', 'duration_seconds': 15,
            'intent': {**intent_reply(brief), 'source_request': brief},
            'storyboard': authoring_from_legacy(draft())}


@pytest.mark.asyncio
async def test_authored_import_keeps_attribution_and_uses_formal_semantic_admission(tmp_path, monkeypatch):
    original = source()
    before = deepcopy(original)
    workspace = prepare_authored_run(original, tmp_path)
    assert original == before
    assert workspace.options.paid_take_budget == 0
    assert not workspace.manifest.paid_take_budget.reservations
    assert json.loads((workspace.path / 'authored_input.json').read_text()) == original
    assert json.loads((workspace.path / 'authorship.json').read_text())['mode'] == 'supplied_action_script'
    story = workspace.load_storyboard()
    reviewer = Responses(verdict_for(story))
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: reviewer)
    monkeypatch.setattr('pipeline.orchestrator.generate_storyboard', lambda **kw: pytest.fail('must not reauthor'))
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('must not generate'))
    run = VideoPipeline.from_workspace(workspace.path)
    run.plan_only = True
    await run.run()
    assert len(reviewer.calls) == 1
    assert run.run_workspace.manifest.stage == 'planning_ready'


@pytest.mark.parametrize('change', ['author', 'actor', 'duration', 'intent', 'budget'])
def test_bad_authored_input_cannot_create_a_workspace(tmp_path, change):
    value = source()
    budget = 0
    if change == 'author': value['authored_by'] = ' '
    elif change == 'actor': value['storyboard']['shots'][0]['action_plan'][0]['actor'] = 'unknown'
    elif change == 'duration': value['duration_seconds'] = True
    elif change == 'intent': value['intent']['intent_evidence'] = 'invented'
    else: budget = -1
    with pytest.raises(ValueError):
        prepare_authored_run(value, tmp_path, paid_take_budget=budget)
    assert not list(tmp_path.iterdir())
