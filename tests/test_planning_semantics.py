"""Admission invariants; test verdicts are not claims of model accuracy."""
import json
from copy import deepcopy

import pytest

from pipeline.planning_review import ensure_planning_semantics, validate_planning_verdict
from pipeline.storyboard import _compile_storyboard_contract
from pipeline.action_authoring import compile_authored_storyboard
from authoring_cases import authoring_from_legacy
from test_director_intent import draft, Responses
from test_action_authoring import plan_for


def verdict_for(storyboard):
    from pipeline.action_results import result_claims
    return {'units': [{'shot_id': s['shot_id'], 'beat_index': i,
        'cause_and_participants_valid': True, 'classification_matches_description': True,
        **({'result_admission': {
            'decisions': [{'source': p, 'basis': 'response', 'minimal': True,
                           'evidence': 'synthetic test annotation: retain original obligation'}
                          for p in result_claims(b)],
            'requirements_preserved': True, 'continuous_performance': True,
            'reason': 'synthetic protocol annotation, not a model accuracy judgment'}}
           if s.get('result_admission') == 'review-required-v1' else {}),
        'evidence_fields': ['action', 'visible_result'], 'reason': 'test annotation: independent cause and response'}
        for s in storyboard['shots'] for i, b in enumerate(s['action_beats'])],
        'continuity_valid': True, 'requested_outcome_valid': True, 'reason': 'test annotation'}


def compiled():
    plan = plan_for()
    return _compile_storyboard_contract(compile_authored_storyboard(authoring_from_legacy(draft()), plan),
        '16:9', '720p', 'cinematic', production_plan=plan)


@pytest.mark.parametrize('mutate', [
    lambda v: v['units'].pop(),
    lambda v: v['units'].append(v['units'][0]),
    lambda v: v['units'][0].update(evidence_fields=['invented_field']),
    lambda v: v['units'][0].update(cause_and_participants_valid='true'),
])
def test_incomplete_fabricated_or_coerced_verdict_cannot_pass(mutate):
    story = compiled()
    raw = verdict_for(story)
    mutate(raw)
    with pytest.raises(ValueError):
        validate_planning_verdict(raw, story)


@pytest.mark.parametrize('accepted', [True, False])
def test_acceptance_and_rejection_are_both_cached_without_regeneration(tmp_path, monkeypatch, accepted):
    story = compiled()
    raw = verdict_for(story)
    raw['units'][0]['cause_and_participants_valid'] = accepted
    client = Responses(raw)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: client)
    save = lambda name, data: (tmp_path / name).write_text(json.dumps(data))
    for _ in range(2):
        if accepted:
            ensure_planning_semantics(story, tmp_path, save)
        else:
            with pytest.raises(ValueError, match='分镜语义验收未通过'):
                ensure_planning_semantics(story, tmp_path, save)
    assert len(client.calls) == 1
    assert client.calls[0]['response_format']['json_schema']['strict'] is True
    changed = deepcopy(story)
    changed['shots'][0]['primary_action'] += ' changed'
    with pytest.raises(ValueError, match='不一致'):
        ensure_planning_semantics(changed, tmp_path, save)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_formal_pipeline_rejects_semantic_failure_before_any_video_reservation(tmp_path, monkeypatch):
    import config
    from pipeline.orchestrator import VideoPipeline
    story = compiled()
    raw = verdict_for(story)
    raw['units'][0].update(classification_matches_description=False, reason='effect attribution contradicted')
    client = Responses(raw)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: client)
    monkeypatch.setattr('pipeline.orchestrator.generate_storyboard', lambda **kw: deepcopy(story))
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('paid stage reached'))
    pipeline = VideoPipeline(paid_take_budget=3)
    with pytest.raises(ValueError, match='分镜语义验收未通过'):
        await pipeline.run('15秒连续交互')
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations
    assert (pipeline.workspace / 'planning_semantic_review.json').is_file()
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_real_plan_and_real_review_replay_stop_formal_pipeline_before_payment(tmp_path, monkeypatch):
    from pathlib import Path
    import hashlib
    import config
    from pipeline.models import validate_storyboard
    from pipeline.orchestrator import VideoPipeline
    fixtures = Path(__file__).parent / 'fixtures'
    story = json.loads((fixtures / 'real_planning_semantic_failure.json').read_text())['storyboard']
    verdict = json.loads((fixtures / 'real_planning_semantic_verdict.json').read_text())['verdict']
    issues = validate_planning_verdict(verdict, story)
    assert len(issues) == 3
    assert all(label in '；'.join(issues) for label in ['Shot 1 动作 1', 'Shot 3 动作 2', 'Shot 3 动作 3'])
    hold = verdict['units'][-1]
    assert hold['classification_matches_description'] is True  # A labeled hold is legitimate.
    client = Responses(verdict)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: client)
    def replay(**kwargs):
        # Preserve the recorded verdict under its historical wire contract, not
        # as a fabricated response to today's evidence-bearing request.
        payload = {'version': 'planning-semantics-v2-field-evidence', 'model': config.LLM_MODEL,
                   'storyboard': validate_storyboard(story)}
        kwargs['save_diagnostic']('planning_semantic_review.json', {
            'fingerprint': hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            'verdict': verdict})
        return deepcopy(story)
    monkeypatch.setattr('pipeline.orchestrator.generate_storyboard', replay)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('paid stage reached'))
    pipeline = VideoPipeline(paid_take_budget=3)
    with pytest.raises(ValueError, match='分镜语义验收未通过'):
        await pipeline.run(story['production_plan']['director_intent']['source_request'])
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations
    evidence = json.loads((pipeline.workspace / 'planning_semantic_evidence.json').read_text())
    assert evidence['units'][0]['source_fields']['action'] == story['shots'][0]['action_beats'][0]['action']
    assert len(client.calls) == 0
