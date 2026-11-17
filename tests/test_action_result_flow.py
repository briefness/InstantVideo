"""Formal workflow replay with annotated SDK responses; no claims of model accuracy."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.causality import compile_action_contract
from pipeline.planning_review import validate_planning_verdict
from test_action_results import annotated_story, scene_shot
from test_director_intent import Responses, intent_reply
from test_planning_semantics import verdict_for
from test_sequence_review import offline_frames


def annotated_verdict(story):
    verdict = verdict_for(story)
    by_id = {s['shot_id']: s for s in story['shots']}
    for unit in verdict['units']:
        beat = by_id[unit['shot_id']]['action_beats'][unit['beat_index']]
        if beat.get('outcomes'):
            unit.update(outcome_contract_valid=True, evidence_fields=['action', 'outcomes'])
            if beat.get('scene_changes'):
                unit['evidence_fields'].append('scene_changes')
    return verdict


@pytest.mark.asyncio
@pytest.mark.parametrize('brief,action,target_state,prop,prop_state', [
    ('三段展示工人交接货物的连续动作', 'passes the parcel to the receiver', 'holds the parcel securely', 'Parcel', 'in the receiver hands'),
    ('三段连续追逐与开门脱险', 'pulls the companion through the doorway', 'safely outside the room', 'Door', 'open'),
    ('三段展示两人交谈并递交信件的连续动作', 'hands the envelope to the listener', 'has received the envelope', 'Envelope', 'held by the listener'),
])
async def test_formal_plan_only_preserves_multiple_results_and_resumes_without_calls(
        tmp_path, monkeypatch, brief, action, target_state, prop, prop_state):
    import config
    from pipeline.orchestrator import VideoPipeline

    brief = '15秒，' + brief
    raw = annotated_story()
    raw['scene_entities'] = [prop, 'Dust']
    for shot in raw['shots']:
        shot.update(scene_description='Two people in a room with a visible doorway and props.',
                    key_props=[prop, 'Dust'])
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit.update(action=action, result=target_state, weight=3.)
                unit['resolution'].update(target_state=target_state, scene_changes=[
                    {'entity': prop, 'state': prop_state, 'importance': 'required'},
                    {'entity': 'Dust', 'state': 'briefly airborne', 'importance': 'incidental'}])
    author = Responses({**intent_reply(brief), 'choreography': 'flexible',
        'completion_goals': [{'goal': action + '; ' + target_state + '; ' + prop_state,
                              'basis': 'user', 'evidence': brief}],
        'requested_duration_seconds': 15, 'duration_evidence': '15秒',
        'requested_shot_count': 3, 'shot_count_evidence': '三段'}, raw)
    calls = []
    def review(**kwargs):
        from types import SimpleNamespace
        from pipeline.action_results import result_claims
        story = json.loads(kwargs['messages'][-1]['content'])['storyboard']
        calls.append(story)
        verdict = annotated_verdict(story)
        by_id = {s['shot_id']: s for s in story['shots']}
        for unit in verdict['units']:
            claims = result_claims(by_id[unit['shot_id']]['action_beats'][unit['beat_index']])
            for decision in (unit.get('result_admission') or {}).get('decisions', []):
                if claims[decision['source']]['entity'] == 'Dust':
                    decision.update(basis='direction', evidence='offline annotation: dust does not affect transfer or escape')
        return Responses(verdict).create(**kwargs)
    from types import SimpleNamespace
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: author)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=review))))
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda *a, **kw: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0)
    await pipeline.run(brief)
    assert len(author.calls) == 5 and len(calls) == 1
    from pipeline.action_results import result_claims
    for shot in calls[0]['shots']:
        assert shot['result_admission'] == 'review-required-v1'
        assert any(c['entity'] == 'Dust' for b in shot['action_beats'] for c in result_claims(b).values())
    admitted = json.loads((pipeline.workspace / 'result_admission.json').read_text())['execution_storyboard']
    for shot in admitted['shots']:
        contract = compile_action_contract(shot).review_projection
        event = contract['action_events'][0]
        criteria = contract['completion_contract']['criteria']
        function_keys = criteria[f"beat:{event['event_id']}/completion"]['supports']
        assert function_keys and any(prop_state in criteria[k]['action'] for k in function_keys)
        assert 'functional_responses' not in event  # One admitted goal owns completion.
        assert 'Dust' not in json.dumps(contract['completion_contract'])
        assert any(c['entity'] == 'Dust' for b in shot['action_beats'] for c in result_claims(b).values())
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations
    resumed = VideoPipeline.from_workspace(pipeline.workspace)
    resumed.plan_only = True
    await resumed.run()
    assert len(author.calls) == 5 and len(calls) == 1


def test_planning_reviewer_must_read_classification_and_cannot_ignore_missing_dependency():
    story = {'shots': [scene_shot()]}
    verdict = annotated_verdict(story)
    assert validate_planning_verdict(verdict, story) == []
    verdict['units'][0]['evidence_fields'].remove('scene_changes')
    with pytest.raises(ValueError, match='场景结果分类'):
        validate_planning_verdict(verdict, story)
    verdict = annotated_verdict(story)
    verdict.update(continuity_valid=False, reason='next shot relies on an optional scene change')
    assert validate_planning_verdict(verdict, story)
    verdict = annotated_verdict(story)
    verdict['units'][0].update(outcome_contract_valid=False, reason='user requirement demoted to incidental')
    assert validate_planning_verdict(verdict, story)


def test_continuation_cannot_drop_a_required_object_result(tmp_path):
    from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
    from pipeline.performance import outcome_text
    from pipeline.action_results import required_outcomes
    from test_continuation import scenario, Planner
    _, previous, tail = scenario(tmp_path)
    shot = scene_shot()
    shot.update(shot_id=2, scene_id='roof', continuity_from_previous='seamless')
    missing = outcome_text(shot['action_beats'][0]['outcomes'])
    with pytest.raises(ContinuationUnavailableError, match='结构无效'):
        prepare_continuation(shot, previous, str(tail), tmp_path, client=Planner(entry_visible_result=missing))
    expected = outcome_text(required_outcomes(shot['action_beats'][0]))
    planner = Responses({'compatible': True, 'entry_action': 'continues the approach', 'reason': 'compatible'})
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=planner)
    assert result['action_beats'][0]['scene_changes'] == shot['action_beats'][0]['scene_changes']
    assert result['action_beats'][0]['visible_result'] == expected


@pytest.mark.parametrize('required_visible', [True, False])
def test_visual_review_consumes_classified_contract_and_invalidates_changed_cache(tmp_path, monkeypatch, required_visible):
    from test_sequence_review import reviewer_for
    from test_performance_outcomes import outcome_response
    response = outcome_response()
    response['action_event_evidence'][0]['result_visible'] = required_visible
    video, reviewer, sdk = reviewer_for(tmp_path, response)
    shot = scene_shot()
    assert reviewer.review(str(video), shot).accepted is required_visible
    assert reviewer.review(str(video), shot).accepted is required_visible
    assert len(sdk.calls) == 1
    shot['action_beats'][0]['scene_changes'][1]['importance'] = 'required'
    reviewer.review(str(video), shot)
    assert len(sdk.calls) == 2


def test_real_old_response_cannot_slip_through_new_formal_protocol(monkeypatch):
    from pipeline.storyboard import generate_storyboard
    from pipeline.authoring_contract import AuthoringContractError
    raw = json.loads((Path(__file__).parent / 'fixtures/real_outcome_ownership_failure.json').read_text())
    before = deepcopy(raw)
    identity = {k: v for k, v in annotated_story().items() if k != 'shots'}
    identity['scene_entities'] = ['Door', 'Dust']
    client = Responses({**intent_reply('连续动作'), 'choreography': 'flexible'}, identity, raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    with pytest.raises(AuthoringContractError) as rejected:
        generate_storyboard('连续动作', target_duration=15)
    assert {'path': '/shot/action_plan', 'constraint': 'type'} in rejected.value.issues
    assert len(client.calls) == 3 and raw == before
