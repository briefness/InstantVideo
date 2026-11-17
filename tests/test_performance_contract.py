"""Offline observations exercise the real compiler/reviewer, not generated-video claims."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.causality import compile_action_contract
from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
from pipeline.performance import OBSERVED_TIMES
from pipeline.production_plan import build_production_plan
from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
from test_action_sequence import sequence_story
from test_sequence_review import event_response, reviewer_for, offline_frames
from test_continuation import scenario, Planner


def performance_shot():
    shot = sequence_story()['shots'][0]
    shot['production_slot'] = {'choreography': 'flexible', 'min_interaction_seconds': 3.,
        'result_hold_seconds': .5, 'creative_brief': '持续借势变招，保持双方身份', 'narrative_function': 'progress'}
    return shot


def observed_response():
    response = event_response()
    response['creative_brief_valid'] = True
    # The first action finishes after its proposed 2.5 s window. Still coherent.
    for item, times in zip(response['action_event_evidence'], [(0., .6, 1.1, 2.8), (2.8, 3., 3.5, 4.4)]):
        item.update(zip(OBSERVED_TIMES, times))
    return response


def test_new_plan_preserves_legacy_and_reference_motion_fidelity():
    intent = {'source_request': '连续动作', 'intent_evidence': '连续动作', 'content_focus': 'action',
              'pacing': 'dense', 'choreography': 'flexible'}
    plan = build_production_plan('action', 15, intent=intent)
    assert plan['version'] == 'production-plan-v8-request-shape'
    assert all(s['choreography'] == 'flexible' and s['creative_brief'] == '连续动作' for s in plan['slots'])
    old = build_production_plan('action', 15, intent=intent, contract_version='production-plan-v5-payoff-events')
    assert all('choreography' not in s for s in old['slots'])
    exact = build_production_plan('action', 15, intent={**intent, 'motion_control': 'reference_motion',
                                                        'motion_control_evidence': '连续动作'})
    assert all(s['choreography'] == 'exact' for s in exact['slots'])


def test_normal_and_safe_compile_suggestions_without_exact_timestamps():
    shot = performance_shot()
    for prompt in (compile_normal_provider_prompt(shot, {}, has_observed_start=False),
                   compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0)):
        assert 'Choreography is performance-guided:' in prompt
        assert '1.2s' not in prompt and '3.2s' not in prompt and 'Timed action schedule:' not in prompt
        assert 'defender moves back' in prompt and 'attacker steps back' in prompt
    original = sequence_story()['shots'][0]
    assert '1.2s' in compile_normal_provider_prompt(original, {}, has_observed_start=False)


def test_shifted_but_causal_observation_passes_real_review_protocol(tmp_path):
    shot = performance_shot()
    video, reviewer, sdk = reviewer_for(tmp_path, observed_response())
    result = reviewer.review(str(video), shot)
    assert result.accepted and result.action_sequence_valid
    schema = sdk.calls[0]['response_format']['json_schema']['schema']
    assert set(OBSERVED_TIMES) <= set(schema['properties']['action_event_evidence']['items']['properties'])
    assert 'not the suggested beat window' in sdk.calls[0]['messages'][0]['content']


@pytest.mark.parametrize('mutation', [
    lambda r: r['action_event_evidence'][0].update(effect_visible=False),
    lambda r: r['action_event_evidence'][0].update(result_visible=False),
    lambda r: r['action_event_evidence'][0].update(causal_order_valid=False),
    lambda r: r['action_event_evidence'][0].update(scope_valid=False),
    lambda r: r['action_event_evidence'][1].update(observed_end_seconds=4.9),
    lambda r: r['action_event_evidence'][1].update(observed_start_seconds=.1, observed_effect_seconds=.2),
    lambda r: r['action_event_evidence'][0].update(observed_reaction_seconds=.2),
    lambda r: r.update(identity_continuity_valid=False),
    lambda r: r.update(blocking_valid=False),
    lambda r: r.update(creative_brief_valid=False),
])
def test_flexibility_never_waives_quality_causality_or_result(tmp_path, mutation):
    response = observed_response()
    mutation(response)
    video, reviewer, _ = reviewer_for(tmp_path, response)
    assert not reviewer.review(str(video), performance_shot()).accepted


@pytest.mark.parametrize('invalid', [True, '1.0', float('nan'), float('inf'), None])
def test_observation_boundary_rejects_non_finite_or_non_numeric_time(invalid):
    contract = compile_action_contract(performance_shot())
    response = observed_response()['action_event_evidence']
    response[0]['observed_start_seconds'] = invalid
    assert contract.event_evidence_shape_issues(response)


def test_overlapping_time_does_not_inflate_observed_action_density():
    shot = performance_shot()
    evidence = observed_response()['action_event_evidence']
    for item, times in zip(evidence, [(0., .5, .8, 2.), (1., 1.3, 1.6, 2.5)]):
        item.update(zip(OBSERVED_TIMES, times))
    assert any('时长不足' in i for i in compile_action_contract(shot).event_evidence_issues(evidence))


def test_flexible_interaction_entry_adapts_path_but_freezes_results(tmp_path):
    _, previous, tail = scenario(tmp_path)
    shot = performance_shot()
    shot.update(shot_id=2, scene_id='roof', continuity_from_previous='seamless')
    original = deepcopy(shot)
    planner = Planner(entry_action='continues the already-low blade toward the same guard',
                      entry_visible_result=shot['action_beats'][0]['visible_result'])
    changed = prepare_continuation(shot, previous, str(tail), tmp_path, client=planner)
    assert changed['action_beats'][0]['action'] == planner.result['entry_action']
    assert changed['action_beats'][0]['interaction_geometry'] == original['action_beats'][0]['interaction_geometry']
    assert changed['action_beats'][1:] == original['action_beats'][1:]
    assert changed['end_state'] == original['end_state'] and original == shot
    assert prepare_continuation(shot, previous, str(tail), tmp_path, client=planner, allow_create=False) == changed
    assert len(planner.calls) == 1


def test_interaction_entry_cannot_change_result(tmp_path):
    _, previous, tail = scenario(tmp_path)
    shot = performance_shot()
    shot.update(scene_id='roof', continuity_from_previous='seamless')
    with pytest.raises(ContinuationUnavailableError, match='不能改变作用结果'):
        prepare_continuation(shot, previous, str(tail), tmp_path, client=Planner(entry_visible_result='opponent dies'))
    assert not list((tmp_path / 'continuations').glob('*.json'))


def test_real_rejected_event_evidence_is_not_relabelled_as_success():
    from pipeline.models import validate_storyboard
    record = json.loads((Path(__file__).parent / 'fixtures/real_performance_failure.json').read_text())
    story = record['storyboard']
    # Loading a legacy workspace must not add policy fields and invalidate its hashes.
    assert validate_storyboard(story) == story
    shot = story['shots'][0]
    assert not compile_action_contract(shot).review_projection.get('performance_policy')
    assert len(record['reports']) == 2
    for report in record['reports']:
        assert report['accepted'] is False
        assert compile_action_contract(shot).event_evidence_issues(report['evidence'])
        flexible = deepcopy(shot)
        flexible['production_slot']['choreography'] = 'flexible'
        # Old booleans do not become new measured observations by changing policy.
        assert compile_action_contract(flexible).event_evidence_shape_issues(report['evidence'])


@pytest.mark.asyncio
async def test_formal_natural_language_entry_freezes_performance_policy_and_resumes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import config
    from pipeline.orchestrator import VideoPipeline
    from test_director_intent import Responses, intent_reply, draft
    from test_planning_semantics import verdict_for
    from authoring_cases import authoring_from_legacy

    brief = '15秒动作短片，连续借势变招，编舞由导演设计'
    authored = authoring_from_legacy(draft())
    authored['scene_entities'] = []
    authored['locations'] = [{'id': 'training_roof', 'description': 'training roof'}]
    for shot in authored['shots']:
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit.pop('carry', None)
                unit['link_previous'] = True  # Explicit continuous new-protocol annotation.
                unit['resolution'] = {'target_state': unit['result'], 'actor_state': None, 'scene_changes': []}
                unit['weight'] = 3.
    author = Responses({**intent_reply(brief), 'choreography': 'flexible'}, authored)
    review_calls = []
    class ReviewClient:
        def __init__(self):
            self.chat = self.completions = self
        def create(self, **kwargs):
            review_calls.append(kwargs)
            data = json.loads(kwargs['messages'][-1]['content'])
            verdict = verdict_for(data['storyboard'])
            for unit in verdict['units']:
                beat = data['storyboard']['shots'][unit['shot_id'] - 1]['action_beats'][unit['beat_index']]
                if beat.get('outcomes'):
                    unit.update(outcome_contract_valid=True, evidence_fields=['outcomes', 'action'])
            return Responses(verdict).create(**kwargs)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: author)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: ReviewClient())
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('paid stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0)
    result = await pipeline.run(brief)
    saved = Path(result).read_bytes()
    storyboard = json.loads(saved)
    assert len(storyboard['shots']) == 3
    for shot in storyboard['shots']:
        policy = compile_action_contract(shot).review_projection['performance_policy']
        assert policy['creative_brief'] == brief
        assert policy['version'] == 'performance-v2-outcomes'
        prompt = compile_normal_provider_prompt(shot, storyboard, has_observed_start=False)
        assert 'Preserve the following event dependencies' in prompt
        assert policy['time_basis'] == 'sample-bounds-v1'
        assert all(beat['action'] in prompt for beat in shot['action_beats'])
    resumed = VideoPipeline.from_workspace(pipeline.workspace)
    resumed.plan_only = True
    assert await resumed.run() == result
    assert Path(result).read_bytes() == saved
    assert len(author.calls) == 5 and len(review_calls) == 1
    assert not resumed.run_workspace.manifest.paid_take_budget.reservations


def test_cached_performance_review_cannot_hide_missing_user_requirement(tmp_path):
    shot = performance_shot()
    video, reviewer, sdk = reviewer_for(tmp_path, observed_response())
    assert reviewer.review(str(video), shot).accepted
    cache = next((tmp_path / 'semantic_reviews').glob('*.json'))
    saved = json.loads(cache.read_text())
    saved['review']['creative_brief_valid'] = False
    cache.write_text(json.dumps(saved))
    assert not reviewer.review(str(video), shot).accepted
    assert len(sdk.calls) == 1


def test_safe_recompile_does_not_reintroduce_raw_creative_brief():
    shot = performance_shot()
    shot['production_slot']['creative_brief'] = 'raw-rejected-brief-marker'
    normal = compile_normal_provider_prompt(shot, {}, has_observed_start=False)
    safe = compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0)
    assert 'raw-rejected-brief-marker' in normal
    assert 'raw-rejected-brief-marker' not in safe
    assert 'Choreography is performance-guided:' in safe
    assert 'attacker steps back' in safe
