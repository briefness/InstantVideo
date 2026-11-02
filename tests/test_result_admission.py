"""Real failures stay failures; classifications below are explicit offline labels."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.action_results import result_claims, required_outcomes
from pipeline.causality import compile_action_contract
from pipeline.models import validate_storyboard
from pipeline.planning_review import validate_planning_verdict, ensure_planning_semantics
from pipeline.result_admission import apply_result_admission
from test_action_result_flow import annotated_verdict
from test_director_intent import Responses
from test_full_production_flow import production_services


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import httpx
    import requests
    def forbidden(*a, **kw):
        pytest.fail('result admission validation must remain offline')
    monkeypatch.setattr(httpx.Client, 'send', forbidden)
    monkeypatch.setattr(httpx.AsyncClient, 'send', forbidden)
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)


def recorded():
    return json.loads((Path(__file__).parent / 'fixtures/real_result_admission_failure.json').read_text())['sources']


def candidate():
    story = recorded()['storyboard.json']['value']
    for shot in story['shots']:
        shot['result_admission'] = 'review-required-v1'
    return story


def annotated(story):
    verdict = annotated_verdict(story)
    # This is a counterfactual protocol test, NOT a re-review of the actual film.
    # First-shot optional poses remain in original source and generation text.
    for unit in verdict['units']:
        if unit['shot_id'] == 1:
            for decision in unit['result_admission']['decisions']:
                decision.update(basis='direction', evidence='offline annotation: replaceable pose, same cause and response')
    return verdict


def test_actual_old_approval_is_not_evidence_of_result_necessity():
    sources = recorded()
    story = sources['storyboard.json']['value']
    raw = sources['planning_semantic_review.json']['value']['verdict']
    assert validate_planning_verdict(raw, story) == []
    old = compile_action_contract(story['shots'][0])
    report = next(v['value'] for k, v in sources.items() if k.startswith('semantic_reviews/'))
    assert report['review']['accepted'] is False
    assert old.event_evidence_issues(report['action_event_evidence'])
    with pytest.raises(ValueError):
        validate_planning_verdict(raw, candidate())
    assert compile_action_contract(story['shots'][0]).fingerprint() == old.fingerprint()


def test_classification_preserves_all_authored_text_but_changes_shared_obligations():
    story = candidate()
    original = deepcopy(story)
    raw = annotated(story)
    assert validate_planning_verdict(raw, story) == []
    execution = validate_storyboard(apply_result_admission(story, raw['units']))
    assert story == original
    for source, executed in zip(story['shots'][0]['action_beats'], execution['shots'][0]['action_beats']):
        assert result_claims(source) == result_claims(executed)
        assert source['action'] == executed['action']
    contract = compile_action_contract(execution['shots'][0])
    assert contract.review_projection['performance_policy']['result_authority'] == 'reviewed-results-v1'
    assert all(e['outcomes'] == [] for e in contract.review_projection['action_events'])
    assert '屈膝沉腰' not in json.dumps(contract.review_projection, ensure_ascii=False)
    assert '屈膝沉腰' in ' '.join(contract.prompt_parts)
    assert 'Suggested performance details (replaceable)' in ' '.join(contract.prompt_parts)
    assert contract.event_evidence_fields() == compile_action_contract(story['shots'][0]).event_evidence_fields()
    final = compile_action_contract(execution['shots'][-1])
    assert final.review_projection['required_final_outcome']
    assert all(e['outcomes'] for e in final.review_projection['action_events'])
    from pipeline.provider_prompt import compile_policy_safe_prompt
    safe = compile_policy_safe_prompt(execution['shots'][0], has_state_reference=False,
                                      image_role=None, reference_count=0)
    assert '屈膝沉腰' not in safe
    assert 'impact' in safe and 'evasion' in safe


@pytest.mark.parametrize('change', [
    lambda a: a['decisions'].pop(),
    lambda a: a['decisions'].append(a['decisions'][0]),
    lambda a: a['decisions'][0].update(source='outcomes/999'),
    lambda a: a['decisions'][0].update(basis='user', evidence='a quote not in the original request'),
    lambda a: a['decisions'][0].update(basis='continuity', evidence='/shots/0/action_beats/0/action'),
    lambda a: a['decisions'][0].update(basis='continuity', evidence='/shots/-1/start_state/prop_state'),
    lambda a: a.update(requirements_preserved='true'),
])
def test_unbound_incomplete_or_fabricated_audits_cannot_change_obligations(change):
    story = candidate()
    raw = annotated(story)
    change(raw['units'][1]['result_admission'])
    before = deepcopy(story)
    with pytest.raises(ValueError):
        apply_result_admission(story, raw['units'])
    assert story == before


@pytest.mark.parametrize('flag', ['requirements_preserved', 'continuous_performance'])
def test_missing_user_goal_or_broken_whole_performance_blocks(flag):
    story = candidate()
    raw = annotated(story)
    raw['units'][1]['result_admission'][flag] = False
    assert validate_planning_verdict(raw, story)
    with pytest.raises(ValueError):
        apply_result_admission(story, raw['units'])


def test_mixed_required_and_optional_sentence_cannot_be_demoted_wholesale():
    story = candidate()
    raw = annotated(story)
    raw['units'][1]['result_admission']['decisions'][0]['minimal'] = False
    assert validate_planning_verdict(raw, story)
    with pytest.raises(ValueError):
        apply_result_admission(story, raw['units'])


def test_review_cache_freezes_classification_without_mutating_draft(tmp_path, monkeypatch):
    from planning_protocol_cases import annotated_outcome
    story = candidate()
    client = Responses(annotated_outcome(annotated(story), story))
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: client)
    save = lambda n, d: (tmp_path / n).write_text(json.dumps(d))
    first = ensure_planning_semantics(story, tmp_path, save)
    again = ensure_planning_semantics(story, tmp_path, save)
    assert first == again and len(client.calls) == 1
    assert story == candidate()
    story['shots'][0]['action_beats'][1]['outcomes'][0]['state'] += ' modified'
    with pytest.raises(ValueError, match='不一致'):
        ensure_planning_semantics(story, tmp_path, save)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_formal_admission_failure_stops_before_video_budget(tmp_path, monkeypatch):
    from planning_protocol_cases import annotated_outcome
    import config
    from pipeline.orchestrator import VideoPipeline
    story = candidate()
    raw = annotated(story)
    raw['units'][1]['result_admission']['requirements_preserved'] = False
    raw = annotated_outcome(raw, story)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.orchestrator.generate_storyboard', lambda **kw: deepcopy(story))
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: Responses(raw))
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('paid stage reached'))
    pipeline = VideoPipeline(paid_take_budget=3)
    with pytest.raises(ValueError, match='未通过'):
        await pipeline.run(story['production_plan']['director_intent']['source_request'])
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations


@pytest.mark.asyncio
async def test_classification_reaches_formal_generator_and_resume(production_services, monkeypatch):
    from pipeline.orchestrator import VideoPipeline
    brief, services = production_services
    # Keep required results; downgrade only one optional actor pose. The existing
    # services are explicitly synthetic, including all visual observations.
    base = services.create
    def classified(**kwargs):
        response = base(**kwargs)
        content = kwargs['messages'][-1]['content']
        if isinstance(content, str) and '"storyboard"' in content:
            payload = json.loads(content)
            if payload.get('version') == 'planning-semantics-v6-bound-targets':
                verdict = json.loads(response.choices[0].message.content)
                for unit in verdict['units'].values():
                    for source, d in (unit.get('result_admission') or {}).get('decisions', {}).items():
                        if source == 'outcomes/1':
                            d.update(basis='direction', evidence='synthetic replaceable actor pose')
                response.choices[0].message.content = json.dumps(verdict)
        return response
    monkeypatch.setattr(services, 'create', classified)
    pipeline = VideoPipeline(resolution='720p', platforms=['youtube'], paid_take_budget=3, take_policy='single_pass')
    output = await pipeline.run(brief)
    assert Path(output).is_file()
    assert len(services.submissions) == 3
    assert all(r['performance_policy']['result_authority'] == 'reviewed-results-v1' for r in services.reviews)
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert len(services.submissions) == 3


def test_continuation_uses_admitted_requirements_without_mutating_source(tmp_path):
    from pipeline.continuation import prepare_continuation
    from test_continuation import scenario
    _, previous, tail = scenario(tmp_path)
    story = candidate()
    verdict = annotated(story)
    for unit in verdict['units']:
        if unit['shot_id'] == 2:
            for d in unit['result_admission']['decisions']:
                d.update(basis='direction', evidence='synthetic replaceable pose')
    shot = apply_result_admission(story, verdict['units'])['shots'][1]
    shot.update(scene_id='roof', continuity_from_previous='seamless')
    original = deepcopy(shot)
    client = Responses({'compatible': True, 'entry_action': 'continue the incoming movement',
                        'reason': 'synthetic compatible continuation'})
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=client)
    assert shot == original
    assert result['action_beats'][0]['outcomes'] == original['action_beats'][0]['outcomes']
    context = json.loads(client.calls[0]['messages'][-1]['content'])
    assert context['entry_constraint']['required_visible_result'] == ''
    assert all(not b.get('outcomes') for b in context['planned_shot']['action_beats'])
    assert prepare_continuation(shot, previous, str(tail), tmp_path, client=client, allow_create=False) == result
    assert len(client.calls) == 1


@pytest.mark.parametrize('subject,object_id,result,pose', [
    ('Courier', 'Parcel', 'has received the parcel', 'elbow bent precisely'),
    ('Listener', 'Letter', 'holds the letter', 'stands with left foot forward'),
    ('Worker', 'Panel', 'panel is joined', 'tool points diagonally down'),
])
def test_functional_result_survives_optional_implementation_across_domains(subject, object_id, result, pose):
    story = candidate()
    beat = story['shots'][0]['action_beats'][1]
    beat['outcomes'] = [{'entity': subject, 'state': result}, {'entity': subject, 'state': pose}]
    raw = annotated(story)
    raw['units'][1]['result_admission']['decisions'][0].update(
        basis='response', evidence='synthetic essential transfer or transformation')
    execution = apply_result_admission(story, raw['units'])
    assert required_outcomes(execution['shots'][0]['action_beats'][1]) == [{'entity': subject, 'state': result}]
    assert result_claims(execution['shots'][0]['action_beats'][1]) == result_claims(beat)


def test_valid_forward_dependency_and_user_quote_remain_required():
    story = candidate()
    raw = annotated(story)
    d = raw['units'][1]['result_admission']['decisions']
    d[0].update(basis='user', evidence='轻功入场要帅')
    d[1].update(basis='continuity', evidence='/shots/1/start_state/prop_state')
    execution = apply_result_admission(story, raw['units'])
    assert len(required_outcomes(execution['shots'][0]['action_beats'][1])) == 2
    # This checks binding only; relevance of the quoted requirement is a semantic judgment.


def test_optional_external_effect_still_has_bounded_permission_in_review():
    story = candidate()
    beat = story['shots'][0]['action_beats'][1]
    beat['scene_changes'][0]['importance'] = 'required'
    raw = annotated(story)
    execution = apply_result_admission(story, raw['units'])
    contract = compile_action_contract(execution['shots'][0])
    event = contract.review_projection['action_events'][0]
    optional = beat['scene_changes'][0]
    assert {'entity': optional['entity'], 'state': optional['state']} in event['incidental_changes']
    assert not event['outcomes']
