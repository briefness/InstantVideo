"""Requirement authority and grounded outcome judgments, without network calls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_planning_citations import recorded
from test_planning_review_protocol import synthetic_wire, sdk_reviewer, save_to
from planning_protocol_cases import sourced_wire


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    def deny(*args, **kwargs):
        pytest.fail('outcome regression attempted network access')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)


def test_real_identity_downgrade_stops_before_any_shot_authored():
    from pipeline.storyboard_authoring import generate_event_storyboard
    from test_director_intent import Responses
    raw = json.loads((Path(__file__).parent / 'fixtures/real_outcome_identity_failure.json').read_text())
    identity = raw['sources']['planning_identity.json']['value']
    story = recorded()['storyboard.json']['value']
    before = deepcopy((identity, story))
    client = Responses(identity)
    with pytest.raises(ValueError, match='用户结局'):
        generate_event_storyboard(client, story['production_plan']['director_intent']['source_request'],
            story['production_plan'], aspect_ratio='16:9', resolution='480p', style='cinematic')
    assert len(client.calls) == 1 and (identity, story) == before
    schema = client.calls[0]['response_format']['json_schema']['schema']
    arc_type = schema['properties']['story_arc']['$ref'].rsplit('/', 1)[-1]
    assert schema['$defs'][arc_type]['properties']['resolution']['const'] == story['production_plan']['director_intent']['ending_evidence']


@pytest.mark.parametrize('relation', ['contradicted', 'undetermined'])
def test_grounded_negative_outcome_cannot_be_overridden_by_general_approval(tmp_path, monkeypatch, relation):
    from pipeline.planning_review import ensure_planning_semantics
    story = recorded()['storyboard.json']['value']
    wire = sourced_wire(synthetic_wire(story), story)
    wire.pop('requested_outcome_valid')
    wire['requested_outcome'] = {'relation': relation,
        'evidence': ['/shots/2/action_beats/1/visible_result'],
        'reason': 'Offline annotation: incapacitation does not establish death.'}
    sdk, calls = sdk_reviewer(monkeypatch, wire)
    with sdk:
        for _ in range(2):
            with pytest.raises(ValueError, match='incapacitation'):
                ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
    assert len(calls) == 1 and not (tmp_path / 'result_admission.json').exists()
    payload = json.loads(calls[0]['messages'][-1]['content'])
    assert payload['outcome_review']['requirement'] == story['production_plan']['director_intent']['ending_evidence']
    assert 'story_arc' not in json.dumps(payload['outcome_review']['sources'])
    assert json.loads((tmp_path / 'planning_semantic_review.json').read_text())['verdict'] == wire


@pytest.mark.parametrize('evidence', [[], ['/story_arc/resolution'], ['/shots/2/production_slot/required_outcome'],
    ['/shots/2/end_state'], ['/shots/2/start_state/subject'], ['https://example.invalid']])
def test_unsubstantiated_approval_is_rejected(evidence):
    from pipeline.requested_outcome import outcome_review_context, decode_outcome_review
    story = recorded()['storyboard.json']['value']
    with pytest.raises(ValueError, match='结局.*证据'):
        decode_outcome_review({'requested_outcome': {'relation': 'entailed', 'evidence': evidence,
            'reason': 'Synthetic approval'}}, outcome_review_context(story))


def test_partial_batch_does_not_claim_to_have_observed_final_outcome():
    from pipeline.requested_outcome import outcome_review_context
    story = recorded()['storyboard.json']['value']
    partial = {**story, 'shots': story['shots'][:2], 'review_scope': {'is_final_batch': False}}
    assert outcome_review_context(partial) is None
    assert outcome_review_context({**story, 'shots': story['shots'][2:],
        'review_scope': {'is_final_batch': True}})['requirement']


@pytest.mark.parametrize('ending', ['最后双方和解，不能有人死亡。', '最终钥匙交到收件人手中。',
    '水必须完全结冰，不能只是变冷。', '结尾保留悬念，不要确认胜负。'])
def test_requirement_is_original_text_not_genre_or_weaker_summary(ending):
    from pipeline.requested_outcome import requested_ending
    plan = {'director_intent': {'source_request': ending, 'ending_evidence': ending,
                               'ending_requirement': 'A lossy summary'}}
    assert requested_ending(plan) == ending


def test_unknown_is_not_falsehood_or_success_and_empty_sources_are_representable():
    from pipeline.requested_outcome import outcome_review_context, decode_outcome_review, outcome_response_schema
    from pipeline.planning_review import PlanVerdict
    from pipeline.llm_options import ark_output_schema
    story = recorded()['storyboard.json']['value']
    empty = {**story, 'shots': [{'action_beats': []}]}
    context = outcome_review_context(empty)
    schema = ark_output_schema(outcome_response_schema(PlanVerdict.model_json_schema(), context))
    shape = schema['properties']['requested_outcome']['properties']
    assert shape['relation']['const'] == 'undetermined'
    assert shape['evidence']['const'] == []
    raw = {'reason': 'general approval', 'requested_outcome': {
        'relation': 'undetermined', 'evidence': [], 'reason': 'No established result is available.'}}
    result, evidence = decode_outcome_review(raw, context)
    assert result['requested_outcome_valid'] is False
    assert evidence['relation'] == 'undetermined' and evidence['source_fields'] == {}


def test_unmodified_real_approval_cannot_answer_new_outcome_contract():
    from pipeline.requested_outcome import outcome_review_context, decode_outcome_review
    record = recorded()
    raw = record['planning_semantic_review.json']['value']['verdict']
    before = deepcopy(raw)
    with pytest.raises(ValueError, match='不能只返回整体批准'):
        decode_outcome_review(raw, outcome_review_context(record['storyboard.json']['value']))
    assert raw == before


@pytest.mark.parametrize('change', ['requirement', 'unknown_version', 'omit_version'])
def test_cache_cannot_hide_changed_requirement_or_protocol(tmp_path, monkeypatch, change):
    from pipeline.planning_review import ensure_planning_semantics
    story = recorded()['storyboard.json']['value']
    wire = sourced_wire(synthetic_wire(story), story)
    wire.pop('requested_outcome_valid')
    wire['requested_outcome'] = {'relation': 'undetermined', 'evidence': [], 'reason': 'Not established.'}
    sdk, calls = sdk_reviewer(monkeypatch, wire)
    with sdk:
        with pytest.raises(ValueError, match='Not established'):
            ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
        if change == 'requirement':
            story['production_plan']['director_intent']['ending_evidence'] = '结局清楚、非血腥。'
        else:
            path = tmp_path / 'planning_semantic_review.json'
            cache = json.loads(path.read_text())
            if change == 'unknown_version':
                cache['outcome_version'] = 'unknown'
            else:
                cache.pop('outcome_version')
            save_to(tmp_path)('planning_semantic_review.json', cache)
        with pytest.raises(ValueError, match='不一致|协议无效'):
            ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_formal_plan_preserves_requirement_through_prompt_review_and_resume(tmp_path, monkeypatch):
    import config
    from types import SimpleNamespace
    from pipeline.orchestrator import VideoPipeline
    from pipeline.causality import compile_action_contract
    from pipeline.provider_prompt import compile_policy_safe_prompt
    from test_action_results import annotated_story
    from test_action_result_flow import annotated_verdict
    from test_director_intent import Responses, intent_reply
    from planning_protocol_cases import annotated_outcome
    ending = '最后双方仍能继续行动。'
    brief = '15秒，三段连续双人训练，动作密集自然。' + ending
    # Explicit synthetic providers exercise the complete data path; their labels
    # are not an assessment of model comprehension or cinematic performance.
    raw = annotated_story()
    raw['scene_entities'] = ['Door', 'Dust']
    raw['story_arc']['resolution'] = ending
    for shot in raw['shots']:
        for unit in shot['action_plan']:
            if unit['kind'] == 'interaction':
                unit['weight'] = 3.
    # A declared payoff reallocates one interaction to the middle slot. Supply
    # that event explicitly in this synthetic fixture, keeping the real compiler.
    extra = deepcopy(next(u for u in raw['shots'][1]['action_plan'] if u['kind'] == 'interaction'))
    extra.update(id='additional_exchange', action='continues with a second open-palm redirection')
    raw['shots'][1]['action_plan'].insert(-1, extra)
    author = Responses({**intent_reply(brief), 'choreography': 'flexible',
        'ending_requirement': ending, 'ending_evidence': ending}, raw)
    calls = []
    def review(**kwargs):
        payload = json.loads(kwargs['messages'][-1]['content'])
        calls.append(payload)
        story = payload['storyboard']
        return Responses(annotated_outcome(annotated_verdict(story), story)).create(**kwargs)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: author)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=review))))
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0, take_policy='single_pass')
    await pipeline.run(brief)
    assert pipeline.run_workspace.manifest.stage == 'planning_ready'
    assert len(author.calls) == 5 and len(calls) == 1
    for call in author.calls[2:]:
        assert json.loads(call['messages'][-1]['content'])['story']['story_arc']['resolution'] == ending
    story = json.loads((pipeline.workspace / 'storyboard.json').read_text())
    final = story['shots'][-1]
    contract = compile_action_contract(final)
    assert contract.review_projection['required_final_outcome'] == ending
    assert ending in compile_policy_safe_prompt(final, has_state_reference=False,
        image_role=None, reference_count=0)
    assert calls[0]['outcome_review']['requirement'] == ending
    cached = json.loads((pipeline.workspace / 'planning_semantic_evidence.json').read_text())
    assert cached['requested_outcome']['source_fields']
    resumed = VideoPipeline.from_workspace(pipeline.workspace)
    resumed.plan_only = True
    await resumed.run()
    assert len(author.calls) == 5 and len(calls) == 1
    assert not resumed.run_workspace.manifest.paid_take_budget.reservations
