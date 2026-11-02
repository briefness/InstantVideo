"""Unedited real-response replay and lossless author-to-execution boundaries."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.action_authoring import AuthoredShot, compile_action_units
from pipeline.action_results import required_outcomes, incidental_changes
from pipeline.production_plan import build_production_plan
from test_director_intent import Responses
from test_full_production_flow import production_services


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import httpx
    import requests
    def forbidden(*args, **kwargs):
        pytest.fail('authoring replay must remain offline')
    monkeypatch.setattr(httpx.Client, 'send', forbidden)
    monkeypatch.setattr(httpx.AsyncClient, 'send', forbidden)
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)


def recorded_run():
    data = json.loads((Path(__file__).parent / 'fixtures/real_authoring_boundary_failure.json').read_text())
    sources = data['sources']
    intent = json.loads(sources['planning_intent_response.json']['value']['content'])
    intent['source_request'] = data['request']
    story = deepcopy(sources['planning_identity.json']['value'])
    story['shots'] = [{'shot_id': i, **deepcopy(sources[f'planning_shot_{i:03d}.json']['value']['shot'])}
                      for i in range(1, 4)]
    return intent, story


def assert_claims_preserved(unit, beat):
    resolution = unit['resolution']
    required = [(unit['target'], resolution['target_state'])]
    if resolution['actor_state'] is not None:
        required.append((unit['actor'], resolution['actor_state']))
    required.extend((c['entity'], c['state']) for c in resolution['scene_changes'] if c['importance'] == 'required')
    optional = [(c['entity'], c['state']) for c in resolution['scene_changes'] if c['importance'] == 'incidental']
    assert Counter((c['entity'], c['state']) for c in required_outcomes(beat)) == Counter(required)
    assert Counter((c['entity'], c['state']) for c in incidental_changes(beat)) == Counter(optional)


def test_real_last_shot_lowers_every_claim_without_editing_source():
    intent, story = recorded_run()
    original = deepcopy(story)
    shot = story['shots'][-1]
    assert len(shot['mood']) == 145
    parsed = AuthoredShot.model_validate(shot)
    beats = compile_action_units(parsed, build_production_plan('action', 15, intent=intent)['slots'][-1])
    assert parsed.mood == shot['mood'] and story == original
    assert len(beats) == len(shot['action_plan'])
    for unit, beat in zip(shot['action_plan'], beats):
        assert beat['action'] == unit['action'] and beat['visible_result'] == unit['result']
        if unit['kind'] == 'interaction':
            assert_claims_preserved(unit, beat)


def test_real_three_shot_claims_survive_but_invalid_staging_is_not_repaired():
    from pipeline.storyboard import _compile_storyboard_contract
    from pipeline.action_authoring import compile_authored_storyboard
    from pipeline.readiness import ensure_storyboard_ready
    intent, story = recorded_run()
    original = deepcopy(story)
    # Replay historical authoring at the compiler seam. It is not a response
    # to the new identity request that binds the user's original outcome.
    plan = build_production_plan('action', 15, intent=intent)
    result = _compile_storyboard_contract(compile_authored_storyboard(story, plan),
        '16:9', '720p', 'cinematic', production_plan=plan)
    # 曾被最后一次交互的目标覆盖，掩盖了真实原稿里的非法实体引用。
    with pytest.raises(ValueError, match='blocking.action_target 引用未注册实体'):
        ensure_storyboard_ready(result)
    # 历史别名适配仍可能解析显式角色名，但不能把无实体的表演描述补成目标。
    invalid = story['shots'][-1]['blocking']['SwordMaster_Mo']['action_target']
    assert result['shots'][-1]['blocking']['SwordMaster_Mo']['action_target'] == invalid
    assert story == original
    assert [shot['duration'] for shot in result['shots']] == [5, 5, 5]
    for source, shot in zip(story['shots'], result['shots']):
        assert shot['mood'] == source['mood']
        for unit, beat in zip(source['action_plan'], shot['action_beats']):
            if unit['kind'] == 'interaction':
                assert_claims_preserved(unit, beat)


@pytest.mark.parametrize('actor,target,other,states', [
    ('Courier', 'Receiver', 'Parcel', ['holds parcel', 'seal intact', 'sleeve flutters']),
    ('Welder', 'Panel', 'Clamp', ['seam joined', 'surface cooled', 'brief glow']),
    ('Speaker', 'Listener', 'Letter', ['holds letter', 'has read the message', 'smiles']),
])
def test_general_claim_partition_is_lossless_and_source_mapped(actor, target, other, states):
    from pipeline.action_results import AuthoredResolution, lower_authored_resolution
    raw = {'target_state': states[0], 'actor_state': 'remains ready', 'scene_changes': [
        {'entity': target, 'state': states[1], 'importance': 'required'},
        {'entity': target, 'state': states[2], 'importance': 'incidental'},
        {'entity': other, 'state': 'stable', 'importance': 'required'},
        {'entity': other, 'state': 'briefly illuminated', 'importance': 'incidental'},
        {'entity': actor, 'state': 'turns slightly', 'importance': 'incidental'},
        {'entity': target, 'state': states[0], 'importance': 'required'},
    ]}
    original = deepcopy(raw)
    lowered = lower_authored_resolution(actor, target, AuthoredResolution.model_validate(raw))
    assert_claims_preserved({'actor': actor, 'target': target, 'resolution': raw}, lowered)
    assert raw == original
    assert len(lowered['source_map']) == 2 + len(raw['scene_changes'])
    for mapping in lowered['source_map']:
        source = raw
        for key in mapping['source'].split('/')[1:]:
            source = source[int(key)] if isinstance(source, list) else source[key]
        destination, index = mapping['destination'].split('/')
        claim = lowered[destination][int(index)]
        assert claim['entity'] == mapping['entity']
        assert claim['state'] == (source['state'] if isinstance(source, dict) else source)
        assert claim.get('importance', 'required') == mapping['importance']


@pytest.mark.parametrize('event_weight', [.001, .5, 1., 100.])
def test_compiler_owns_whole_film_budget_without_changing_authored_content(event_weight):
    from pipeline.action_authoring import compile_authored_storyboard
    from pipeline.film_rhythm import planned_interaction_seconds, planning_rhythm_context
    from pipeline.action_timing import action_timing_issues
    from test_action_results import annotated_story
    from test_director_intent import intent_reply
    story = annotated_story()
    for shot in story['shots']:
        for unit in shot['action_plan']:
            unit['weight'] = event_weight if unit['kind'] == 'interaction' else 5.
    original = deepcopy(story)
    intent = {**intent_reply('offline interaction'), 'source_request': 'offline interaction', 'choreography': 'flexible'}
    plan = build_production_plan('action', 15, intent=intent)
    compiled = compile_authored_storyboard(story, plan)
    assert story == original
    assert planned_interaction_seconds(compiled['shots']) >= 10.5 - 1e-6
    for source, shot, slot in zip(story['shots'], compiled['shots'], plan['slots']):
        assert not action_timing_issues({**shot, 'production_slot': slot})
        assert [u['action'] for u in source['action_plan']] == [b['action'] for b in shot['action_beats']]
        assert shot['action_beats'][-1]['end_seconds'] == 5
    # Early preferred timings are provisional; capacity in earlier shots remains
    # available rather than failing before the final authoring call.
    prefix = compile_authored_storyboard({**story, 'shots': story['shots'][:2]}, {**plan, 'slots': plan['slots'][:2]})
    context = planning_rhythm_context(plan, prefix['shots'])
    assert context['prior_reallocatable_seconds'] >= 0


def test_impossible_whole_film_budget_is_not_hidden_by_a_tolerance_or_dropped_action():
    from pipeline.action_authoring import compile_authored_storyboard
    intent, story = recorded_run()
    plan = build_production_plan('action', 15, intent=intent)
    plan['min_interaction_seconds'] = 15.
    original = deepcopy(story)
    with pytest.raises(ValueError, match='不可同时满足'):
        compile_authored_storyboard(story, plan)
    assert story == original


@pytest.mark.asyncio
async def test_original_invalid_record_stops_before_review_and_on_resume(tmp_path, monkeypatch):
    import config
    from pipeline.orchestrator import VideoPipeline
    from test_action_result_flow import annotated_verdict
    from pipeline.storyboard import _compile_storyboard_contract
    from pipeline.action_authoring import compile_authored_storyboard
    from planning_protocol_cases import annotated_outcome
    intent, story = recorded_run()
    request = intent['source_request']
    plan = build_production_plan('action', 15, intent=intent)
    historical = _compile_storyboard_contract(compile_authored_storyboard(story, plan),
        '16:9', '720p', 'cinematic', production_plan=plan)
    for shot in historical['shots']:
        shot['result_admission'] = 'review-required-v1'
    reviews, bindings = [], []
    class Admission:
        def __init__(self):
            self.chat = self.completions = self
        def create(self, **kwargs):
            context = json.loads(kwargs['messages'][-1]['content'])
            if context['version'] == 'scene-bindings-v1':
                bindings.append(context)
                verdict = {'bindings': [{'shot_id': i, 'anchor_shot_id': 1, 'certain': True,
                    'reason': 'offline annotation: same roof'} for i in (1, 2, 3)]}
            else:
                reviews.append(context)
                verdict = annotated_outcome(annotated_verdict(context['storyboard']), context['storyboard'])
            return Responses(verdict).create(**kwargs)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.orchestrator.generate_storyboard', lambda **kw: deepcopy(historical))
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: Admission())
    monkeypatch.setattr('pipeline.scene_binding.OpenAI', lambda **kw: Admission())
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda *a, **kw: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0)
    with pytest.raises(ValueError, match='blocking.action_target 引用未注册实体'):
        await pipeline.run(request)
    assert not reviews and not bindings
    for source, shot in zip(story['shots'], historical['shots']):
        for unit, beat in zip(source['action_plan'], shot['action_beats']):
            if unit['kind'] == 'interaction':
                assert_claims_preserved(unit, beat)
    resumed = VideoPipeline.from_workspace(pipeline.workspace)
    resumed.plan_only = True
    with pytest.raises(ValueError, match='blocking.action_target 引用未注册实体'):
        await resumed.run()
    assert not reviews and not bindings
    assert not resumed.run_workspace.manifest.paid_take_budget.reservations


def test_contradictions_survive_lowering_and_semantic_rejection_is_still_binding(tmp_path, monkeypatch):
    from pipeline.action_authoring import compile_authored_storyboard
    from pipeline.storyboard import _compile_storyboard_contract
    from pipeline.planning_review import ensure_planning_semantics
    from test_action_result_flow import annotated_verdict
    from test_action_results import annotated_story
    from test_director_intent import intent_reply
    raw = annotated_story()
    unit = raw['shots'][0]['action_plan'][1]
    unit['resolution']['scene_changes'].append(
        {'entity': 'Door', 'state': 'closed and locked', 'importance': 'required'})
    plan = build_production_plan('action', 15, intent={**intent_reply('offline'),
        'source_request': 'offline', 'choreography': 'flexible'})
    story = _compile_storyboard_contract(compile_authored_storyboard(raw, plan),
        '16:9', '720p', 'cinematic', production_plan=plan)
    claims = required_outcomes(story['shots'][0]['action_beats'][1])
    assert {'entity': 'Door', 'state': 'open'} in claims
    assert {'entity': 'Door', 'state': 'closed and locked'} in claims
    verdict = annotated_verdict(story)
    verdict['units'][1].update(outcome_contract_valid=False, reason='offline annotation: same door open and locked')
    reviewer = Responses(verdict)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: reviewer)
    saved = {}
    with pytest.raises(ValueError, match='结果合同无效'):
        ensure_planning_semantics(story, tmp_path, lambda name, value: saved.update({name: value}))
    actual = json.loads(reviewer.calls[0]['messages'][-1]['content'])['storyboard']
    assert actual == story
    assert saved['planning_semantic_evidence.json']['issues']


@pytest.mark.parametrize('change', [
    lambda shot: shot.update(mood='x' * 1601),
    lambda shot: shot.update(mood={'injected': 'object'}),
    lambda shot: shot['action_plan'][0]['resolution'].update(scene_changes=[
        {'entity': 'Object', 'state': 'unchanged', 'importance': 'required'}] * 15),
    lambda shot: shot['action_plan'][0]['resolution']['scene_changes'][0].update(importance='ignore'),
])
def test_prose_and_claim_resource_boundaries_still_reject_invalid_input(change):
    _, story = recorded_run()
    shot = story['shots'][-1]
    change(shot)
    with pytest.raises(ValueError):
        AuthoredShot.model_validate(shot)


@pytest.mark.asyncio
async def test_multiple_primary_claims_cross_actual_generator_review_continuation_and_export(production_services, monkeypatch):
    from pipeline.orchestrator import VideoPipeline
    from pipeline.action_results import result_claims
    from test_action_result_flow import annotated_verdict
    from test_full_production_flow import completion
    from planning_protocol_cases import wire_verdict
    brief, services = production_services
    responses = list(services.author.values)
    for response in responses:
        if 'shot' not in response:
            continue
        for unit in response['shot']['action_plan']:
            if unit['kind'] == 'interaction':
                unit['resolution']['scene_changes'].extend([
                    {'entity': unit['target'], 'state': 'keeps both hands free', 'importance': 'required'},
                    {'entity': unit['target'], 'state': 'clothes flutter', 'importance': 'incidental'}])
    services.author.values = iter(responses)
    original_create = services.create
    def classify_details(**kwargs):
        content = kwargs['messages'][-1]['content']
        context = json.loads(content) if isinstance(content, str) else {}
        if 'storyboard' not in context:
            return original_create(**kwargs)
        services.planning.append(context)
        story = context['storyboard']
        verdict = annotated_verdict(story)
        by_id = {s['shot_id']: s for s in story['shots']}
        for unit in verdict['units']:
            claims = result_claims(by_id[unit['shot_id']]['action_beats'][unit['beat_index']])
            for decision in (unit.get('result_admission') or {}).get('decisions', []):
                if claims[decision['source']]['state'] == 'clothes flutter':
                    decision.update(basis='direction', evidence='offline annotation: flutter is replaceable atmosphere')
        return completion(wire_verdict(verdict, kwargs))
    monkeypatch.setattr(services, 'create', classify_details)
    pipeline = VideoPipeline(resolution='720p', platforms=['youtube'], paid_take_budget=3,
                             take_policy='single_pass')
    output = await pipeline.run(brief)
    assert Path(output).is_file() and len(services.submissions) == 3
    for contract in services.reviews:
        for event in contract['action_events']:
            assert {'entity': event['target'], 'state': 'keeps both hands free'} in event['outcomes']
            assert {'entity': event['target'], 'state': 'clothes flutter'} not in event['outcomes']
    assert all('clothes flutter' in call['prompt'] for call in services.submissions)
    assert len(services.entries) == 2
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert len(services.submissions) == 3
