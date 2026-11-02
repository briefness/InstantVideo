"""Pending creative claims have one approval owner; raw evidence is immutable."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from itertools import product

import jsonschema
import pytest

from pipeline.action_authoring import compile_authored_storyboard
from pipeline.action_results import result_claims
from pipeline.authoring_context import SceneLocation, bind_authored_context
from pipeline.authoring_schedule import bind_authored_schedule
from pipeline.production_plan import build_production_plan
from pipeline.storyboard_authoring import _bind_scene_entities
from test_scheduled_production_flow import scheduled_services


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline replay attempted network'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline replay attempted network'))


def captured():
    record = json.loads((Path(__file__).parent / 'fixtures/real_candidate_authority_failure.json').read_text())
    for source in record['sources'].values():
        assert hashlib.sha256(source['text'].encode()).hexdigest() == source['sha256']
    return record


def prepare_record():
    record = captured()
    sources = {n: json.loads(s['text']) for n, s in record['sources'].items()}
    plan = build_production_plan('action', 15, intent=record['intent'])
    identity = sources['planning_identity.json']
    raw = sources['planning_shot_001.json']['shot']
    bound = bind_authored_schedule({'shot_id': 1, **raw}, plan['slots'][0])
    bound = bind_authored_context(bound, [SceneLocation.model_validate(s) for s in identity['locations']],
                                  identity['scene_entities'])
    _bind_scene_entities(bound, [c['name'] for c in identity['characters']], identity['scene_entities'])
    frozen = {k: v for k, v in identity.items() if k not in {'locations', 'scene_entities'}}
    return record, raw, bound, frozen, plan


def test_real_record_reaches_pending_review_without_changing_authored_importance():
    original = captured()
    record, raw, bound, identity, plan = prepare_record()
    result = compile_authored_storyboard({**identity, 'shots': [bound]}, {**plan, 'slots': plan['slots'][:1]})
    shot = result['shots'][0]
    assert shot['result_admission'] == 'review-required-v1'
    original_action = raw['action_plan']['interaction_2']
    assert original_action['resolution']['states']['SWORD_WHITE'][0]['importance'] == 'incidental'
    beat = next(b for b in shot['action_beats'] if b['action'] == original_action['action'])
    target_claim = original_action['resolution']['states']['SWORD_WHITE'][0]['state']
    assert {'entity': 'SWORD_WHITE', 'state': target_claim} in list(result_claims(beat).values())
    assert 'result_directions' not in beat  # no approval or classification fabricated
    assert captured() == original == record
    assert shot['action_beats'][-1]['end_seconds'] == pytest.approx(5)


@pytest.mark.parametrize('actor,target,other', [
    ('Lead', 'Partner', 'Floor'), ('Sender', 'Receiver', 'Parcel'), ('Operator', 'Panel', 'Lamp'),
])
def test_author_annotations_cannot_preapprove_or_hide_claims(actor, target, other):
    from pipeline.action_results import CandidateResolution, lower_authored_resolution
    expected = {(actor, 'continues the movement'), (target, 'responds to the action'), (other, 'remains visible')}
    projection = None
    for labels in product((None, 'required', 'incidental'), repeat=3):
        raw = {'states': {owner: [{'state': state, **({'importance': label} if label else {})}]
                          for (owner, state), label in zip(sorted(expected), labels)}}
        original = deepcopy(raw)
        lowered = lower_authored_resolution(actor, target, CandidateResolution.model_validate(raw), pending_review=True)
        view = {key: lowered[key] for key in ('outcomes', 'scene_changes')}
        assert {(v['entity'], v['state']) for v in result_claims(view).values()} == expected
        assert raw == original
        if projection is None:
            projection = view
        assert view == projection
        assert {s['importance'] for s in lowered['source_map']} == set(labels)


def test_pending_claims_receive_independent_decisions_before_becoming_obligations():
    from pipeline.action_results import CandidateResolution, lower_authored_resolution, required_outcomes, directed_outcomes
    from pipeline.result_admission import apply_result_admission
    resolution = CandidateResolution(states={
        'A': [{'state': 'left wrist points down', 'importance': 'required'}],
        'B': [{'state': 'holds the parcel securely', 'importance': 'incidental'}],
        'Dust': [{'state': 'briefly airborne', 'importance': 'incidental'}]})
    lowered = lower_authored_resolution('A', 'B', resolution, pending_review=True)
    beat = {'actor': 'A', 'target': 'B', 'activity': 'interaction',
            **{k: lowered[k] for k in ('outcomes', 'scene_changes')}}
    shot = {'shot_id': 1, 'result_admission': 'review-required-v1', 'action_beats': [beat]}
    story = {'shots': [shot], 'production_plan': {'director_intent': {'source_request': 'Receiver holds the parcel securely'}}}
    decisions = [{'source': 'outcomes/0', 'basis': 'direction', 'evidence': 'wrist orientation can vary', 'minimal': True},
                 {'source': 'outcomes/1', 'basis': 'user', 'evidence': 'holds the parcel securely', 'minimal': True},
                 {'source': 'scene_changes/0', 'basis': 'direction', 'evidence': 'dust is replaceable atmosphere', 'minimal': True}]
    units = [{'shot_id': 1, 'beat_index': 0, 'result_admission': {'decisions': decisions,
              'requirements_preserved': True, 'continuous_performance': True, 'reason': 'independent test annotation'}}]
    original = deepcopy(story)
    admitted = apply_result_admission(story, units)['shots'][0]
    assert story == original and shot['result_admission'] == 'review-required-v1'
    assert admitted['result_admission'] == 'reviewed-results-v1'
    assert required_outcomes(admitted['action_beats'][0]) == [{'entity': 'B', 'state': 'holds the parcel securely'}]
    assert len(directed_outcomes(admitted['action_beats'][0])) == 2
    assert result_claims(admitted['action_beats'][0]) == result_claims(beat)
    incomplete = deepcopy(units)
    incomplete[0]['result_admission']['decisions'].pop()
    with pytest.raises(ValueError, match='逐条覆盖'):
        apply_result_admission(story, incomplete)
    rejected = deepcopy(units)
    rejected[0]['result_admission']['requirements_preserved'] = False
    with pytest.raises(ValueError, match='未通过'):
        apply_result_admission(story, rejected)


def test_new_wire_does_not_require_importance_or_allow_authored_approval():
    from test_authoring_schedule import raw_shot, schema_for, slot
    from pipeline.llm_options import ark_output_schema
    raw = raw_shot()
    for unit in raw['action_plan'].values():
        for states in unit['resolution']['states'].values():
            for claim in states:
                claim.pop('importance')
    for schema in (schema_for(slot()), ark_output_schema(schema_for(slot()))):
        jsonschema.validate({'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}, schema)
        assert 'result_admission' not in schema['properties']['shot']['properties']
        assert 'importance' not in schema['$defs']['CandidateState']['required']
    bound = bind_authored_schedule(raw, slot())
    from pipeline.action_authoring import AuthoredShot, compile_action_units
    assert compile_action_units(AuthoredShot.model_validate(bound), slot())
    for override in ('review-required-v1', 'reviewed-results-v1', None):
        with pytest.raises(ValueError, match='程序管理'):
            bind_authored_schedule({**raw, 'result_admission': override}, slot())
    with pytest.raises(ValueError, match='精确执行合同'):
        compile_action_units(AuthoredShot.model_validate(bound), {**slot(), 'choreography': 'exact'})


def test_missing_target_and_legacy_meaning_are_not_repaired():
    from pipeline.action_results import CandidateResolution, EntityResolution, lower_authored_resolution
    for pending in (False, True):
        with pytest.raises(ValueError, match='目标'):
            lower_authored_resolution('A', 'B', EntityResolution(states={'A': [{'state': 'ready', 'importance': 'required'}]}),
                                      pending_review=pending)
    with pytest.raises(ValueError, match='目标'):
        lower_authored_resolution('A', 'B', EntityResolution(states={'B': [{'state': 'evades', 'importance': 'incidental'}]}))
    with pytest.raises(ValueError, match='独立必要性审查'):
        lower_authored_resolution('A', 'B', CandidateResolution(states={'B': [{'state': 'evades'}]}))


def test_candidate_direction_stays_in_generation_without_becoming_a_visual_obligation():
    from pipeline.action_results import CandidateResolution, lower_authored_resolution
    from pipeline.causality import compile_action_contract
    from pipeline.result_admission import apply_result_admission
    from test_action_results import scene_shot
    from test_action_result_flow import annotated_verdict
    shot = scene_shot()
    shot['result_admission'] = 'review-required-v1'
    for action in shot['action_beats']:
        if action['activity'] == 'interaction':
            action.setdefault('scene_changes', [])
    beat = shot['action_beats'][0]
    lowered = lower_authored_resolution(beat['actor'], beat['target'], CandidateResolution(states={
        beat['target']: [{'state': 'remains able to continue'}],
        'Dust': [{'state': 'a small dust plume trails behind the movement', 'importance': 'required'}]}), pending_review=True)
    beat.update({k: lowered[k] for k in ('outcomes', 'scene_changes')})
    story = {'shots': [shot]}
    verdict = annotated_verdict(story)
    audit = next(u for u in verdict['units'] if u['beat_index'] == 0)['result_admission']
    direction = next(d for d in audit['decisions'] if d['source'] == 'scene_changes/0')
    direction.update(basis='direction', evidence='dust is decorative and replaceable')
    execution = apply_result_admission(story, verdict['units'])['shots'][0]
    contract = compile_action_contract(execution)
    assert 'a small dust plume trails behind the movement' in ' '.join(contract.prompt_parts)
    event = contract.review_projection['action_events'][0]
    assert all(c['entity'] != 'Dust' for c in event['outcomes'])
    assert {'entity': 'Dust', 'state': 'a small dust plume trails behind the movement'} in event['incidental_changes']
    assert [(b['start_seconds'], b['end_seconds']) for b in execution['action_beats']] == [
        (b['start_seconds'], b['end_seconds']) for b in shot['action_beats']]


@pytest.mark.parametrize('value', [None, '', '   ', 12])
def test_invalid_candidate_state_is_not_coerced(value):
    from pipeline.action_results import CandidateResolution
    with pytest.raises(ValueError):
        CandidateResolution.model_validate({'states': {'B': [{'state': value}]}})


def test_real_formal_replay_advances_once_without_inventing_uncaptured_shots(tmp_path, monkeypatch):
    import config
    from pipeline.orchestrator import VideoPipeline
    from test_director_intent import Responses
    import asyncio
    record = captured()
    sources = {n: json.loads(s['text']) for n, s in record['sources'].items()}
    class EndOfCapture(RuntimeError):
        pass
    class Replay(Responses):
        def create(self, **kwargs):
            if len(self.calls) == 3:
                assert json.loads(kwargs['messages'][-1]['content'])['slot']['shot_id'] == 2
                raise EndOfCapture('second shot was never recorded')
            return super().create(**kwargs)
    client = Replay({k: v for k, v in record['intent'].items() if k != 'source_request'},
                    sources['planning_identity.json'], sources['planning_shot_001.json'])
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0, take_policy='single_pass')
    with pytest.raises(EndOfCapture):
        asyncio.run(pipeline.run(record['intent']['source_request']))
    assert len(client.calls) == 3
    assert json.loads((pipeline.workspace / 'planning_shot_001.json').read_text()) == sources['planning_shot_001.json']
    trace = json.loads((pipeline.workspace / 'planning_lowering.json').read_text())
    assert trace['version'] == 'authoring-lowering-v2-candidates'
    assert any(c['entity'] == 'SWORD_WHITE' and c['importance'] == 'incidental'
               and c['destination'].startswith('outcomes/') for u in trace['units'] for c in u['claims'])
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations


@pytest.mark.asyncio
async def test_direct_generator_cannot_spend_on_unreviewed_candidates(scheduled_services, tmp_path):
    from pipeline.storyboard import generate_storyboard
    from pipeline.generator import VideoGenerator
    from pipeline.readiness import GenerationReadinessError
    brief, services = scheduled_services('货物交接')
    story = generate_storyboard(brief)
    assert all(s['result_admission'] == 'review-required-v1' for s in story['shots'])
    reservations = []
    def reserve(shot_id):
        reservations.append(shot_id)
        pytest.fail('unreviewed candidates reserved a paid take')
    generator = VideoGenerator(str(tmp_path / 'unreviewed'), reserve_paid_take=reserve, review_content=False)
    with pytest.raises(GenerationReadinessError, match='尚未完成必要性审查'):
        await generator.generate_all(story)
    assert not services.submissions and not reservations


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
async def test_unlabelled_creation_reaches_review_generation_and_full_export(scheduled_services, monkeypatch, topic):
    import config
    from pipeline.orchestrator import VideoPipeline
    from pipeline.run_state import RunWorkspace
    from test_full_production_flow import completion
    from tools.ffmpeg_ops import get_video_duration
    brief, services = scheduled_services(topic)
    create = services.author.create
    authored_states = []
    def unlabelled(**kwargs):
        response = create(**kwargs)
        value = json.loads(response.choices[0].message.content)
        def visit(node):
            if isinstance(node, dict):
                if isinstance(node.get('resolution'), dict) and 'states' in node['resolution']:
                    for entity, states in node['resolution']['states'].items():
                        for claim in states:
                            claim.pop('importance', None)
                            authored_states.append((entity, claim['state']))
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)
        visit(value)
        return completion(value)
    monkeypatch.setattr(services.author, 'create', unlabelled)
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], paid_take_budget=3, take_policy='single_pass')
    output = await pipeline.run(brief)
    assert Path(output).is_file() and get_video_duration(output) == pytest.approx(15, abs=1/24)
    assert len(services.author.calls) == 5 and len(services.planning) == 1
    assert len(services.submissions) == len(services.reviews) == 3
    assert all(s['model'] == config.SEEDANCE_MODEL for s in services.submissions)
    planned = services.planning[0]['storyboard']
    candidates = [(c['entity'], c['state']) for s in planned['shots'] for b in s['action_beats']
                  for c in result_claims(b).values()]
    assert sorted(candidates) == sorted(authored_states)
    execution = json.loads((pipeline.workspace / 'result_admission.json').read_text())['execution_storyboard']
    assert all(s['result_admission'] == 'reviewed-results-v1' for s in execution['shots'])
    assert len(RunWorkspace.resume(pipeline.workspace).manifest.paid_take_budget.reservations) == 3
    counts = (len(services.author.calls), len(services.planning), len(services.submissions), len(services.reviews))
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == (len(services.author.calls), len(services.planning), len(services.submissions), len(services.reviews))
