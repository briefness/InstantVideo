"""分配结果抵达生成边界；观察独立、历史字节不变、失败稿不修写。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from pipeline.action_authoring import compile_authored_storyboard
from pipeline.causality import compile_action_contract
from pipeline.film_rhythm import planned_interaction_seconds
from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
from pipeline.semantic_review import _review_contract
from pipeline.storyboard import _compile_storyboard_contract
from test_creation_schedule_flow import SCENARIOS, scenario_story
from test_scheduled_production_flow import scheduled_services


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        pytest.fail('execution projection replay attempted a network connection')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)


def compile_story(raw, plan):
    return _compile_storyboard_contract(compile_authored_storyboard(raw, plan),
        '16:9', '480p', 'cinematic', production_plan=plan)


def prompts(shot):
    return (compile_normal_provider_prompt(shot, {}, has_observed_start=False),
            compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0))


@pytest.mark.parametrize('topic', SCENARIOS)
def test_allocation_is_visible_to_both_generators_without_creating_review_quotas(topic):
    _, _, source, plan = scenario_story(topic)
    original = deepcopy(source)
    baseline = compile_story(source, {**plan, 'min_interaction_seconds': 0.})
    allocated = compile_story(source, plan)
    assert source == original
    assert planned_interaction_seconds(allocated['shots']) >= 10.5 - 1e-6
    for before, shot in zip(baseline['shots'], allocated['shots']):
        target = planned_interaction_seconds([shot])
        for prompt in prompts(shot):
            assert f'Allocate about {target:g}s of this {shot["duration"]:g}s shot' in prompt
            assert 'Show at least 0s' not in prompt
            assert 'not fixed timestamps or a separate per-shot acceptance minimum' in prompt
        assert shot['production_slot']['min_interaction_seconds'] == 0
        assert _review_contract(shot) == _review_contract(before)
        assert 'Allocate about' not in compile_action_contract(shot).event_review_instruction()
        assert 'Allocate about' not in compile_action_contract(shot).review_instruction
        if abs(target - planned_interaction_seconds([before])) > .001:
            assert all(a != b for a, b in zip(prompts(shot), prompts(before)))
        assert [b['action'] for b in shot['action_beats']] == [u['action'] for u in original['shots'][shot['shot_id'] - 1]['action_plan']]


def test_original_compilable_failure_retains_claims_and_transmits_allocated_budget():
    from test_authoring_lowering import recorded_run
    from pipeline.production_plan import build_production_plan
    intent, raw = recorded_run()
    before = deepcopy(raw)
    compiled = compile_story(raw, build_production_plan('action', 15, intent=intent))
    for shot in compiled['shots']:
        for prompt in prompts(shot):
            assert f'Allocate about {planned_interaction_seconds([shot]):g}s' in prompt
    assert raw == before


def test_old_saved_contract_and_both_prompts_remain_byte_identical():
    from test_performance_outcomes import outcome_shot
    shot = outcome_shot()
    assert compile_action_contract(shot).fingerprint() == '57b6430aa1d3afa1c4431b21e1bf8bbca46a420620e2fe5bfdb6d303781e9423'
    assert [hashlib.sha256(p.encode()).hexdigest() for p in prompts(shot)] == [
        'ac8758bbde66326cfa5db360850ce7174b4653584909a7eece11eceebd370544',
        '99d4e382c80acfa81e4f99ddd1a49d8a402a259ace8829ecf60c2b3f682f8e77']


@pytest.mark.parametrize('mode', ['exact', 'reference'])
def test_exact_and_reference_motion_keep_timed_authority(mode):
    from test_performance_outcomes import outcome_shot
    shot = outcome_shot()
    if mode == 'exact':
        shot['production_slot']['choreography'] = 'exact'
    else:
        shot['motion_plan'] = {'version': 'test reference'}
    before = prompts(shot)
    shot['generation_rhythm'] = 'allocated-v1'
    assert prompts(shot) == before
    assert all('Allocate about' not in p for p in before)


def test_overlap_is_counted_once_and_bad_timeline_cannot_be_prompted():
    from test_performance_outcomes import outcome_shot
    shot = outcome_shot()
    shot['action_beats'][1].update(start_seconds=1.8,
        flow_from_previous='attacker continues the preceding contact without resetting')
    shot['generation_rhythm'] = 'allocated-v1'
    expected = planned_interaction_seconds([shot])
    assert expected < sum(b['end_seconds'] - b['start_seconds']
        for b in shot['action_beats'] if b['activity'] == 'interaction')
    for prompt in prompts(shot):
        assert f'Allocate about {expected:g}s' in prompt
    shot['action_beats'][0]['end_seconds'] = float('nan')
    with pytest.raises(ValueError):
        compile_action_contract(shot)


@pytest.mark.parametrize('version', ['', 'allocated-v2', True, {}])
def test_unknown_generation_protocol_fails_closed(version):
    from test_performance_outcomes import outcome_shot
    shot = outcome_shot()
    shot['generation_rhythm'] = version
    with pytest.raises(ValueError, match='未知'):
        compile_action_contract(shot)


def test_real_camera_failure_is_rejected_at_source_without_touching_record():
    from pipeline.action_authoring import authored_shot_response_schema
    from pipeline.authoring_context import SceneLocation, contextual_shot_schema
    from pipeline.authoring_schedule import scheduled_shot_schema, bind_authored_schedule
    from pipeline.camera_contract import camera_contract_issues
    from pipeline.llm_options import ark_output_schema
    from pipeline.production_plan import build_production_plan
    root = Path(__file__).parent / 'fixtures'
    record = json.loads((root / 'real_camera_authoring_failure.json').read_text())
    assert hashlib.sha256(record['text'].encode()).hexdigest() == record['sha256']
    response = json.loads(json.loads(record['text'])['content'])
    before = deepcopy(response)
    context = json.loads((root / 'real_entry_scope_failure.json').read_text())
    identity = json.loads(context['sources']['planning_identity.json']['text'])
    slot = build_production_plan('action', 15, intent=context['intent'])['slots'][1]
    assert response['shot']['camera']['primary_movement'] == 'tracking right'
    assert camera_contract_issues(response['shot'])
    base = contextual_shot_schema(authored_shot_response_schema(
        [c['name'] for c in identity['characters']], require_outcomes=True,
        scene_entities=identity['scene_entities'], bound_handoffs=True),
        [SceneLocation(**s) for s in identity['locations']], identity['scene_entities'])
    schema = scheduled_shot_schema(base, slot)
    for shape in (schema, ark_output_schema(schema)):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(response, shape)
    with pytest.raises(ValueError, match='运镜无法执行'):
        bind_authored_schedule(response['shot'], slot)
    assert response == before


def test_all_registered_camera_moves_survive_wire_lowering_and_execution():
    from pipeline.camera_contract import _MOVES, compile_camera
    from pipeline.authoring_schedule import bind_authored_schedule
    from pipeline.llm_options import ark_output_schema
    from test_authoring_schedule import raw_shot, schema_for, slot
    schema = schema_for(slot())
    assert schema['$defs']['CameraSpec']['properties']['primary_movement']['enum'] == list(_MOVES)
    for movement in _MOVES:
        raw = raw_shot()
        raw['camera'].update(primary_movement=movement, speed='fast')
        wire = {'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}
        for shape in (schema, ark_output_schema(schema)):
            jsonschema.validate(wire, shape)
        lowered = bind_authored_schedule(raw, slot())
        assert compile_camera(lowered)[1]['camera_movement'] == movement
        assert lowered['camera'] == raw['camera']


def test_invalid_camera_stops_formal_creation_before_next_shot_without_retry(monkeypatch):
    from pipeline.storyboard import generate_storyboard
    from test_director_intent import Responses
    brief, intent, raw, _ = scenario_story('货物交接')
    raw['shots'][1]['camera']['primary_movement'] = 'tracking right'
    before = deepcopy(raw)
    client = Responses(intent, raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    with pytest.raises(ValueError, match='运镜无法执行'):
        generate_storyboard(brief, target_duration=15)
    assert len(client.calls) == 4  # intent, identity, first shot, second shot
    assert raw == before


@pytest.mark.asyncio
async def test_rhythm_survives_formal_submission_continuation_export_and_resume(scheduled_services):
    from pipeline.orchestrator import VideoPipeline
    import config
    brief, services = scheduled_services('双人舞')
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'],
                             paid_take_budget=3, take_policy='single_pass')
    output = await pipeline.run(brief)
    storyboard = json.loads((pipeline.workspace / 'storyboard.json').read_text())
    for shot, submitted, review in zip(storyboard['shots'], services.submissions, services.reviews):
        target = planned_interaction_seconds([shot])
        assert f'Allocate about {target:g}s of this {shot["duration"]:g}s shot' in submitted['prompt']
        assert 'Show at least 0s' not in submitted['prompt']
        assert submitted['model'] == config.SEEDANCE_MODEL
        assert review['performance_policy']['min_interaction_seconds'] == 0
        assert 'Allocate about' not in json.dumps(review)
    assert len(services.submissions) == 3 and len(services.entries) == 2
    counts = [len(v) for v in (services.submissions, services.entries, services.reviews, services.author.calls)]
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == [len(v) for v in (services.submissions, services.entries, services.reviews, services.author.calls)]
