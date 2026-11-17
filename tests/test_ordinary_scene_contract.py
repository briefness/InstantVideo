"""真实普通场景失败重放；原稿不作修正，不将接口验收当作语义通过。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.production_plan import build_production_plan
from pipeline.storyboard import _compile_storyboard_contract, _build_shot_batch_prompt
from pipeline.readiness import coverage_readiness_issues
from pipeline.authoring_schedule import bind_authored_schedule, scheduled_shot_schema
from test_authoring_schedule import schema_for, raw_shot, passive, slot
from test_authoring_schedule import interaction, assert_wire_valid


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline regression attempted network'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline regression attempted network'))


def records():
    return json.loads((Path(__file__).parent / 'fixtures/real_ordinary_scene_planning.json').read_text())


@pytest.mark.parametrize('record', records(), ids=lambda r: r['id'])
def test_captured_drafts_are_exact_and_still_expose_unresolved_legacy_multi_actor_contract(record):
    digests = {'suspense': '7d4ef8ea689ae71668e97a079ce2af3749a854f4410f0c174b1b466aa13321d8',
               'observational': 'f9b239dbd6665270172ebfcc6f75311af904cf3168d0a81494f880de1a905053',
               'restrained': 'ff12415fb02585cdb19df4aec14eeef45df17df813bec41c1eaf14cabd391d32'}
    assert hashlib.sha256(record['source_text'].encode()).hexdigest() == digests[record['id']]
    raw = json.loads(record['source_text'])
    before = deepcopy(raw)
    plan = build_production_plan('balanced', 24, intent=record['intent'])
    compiled = _compile_storyboard_contract(deepcopy(raw), '16:9', '480p', record['style'], production_plan=plan)
    assert any('多个主动动作执行者' in issue for shot in compiled['shots'] for issue in coverage_readiness_issues(shot))
    assert raw == before


@pytest.mark.parametrize('record', records(), ids=lambda r: r['id'])
def test_explicit_establish_coverage_survives_compilation_of_real_source(record):
    raw = json.loads(record['source_text'])
    assert raw['shots'][0]['coverage_role'] == 'establish'
    plan = build_production_plan('balanced', 24, intent=record['intent'])
    compiled = _compile_storyboard_contract(deepcopy(raw), '16:9', '480p', record['style'], production_plan=plan)
    assert compiled['shots'][0]['coverage_role'] == 'establish'


def test_zero_quota_accepts_ordered_passive_performers_without_inventing_interaction():
    current = {**slot(0), 'authoring_protocol': 'scene-units-v1'}
    schema = schema_for(current)
    schedule = {'first': passive(), 'following': [{**passive('hold'), 'actor': 'defender'}]}
    result = bind_authored_schedule(raw_shot(schedule), current)
    assert [u['actor'] for u in result['action_plan']] == ['attacker', 'defender']
    assert all(u['kind'] != 'interaction' for u in result['action_plan'])
    assert schema['properties']['shot']['properties']['action_plan']['type'] == 'object'


def test_batch_input_carries_exact_predecessor_state_without_rewriting_it():
    record = records()[0]
    raw = json.loads(record['source_text'])
    plan = build_production_plan('balanced', 24, intent=record['intent'])
    spine = {k: v for k, v in raw.items() if k != 'shots'}
    spine['shot_intents'] = [{'shot_id': 4, 'primary_action': raw['shots'][3]['primary_action']}]
    prior = deepcopy(raw['shots'][2])
    prompt = _build_shot_batch_prompt('brief', spine, plan, plan['slots'][3:], previous_shot=prior)
    assert json.dumps(prior, ensure_ascii=False, indent=2) in prompt
    assert prior == raw['shots'][2]


@pytest.mark.parametrize('units', [
    [passive()],
    [passive(), {**passive('hold'), 'actor': 'defender'}],
    [{'kind': 'interaction', **interaction()}],
    [passive(), {'kind': 'interaction', **interaction()}, passive('hold')],
])
def test_scene_schedule_preserves_exact_order_and_content(units):
    current = {**slot(0), 'authoring_protocol': 'scene-units-v1'}
    raw = raw_shot({'first': units[0], 'following': units[1:]})
    before = deepcopy(raw)
    assert_wire_valid(raw, current)
    bound = bind_authored_schedule(raw, current)
    assert [{k: v for k, v in u.items() if k != 'id'} for u in bound['action_plan']] == units
    assert raw == before


@pytest.mark.parametrize('schedule', [
    {}, {'first': None}, {'first': passive(), 'following': 'not a list'},
    {'first': passive(), 'following': [passive()] * 12},
    {'first': {'kind': 'interaction', **interaction(), 'link_previous': True}},
    {'first': {'kind': 'interaction', **interaction(), 'overlap': True}},
    {'first': passive(), 'interaction_1': interaction()},
])
def test_invalid_scene_schedules_stop_without_repair(schedule):
    before = deepcopy(schedule)
    with pytest.raises(ValueError):
        bind_authored_schedule(raw_shot(schedule), {**slot(0), 'authoring_protocol': 'scene-units-v1'})
    assert schedule == before


def scene_case(topic):
    """独立服务标注，仅验证正式协议；不修改真实失败稿或模拟模型理解精度。"""
    from test_action_results import annotated_story
    from authoring_cases import contextual_shot_fixture, scheduled_shot_fixture
    from test_director_intent import intent_reply
    from pipeline.models import DirectorIntentSpec
    base = annotated_story()
    product = topic == 'product'
    brief = {'family': '记录家人在同一房间相互靠近和落座。',
             'ensemble': '记录排练中两位演员各自走位和停留。',
             'product': '展示桌灯灯罩的材质与缓慢旋转的外观。'}[topic]
    brief += '全片24秒，分四段。'
    reply = {**intent_reply(brief, 'product' if product else 'balanced', 'balanced'),
             'choreography': 'flexible', 'requested_duration_seconds': 24, 'duration_evidence': brief,
             'requested_shot_count': 4, 'shot_count_evidence': brief}
    intent = DirectorIntentSpec.model_validate({**reply, 'source_request': brief,
        'authoring_protocol': 'scene-units-v1'}).model_dump()
    plan = build_production_plan(intent['content_focus'], 24, intent=intent)
    catalog = ['Lamp', 'Shade'] if product else []
    identity = {k: v for k, v in base.items() if k not in {'shots', 'locations'}}
    identity.update(title=topic, scene_entities=catalog,
                    locations=[{'id': 'room', 'description': 'one shared interior'}],
                    theme_elements=catalog if product else ['attacker', 'defender'])
    if product:
        identity['characters'] = []
    actors = catalog if product else ['attacker', 'defender']
    actions = {'family': [('steps beside the chair', 'stands beside the chair'),
                          ('settles onto the chair', 'sits at the shared table')],
               'ensemble': [('walks toward the left floor mark', 'stands on the left floor mark'),
                            ('turns toward the open floor', 'faces the open floor')],
               'product': [('rotates slowly on its base', 'presents its curved profile'),
                           ('remains illuminated', 'shows its translucent linen texture')]}[topic]
    shots, wires = [], []
    previous = None
    for index, current in enumerate(plan['slots']):
        shot = deepcopy(base['shots'][0])
        shot.update(shot_id=index + 1, scene_id='room', scene_description='One shared interior with a table and clear space.',
                    visual_description='Soft window light reveals the surfaces in the shared room.',
                    key_props=catalog, continuity_props=[], mood='observant',
                    story_change=f'{topic} progresses through visible stage {index + 1}',
                    story_state_after=f'{topic} visible stage {index + 1} is complete')
        shot['action_plan'] = [{'id': f'unit_{i + 1}', 'kind': 'transition' if i == 0 else 'hold',
            'actor': actor, 'target': '', 'action': actions[i][0], 'result': actions[i][1], 'weight': 1.}
            for i, actor in enumerate(actors)]
        state = {'location': 'shared room', 'subject': ', '.join(actors),
                 'action_phase': 'the subjects are arranged in the room', 'camera': 'wide shot'}
        shot['start_state'] = deepcopy(previous or state)
        shot['end_state'] = {**state, 'action_phase': '; '.join(result for _, result in actions)}
        previous = shot['end_state']
        if product:
            shot['blocking'] = {}
            shot['camera']['screen_positions'] = {'Lamp': 'left', 'Shade': 'right'}
        wires.append({'shot': scheduled_shot_fixture(
            {k: v for k, v in contextual_shot_fixture(shot).items() if k != 'shot_id'}, current, execution_timing=True)})
        shots.append(shot)
    return brief, reply, identity, wires, shots


@pytest.mark.parametrize('topic', ['family', 'ensemble', 'product'])
def test_formal_non_dense_flow_to_provider_preserves_every_subject_and_state(monkeypatch, tmp_path, topic):
    import jsonschema
    from pipeline.storyboard import generate_storyboard
    from pipeline.provider_prompt import compile_generation_preview
    from pipeline.production_plan import production_plan_issues
    from pipeline.action_timing import action_timing_issues
    from test_director_intent import Responses
    brief, reply, identity, wires, sources = scene_case(topic)
    before = deepcopy(wires)
    client = Responses(reply, identity, *wires)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    saved = {}
    result = generate_storyboard(brief, target_duration=24, allow_repair=False,
        save_diagnostic=lambda name, value: saved.update({name: deepcopy(value)}))
    assert len(client.calls) == 6  # 意图 + 故事 + 四镜；不重试，不调用视频。
    assert not production_plan_issues(result)
    assert all(not action_timing_issues(s) for s in result['shots'])
    assert all(not coverage_readiness_issues(s) for s in result['shots'])
    assert sum(s['duration'] for s in result['shots']) == 24
    from pipeline.planning_review import ensure_planning_semantics
    from test_planning_semantics import verdict_for
    verdict = verdict_for(result)
    for unit in verdict['units']:
        unit.pop('result_admission', None)  # 本组没有交互结果候选。
    reviewer = Responses(verdict)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: reviewer)
    def save_review(name, data):
        (tmp_path / name).write_text(json.dumps(data, ensure_ascii=False))
    original = deepcopy(result)
    result = ensure_planning_semantics(result, tmp_path, save_review)
    assert len(reviewer.calls) == 1
    assert all(s['result_admission'] == 'reviewed-results-v1' for s in result['shots'])
    assert ensure_planning_semantics(original, tmp_path, save_review) == result
    assert len(reviewer.calls) == 1
    preview = compile_generation_preview(result, model='doubao-seedance-2.0-fast')
    for index, (call, source, shot, row) in enumerate(zip(client.calls[2:], sources, result['shots'], preview['shots'])):
        jsonschema.validate(wires[index], call['response_format']['json_schema']['schema'])
        context = json.loads(call['messages'][1]['content'])
        if index:
            assert context['previous_shot'] == {'shot_id': index, **wires[index - 1]['shot']}
        else:
            assert context['previous_shot'] is None
        assert [b['actor'] for b in shot['action_beats']] == [u['actor'] for u in source['action_plan']]
        assert shot['interaction_geometry']['effect_phase'] == 'none'
        for unit in source['action_plan']:
            assert unit['action'] in row['profiles']['normal']['prompt']
            assert unit['result'] in row['profiles']['normal']['prompt']
            assert unit['result'] in row['profiles']['policy_safe']['prompt']
        assert all(not p['preflight_issues'] for p in row['profiles'].values()), {
            name: p['preflight_issues'] for name, p in row['profiles'].items()}
    if topic == 'product':
        assert all(not s['characters'] for s in result['shots'])
        assert all(s['required_visible_entities'] == ['Lamp', 'Shade'] for s in result['shots'])
    assert wires == before


def test_protocol_is_program_owned_and_missing_marker_keeps_legacy_plan():
    from pipeline.director_intent import resolve_director_intent, director_intent_response_schema
    from pipeline.models import DirectorIntentSpec
    from test_director_intent import Responses
    brief, reply, *_ = scene_case('family')
    intent = resolve_director_intent(Responses(reply), brief)
    assert intent['authoring_protocol'] == 'scene-units-v1'
    assert 'authoring_protocol' not in json.dumps(director_intent_response_schema(brief))
    for value in ['scene-units-v1', 'invented', None]:
        with pytest.raises(ValueError, match='未按关键词'):
            resolve_director_intent(Responses({**reply, 'authoring_protocol': value}), brief)
    legacy = {k: v for k, v in intent.items() if k != 'authoring_protocol'}
    assert DirectorIntentSpec.model_validate(legacy).model_dump() == legacy
    assert all('authoring_protocol' not in s for s in build_production_plan('balanced', 24, intent=legacy)['slots'])
    exact = resolve_director_intent(Responses({**reply, 'choreography': 'exact'}), brief)
    assert exact['authoring_protocol'] == 'scene-units-v1'
    assert exact['completion_goals'] == intent['completion_goals']


@pytest.mark.parametrize('unknown_at', ['actor', 'blocking', 'target'])
def test_scene_objects_do_not_bypass_identity_or_entity_registration(unknown_at):
    from pipeline.participants import structured_entity_reference_issues
    shot = {'shot_id': 1, 'characters': [], 'key_props': ['Lamp'],
            'production_slot': {'authoring_protocol': 'scene-units-v1'},
            'action_beats': [{'actor': 'Lamp', 'target': '', 'activity': 'hold'}]}
    if unknown_at == 'blocking':
        shot['blocking'] = {'Lamp': {'action_target': ''}}
    else:
        shot['action_beats'][0][unknown_at] = 'unregistered'
    assert structured_entity_reference_issues(shot, [])


def test_no_character_schema_cannot_request_an_impossible_required_interaction():
    from pipeline.action_authoring import authored_shot_response_schema
    base = authored_shot_response_schema([], require_outcomes=True, scene_entities=['Lamp'],
                                        bound_handoffs=True, allow_scene_subjects=True)
    with pytest.raises(ValueError, match='执行人物'):
        scheduled_shot_schema(base, {**slot(1), 'authoring_protocol': 'scene-units-v1'})


@pytest.mark.parametrize('accepted', [True, False])
@pytest.mark.asyncio
async def test_plan_only_orchestrator_reaches_preview_or_stops_once_without_payment(tmp_path, monkeypatch, accepted):
    import config
    from pipeline.orchestrator import VideoPipeline
    from test_director_intent import Responses
    from test_planning_semantics import verdict_for
    brief, reply, identity, wires, _ = scene_case('family')
    author = Responses(reply, identity, *wires)
    review_calls = []
    class Reviewer:
        def __init__(self):
            from types import SimpleNamespace
            self.chat = SimpleNamespace(completions=self)
        def create(self, **kwargs):
            context = json.loads(kwargs['messages'][-1]['content'])
            verdict = verdict_for(context['storyboard'])
            for unit in verdict['units']:
                unit.pop('result_admission', None)
            verdict['units'][0]['classification_matches_description'] = accepted
            response = Responses(verdict).create(**kwargs)
            review_calls.append(kwargs)
            return response
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: author)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: Reviewer())
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **_: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0, planning_repair=False)
    if accepted:
        await pipeline.run(brief)
        preview = json.loads((pipeline.workspace / 'generation_preview.json').read_text())
        assert len(preview['shots']) == 4
        assert all(not p['preflight_issues'] for s in preview['shots'] for p in s['profiles'].values())
    else:
        with pytest.raises(ValueError, match='分镜语义验收未通过'):
            await pipeline.run(brief)
        restored = VideoPipeline.from_workspace(pipeline.workspace)
        with pytest.raises(ValueError, match='分镜语义验收未通过'):
            await restored.run()
        assert not (pipeline.workspace / 'generation_preview.json').exists()
    assert len(author.calls) == 6 and len(review_calls) == 1
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations
