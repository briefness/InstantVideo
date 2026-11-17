"""时序与运动来源分别建模；原始失败记录不改写，不访问模型服务。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.action_authoring import AuthoredShot, compile_action_units
from pipeline.authoring_context import bind_authored_context, SceneLocation
from pipeline.authoring_schedule import bind_authored_schedule
from pipeline.authoring_timing import bind_execution_timing
from pipeline.action_timing import action_timing_issues, timing_projection, sequence_events

NEW_PROTOCOL = 'story-context-v9-temporal-relations'


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('unexpected network access'))


def captured():
    record = json.loads((Path(__file__).parent / 'fixtures/real_temporal_overlap_failure.json').read_text())
    assert hashlib.sha256(record['response_text'].encode()).hexdigest() == record['response_sha256']
    raw = json.loads(json.loads(record['response_text'])['content'])['shot']
    context = record['context']
    slot = {**context['slot'], 'authoring_protocol': 'scene-units-v1', 'choreography': 'flexible'}
    return raw, slot, context


def compile_record(raw, slot, context, version):
    bound = bind_execution_timing({'shot_id': 3, **raw}, version=version)
    bound = bind_authored_context(bind_authored_schedule(bound, slot),
        [SceneLocation.model_validate(x) for x in context['locations']], context['scene_entities'])
    return compile_action_units(AuthoredShot.model_validate(bound), slot)


def test_exact_record_under_new_temporal_contract_has_no_invented_motion_link():
    raw, slot, context = captured()
    original = deepcopy(raw)
    # 差分协议实验，不迁移旧运行或补写正文。原协议仍应精确复现失败。
    with pytest.raises(ValueError, match='共同参与者'):
        compile_record(raw, slot, context, context['authoring_protocol'])
    beats = compile_record(raw, slot, context, NEW_PROTOCOL)
    assert raw == original
    assert beats[2]['start_seconds'] < beats[1]['end_seconds']
    assert not any(b.get('flow_from_previous') for b in beats)
    source = [raw['action_plan']['first'], *raw['action_plan']['following']]
    assert [(b['actor'], b['target'], b['action'], b['visible_result']) for b in beats] == [
        (u['actor'], u['target'], u['action'], u['result']) for u in source]
    shot = {'shot_id': 3, 'duration': slot['duration'], 'production_slot': slot, 'action_beats': beats}
    assert action_timing_issues(shot) == []
    assert timing_projection(shot)['intervals'][2]['temporal_relation'] == 'overlap_previous'
    assert sequence_events(shot)[1]['temporal_relation'] == 'overlap_previous'


def scene(units, duration=6):
    from test_authoring_schedule import raw_shot
    slot = {'shot_id': 1, 'duration': duration, 'authoring_protocol': 'scene-units-v1',
            'choreography': 'flexible', 'narrative_function': 'progress'}
    raw = raw_shot({'first': units[0], 'following': units[1:]})
    bound = bind_authored_schedule(bind_execution_timing(raw, version=NEW_PROTOCOL), slot)
    beats = compile_action_units(AuthoredShot.model_validate(bound), slot)
    return {**raw, 'duration': duration, 'production_slot': slot, 'action_beats': beats,
            'interaction_geometry': next((b['interaction_geometry'] for b in reversed(beats)
                                          if b['activity'] == 'interaction'), {'effect_phase': 'none'})}


def interaction(actor, target, timing='sequential', source='none', action='changes the object state'):
    from test_authoring_schedule import interaction as base
    unit = base()
    unit.pop('link_previous')
    return {**unit, 'kind': 'interaction', 'actor': actor, 'target': target,
            'action': action, 'timing': timing, 'motion_source': source,
            'resolution': {'states': {target: [{'state': 'object reaches the requested state'}]}}}


@pytest.mark.parametrize('kind', ['interaction', 'transition', 'performance', 'hold'])
@pytest.mark.parametrize('duration', [2, 6, 15])
@pytest.mark.parametrize('topic,a,b,x,y', [
    ('舞台协作', '舞者', '灯光师', '丝带', '灯具'),
    ('厨房准备', '厨师', '助手', '面团', '餐盘'),
    ('设备检修', '技师', '操作员', '阀门', '控制台'),
])
def test_parallel_subjects_reach_timing_prompts_and_review_without_fabricated_handoff(kind, duration, topic,a,b,x,y):
    from pipeline.causality import compile_action_contract
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    from pipeline.performance import performance_projection
    from pipeline.film_rhythm import planned_interaction_seconds
    first = (interaction(a,x,action=topic + '中的第一项工作') if kind == 'interaction' else
             {'kind': kind, 'actor': a, 'target': x, 'action': topic + '中的移动或持续表演', 'result': '本项持续可见'})
    second = interaction(b,y,'overlap_previous',action=topic + '中的独立工作')
    shot = scene([first, second], duration)
    assert not action_timing_issues(shot)
    contract = compile_action_contract(shot)
    assert contract.review_projection['action_timing'] == timing_projection(shot)
    assert performance_projection(shot)['event_order'][-1]['after_effects'] == []
    for prompt in (compile_normal_provider_prompt(shot, {}, has_observed_start=False),
                   compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0)):
        assert 'begin while Beat 1 is still underway' in prompt
        assert b in prompt and y in prompt
        assert 'continuous momentum handoff' not in prompt
    assert planned_interaction_seconds([shot]) <= duration


@pytest.mark.parametrize('timing', ['sequential', 'overlap_previous'])
@pytest.mark.parametrize('source', ['none', 'previous'])
@pytest.mark.parametrize('shared', [True, False])
def test_temporal_and_motion_axes_are_independent(timing, source, shared):
    units = [interaction('A','Object'), interaction('B','Object' if shared else 'Other',timing,source)]
    if source == 'previous' and not shared:
        with pytest.raises(ValueError, match='共同参与者'):
            scene(units)
    else:
        shot = scene(units)
        assert not action_timing_issues(shot)
        a,b = shot['action_beats']
        assert bool(b.get('flow_from_previous')) == (source == 'previous')
        assert (b['start_seconds'] < a['end_seconds']) == (timing == 'overlap_previous')


@pytest.mark.parametrize('kind', ['interaction','transition','hold','performance'])
def test_first_unit_cannot_overlap_or_link_missing_predecessor(kind):
    first = interaction('A','Object','overlap_previous') if kind == 'interaction' else {
        'kind': kind, 'actor': 'A', 'action': 'moves', 'result': 'arrives', 'timing': 'overlap_previous'}
    with pytest.raises(ValueError, match='前项'):
        scene([first])


@pytest.mark.parametrize('mutation', ['missing', 'contradiction', 'gap', 'backward', 'overflow', 'false_link'])
def test_explicit_relations_do_not_accept_corrupted_execution(mutation):
    shot = scene([interaction('A','Object'), interaction('B','Other','overlap_previous')])
    b = shot['action_beats'][1]
    if mutation == 'missing': b.pop('temporal_relation')
    elif mutation == 'contradiction': b['temporal_relation'] = 'sequential'
    elif mutation == 'gap': b['start_seconds'] = shot['action_beats'][0]['end_seconds'] + .1
    elif mutation == 'backward': b['start_seconds'] = -.1
    elif mutation == 'overflow': b['end_seconds'] = 10
    else: b['flow_from_previous'] = 'invented carry'
    assert action_timing_issues(shot)
    assert timing_projection(shot) == {}


def evidence():
    from pipeline.performance import OBSERVED_TIMES
    return [{'event_id': i, **dict(zip(OBSERVED_TIMES, times)),
             'active_intervals': [{'start_seconds': times[0], 'end_seconds': times[3]}]}
            for i,times in [(1,(0.,2.,2.1,2.8)), (2,(1.,1.2,1.5,3.))]]


def test_independent_effects_may_complete_in_reverse_order_but_dependencies_may_not():
    from pipeline.performance import performance_projection, measured_performance, observed_performance_issues
    policy = performance_projection(scene([interaction('A','Object'), interaction('B','Other','overlap_previous')]))
    assert not observed_performance_issues(evidence(), policy)
    assert measured_performance(evidence(), policy, [0.,1.,2.,3.,4.,5.,6.])['status'] == 'passed'
    sequential = performance_projection(scene([interaction('A','Object'), interaction('B','Other')]))
    assert observed_performance_issues(evidence(), sequential)
    with pytest.raises(ValueError, match='依赖'):
        measured_performance(evidence(), sequential, [0.,1.,2.,3.,4.,5.,6.])
    linked = performance_projection(scene([interaction('A','Object'), interaction('B','Object','overlap_previous','previous')]))
    assert observed_performance_issues(evidence(), linked)
    legacy = {k:v for k,v in policy.items() if k != 'event_order'}
    assert observed_performance_issues(evidence(), legacy)


def test_passive_concurrency_is_not_new_interaction_or_phantom_time():
    units = [{'kind': 'performance', 'actor': actor, 'action': 'moves independently',
              'result': 'movement continues', 'timing': timing}
             for actor,timing in [('A','sequential'), ('B','overlap_previous')]]
    shot = scene(units)
    assert not action_timing_issues(shot)
    assert sequence_events(shot) == []
    from pipeline.film_rhythm import planned_interaction_seconds
    assert planned_interaction_seconds([shot]) == 0


def test_actual_record_is_accepted_by_new_wire_schema_without_changing_any_field():
    import jsonschema
    from pipeline.action_authoring import authored_shot_response_schema
    from pipeline.authoring_context import contextual_shot_schema
    from pipeline.authoring_schedule import scheduled_shot_schema
    from pipeline.authoring_timing import execution_timing_schema
    raw, slot, context = captured()
    base = authored_shot_response_schema(['Xiao An','Lin Ayi'], require_outcomes=True,
        scene_entities=context['scene_entities'], bound_handoffs=True, allow_scene_subjects=True)
    base = contextual_shot_schema(base, [SceneLocation.model_validate(x) for x in context['locations']], context['scene_entities'])
    schema = execution_timing_schema(scheduled_shot_schema(base,slot), version=NEW_PROTOCOL)
    jsonschema.validate({'shot': raw},schema)


@pytest.mark.parametrize('topic', ['双人舞','货物交接','绳索救援'])
def test_real_sdk_formal_multishot_flow_carries_relations_into_generation_without_retry(monkeypatch,topic):
    import jsonschema
    from pipeline.storyboard import generate_storyboard
    from pipeline.provider_prompt import compile_normal_provider_prompt
    from test_creation_schedule_flow import scenario_story
    from test_authoring_transport import http_client
    from authoring_cases import contextual_shot_fixture, scheduled_shot_fixture
    brief, intent, raw, plan = scenario_story(topic)
    raw['scene_entities'] = ['Station A','Station B']
    replies = []
    for shot,slot in zip(raw['shots'],plan['slots']):
        shot['key_props'] = ['Station A','Station B']
        for i,unit in enumerate(shot['action_plan']):
            if unit['kind'] == 'interaction':
                actor,target = ('Lead','Station A') if i%2 == 0 else ('Partner','Station B')
                unit.update(actor=actor,target=target,action=f'{actor} adjusts {target}',
                    result=f'{target} reaches position',response='state_change',link_previous=False,overlap=False,
                    resolution={'states':{target:[{'state':'reaches position','importance':'required'}]}})
        reply = scheduled_shot_fixture(contextual_shot_fixture(shot),slot,execution_timing=True)
        # 独立测试服务标注，非真实失败稿修补：准备期间另一参与者开始自己的工作。
        if shot['shot_id'] == 1:
            reply['action_plan']['interaction_1']['timing'] = 'overlap_previous'
        replies.append({'shot':{k:v for k,v in reply.items() if k != 'shot_id'}})
    before = deepcopy(replies)
    values = [{k:v for k,v in intent.items() if k!='source_request'},
              {k:v for k,v in raw.items() if k!='shots'},*replies]
    sdk,calls = http_client(monkeypatch,values)
    with sdk:
        story = generate_storyboard(brief,target_duration=15,allow_repair=False)
    assert len(calls) == 5 and replies == before
    for call,reply in zip(calls[2:],replies):
        assert json.loads(call['messages'][-1]['content'])['authoring_protocol'] == NEW_PROTOCOL
        jsonschema.validate(reply, call['response_format']['json_schema']['schema'])
    first = story['shots'][0]
    assert first['action_beats'][1]['start_seconds'] < first['action_beats'][0]['end_seconds']
    assert not any(b.get('flow_from_previous') for b in first['action_beats'])
    assert 'begin while Beat 1 is still underway' in compile_normal_provider_prompt(first,story,has_observed_start=False)
    assert all(not action_timing_issues(shot) for shot in story['shots'])


def test_filtered_review_retains_dependencies_only_within_its_evidence_scope():
    from test_completion_intent import admitted_capture
    from pipeline.causality import compile_action_contract
    _,_,story = admitted_capture()
    shot = deepcopy(story['shots'][0])
    for beat in shot['action_beats']:
        beat['temporal_relation'] = 'sequential'
    projection = compile_action_contract(shot).review_projection
    ids = [event['event_id'] for event in projection['action_events']]
    order = projection['performance_policy']['event_order']
    assert [row['event_id'] for row in order] == ids
    assert all(set(row['after_ends']+row['after_effects']) <= set(ids) for row in order)
    assert all('temporal_relation' not in event or event['event_id']-1 in ids
               for event in projection['action_events'])


@pytest.mark.parametrize('value', [None, True, 'future', '', [], {}])
def test_invalid_motion_source_and_protocol_never_fall_back(value):
    raw, slot, context = captured()
    raw['action_plan']['first']['motion_source'] = value
    with pytest.raises(ValueError):
        compile_record(raw,slot,context,NEW_PROTOCOL)
    with pytest.raises(ValueError,match='未知'):
        bind_execution_timing({'action_plan': {}},version='future')


def test_declared_relation_cannot_be_ignored_when_time_window_is_missing():
    assert action_timing_issues({'shot_id': 1, 'duration': 6, 'action_beats': [
        {'actor':'A','action':'works','temporal_relation':'overlap_previous'}]})


def test_final_confirmation_cannot_overlap_the_work_it_confirms():
    units = [interaction('A','Object'), {'kind':'hold','actor':'A','target':'Object',
             'action':'shows established result','result':'result stays visible','timing':'overlap_previous'}]
    shot = scene(units)
    # 普通持续表演可以并行；一旦承担独立的结果确认时长，必须在作用完成之后。
    assert not action_timing_issues(shot)
    shot['production_slot']['result_hold_seconds'] = .5
    assert action_timing_issues(shot)
    from test_authoring_schedule import raw_shot
    raw = raw_shot({'first':units[0], 'result_hold':{k:v for k,v in units[1].items() if k!='kind'}})
    with pytest.raises(ValueError,match='结果展示'):
        bind_execution_timing(raw,version=NEW_PROTOCOL)


def test_passive_first_boundary_is_representable_but_never_overlaps_missing_predecessor():
    import jsonschema
    from pipeline.action_authoring import authored_shot_response_schema
    from pipeline.authoring_schedule import scheduled_shot_schema
    from pipeline.authoring_timing import execution_timing_schema
    from test_authoring_schedule import raw_shot
    slot = {'shot_id':1,'duration':6,'authoring_protocol':'scene-units-v1','min_interaction_events':0}
    base = authored_shot_response_schema([], require_outcomes=True, scene_entities=['product'],
        bound_handoffs=True,allow_scene_subjects=True)
    schema = execution_timing_schema(scheduled_shot_schema(base,slot),version=NEW_PROTOCOL)
    first = {'kind':'performance','actor':'product','action':'rotates','result':'reverse side is visible',
             'timing':'overlap_previous'}
    raw = raw_shot({'first':first})
    # 检查动作计划的可达 schema，避免无关测试人物 blocking 干扰边界。
    schedule_schema = {**schema['properties']['shot']['properties']['action_plan'],'$defs':schema['$defs']}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw['action_plan'],schedule_schema)
    first['timing'] = 'sequential'
    jsonschema.validate(raw['action_plan'],schedule_schema)


def test_visual_review_gets_dependencies_without_planned_times_or_missing_passive_ids():
    from pipeline.semantic_review import _review_contract
    first = {'kind':'transition','actor':'A','action':'approaches','result':'arrives'}
    shot = scene([first,interaction('B','Object','overlap_previous'),interaction('A','Other')])
    review = _review_contract(shot)
    assert 'action_timing' not in review
    events = review['action_events']
    assert 'temporal_relation' not in events[0]  # 前驱为未作为独立交互验收的移动。
    assert events[1]['temporal_relation'] == 'sequential'
    assert review['performance_policy']['event_order'][-1]['after_ends'] == [2]
    assert all('start_seconds' not in e for e in events)

