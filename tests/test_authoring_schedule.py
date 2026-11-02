"""既有槽位义务绑定到响应形状；内容、顺序和执行合同保持单一来源。"""
from copy import deepcopy

import jsonschema
import pytest

from pipeline.action_authoring import AuthoredShot, authored_shot_response_schema, compile_action_units
from pipeline.authoring_schedule import bind_authored_schedule, scheduled_shot_schema
from test_action_results import annotated_story


def interaction():
    return {'actor': 'attacker', 'target': 'defender', 'action': 'extends a hand toward the partner',
            'result': 'the partner shifts aside while keeping balance', 'weight': 1.,
            'effect': {'mode': 'directed_path', 'motion': 'sweep', 'source': 'the extended hand',
                       'region': 'the space between the partners', 'unaffected': 'the surrounding floor'},
            'response': 'evasion', 'link_previous': False,
            'resolution': {'states': {'defender': [
                {'state': 'remains balanced outside the path', 'importance': 'required'}]}}}


def passive(kind='transition'):
    return {'kind': kind, 'actor': 'attacker', 'target': '', 'action': 'steps into the open space',
            'result': 'stands within reach', 'weight': 1.}


def slot(count=2, hold=0):
    return {'shot_id': 1, 'duration': 5, 'min_interaction_events': count,
            'result_hold_seconds': hold, 'narrative_function': 'progress', 'choreography': 'flexible'}


def raw_shot(schedule=None):
    shot = deepcopy(annotated_story()['shots'][0])
    shot['action_plan'] = schedule if schedule is not None else {
        'interaction_1': interaction(), 'interaction_2': interaction()}
    return shot


def schema_for(current_slot):
    base = authored_shot_response_schema(['attacker', 'defender'], require_outcomes=True,
                                          scene_entities=['Door', 'Dust'], bound_handoffs=True)
    return scheduled_shot_schema(base, current_slot)


def assert_wire_valid(raw, current_slot):
    jsonschema.validate({'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}, schema_for(current_slot))


def test_slot_keys_are_program_owned_without_changing_shared_schema():
    base = authored_shot_response_schema(['attacker', 'defender'], require_outcomes=True,
                                          scene_entities=['Door', 'Dust'], bound_handoffs=True)
    before = deepcopy(base)
    schema = scheduled_shot_schema(base, slot(3, .5))
    plan = schema['properties']['shot']['properties']['action_plan']
    for branch in plan['oneOf']:
        assert {'interaction_1', 'interaction_2', 'interaction_3', 'result_hold'} <= set(branch['required'])
        assert branch['additionalProperties'] is False
    assert base == before
    assert 'InteractionUnit' not in schema['$defs'] and 'PassiveUnit' not in schema['$defs']
    assert 'ScheduledRequiredInteraction' in schema['$defs']


def test_omitted_opening_adds_no_actions():
    raw = raw_shot()
    before = deepcopy(raw)
    assert_wire_valid(raw, slot())
    result = bind_authored_schedule(raw, slot())
    assert [u['id'] for u in result['action_plan']] == ['unit_1', 'unit_2']
    assert all(u['kind'] == 'interaction' for u in result['action_plan'])
    assert [{k: v for k, v in u.items() if k not in ('id', 'kind')}
            for u in result['action_plan']] == [interaction(), interaction()]
    assert raw == before


def test_fixed_key_order_and_extra_actions_preserve_all_content():
    extra = {'kind': 'interaction', **interaction()}
    extra['action'] = 'turns the same reaching motion into another approach'
    schedule = {'interaction_2': interaction(), 'opening': {'first': passive()},
                'interaction_1': {**interaction(), 'following': [passive('hold'), extra]}}
    raw = raw_shot(schedule)
    before = deepcopy(raw)
    assert_wire_valid(raw, slot())
    result = bind_authored_schedule(raw, slot())
    units = result['action_plan']
    assert [u['kind'] for u in units] == ['transition', 'interaction', 'hold', 'interaction', 'interaction']
    assert [u['action'] for u in units] == [passive()['action'], interaction()['action'],
        passive()['action'], extra['action'], interaction()['action']]
    assert len({u['id'] for u in units}) == len(units)
    assert raw == before


def test_same_actor_linked_overlap_survives_existing_compiler():
    raw = raw_shot()
    raw['action_plan']['interaction_2'].update(link_previous=True, overlap=True, weight=2.)
    result = bind_authored_schedule(raw, slot())
    assert_wire_valid(raw, slot())
    parsed = AuthoredShot.model_validate(result)
    beats = compile_action_units(parsed, slot())
    assert beats[0]['actor'] == beats[1]['actor']
    assert beats[1]['start_seconds'] < beats[0]['end_seconds']
    assert 'flow_from_previous' in beats[1]
    assert result['action_plan'][1]['link_previous'] is True
    assert result['action_plan'][1]['overlap'] is True and result['action_plan'][1]['weight'] == 2.


def test_required_hold_is_last_and_uses_existing_duration_allocation():
    raw = raw_shot({'interaction_1': interaction(),
                    'result_hold': {k: v for k, v in passive('hold').items() if k != 'kind'}})
    current_slot = slot(1, .5)
    assert_wire_valid(raw, current_slot)
    result = bind_authored_schedule(raw, current_slot)
    beats = compile_action_units(AuthoredShot.model_validate(result), current_slot)
    assert [u['kind'] for u in result['action_plan']] == ['interaction', 'hold']
    assert beats[-1]['end_seconds'] - beats[-1]['start_seconds'] >= .5
    assert beats[-1]['end_seconds'] == 5
    raw['action_plan'].pop('result_hold')
    with pytest.raises(jsonschema.ValidationError):
        assert_wire_valid(raw, current_slot)
    with pytest.raises(ValueError, match='必需作用'):
        bind_authored_schedule(raw, current_slot)


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(action_plan=[]),
    lambda s: s['action_plan'].pop('interaction_2'),
    lambda s: s['action_plan'].update(interaction_3=interaction()),
    lambda s: s['action_plan'].update(result_hold=passive()),
    lambda s: s['action_plan'].update(opening={}),
    lambda s: s['action_plan'].update(opening=[]),
    lambda s: s['action_plan'].update(opening=None),
    lambda s: s['action_plan'].update(opening=[{'kind': 'interaction', **interaction()}]),
    lambda s: s['action_plan']['interaction_1'].update(id='model-owned'),
    lambda s: s['action_plan']['interaction_1'].update(kind='interaction'),
    lambda s: s['action_plan']['interaction_1'].update(following=None),
    lambda s: s['action_plan']['interaction_1'].update(following=[{**passive(), 'id': 'override'}]),
    lambda s: s['action_plan']['interaction_1'].update(following=[{**interaction(), 'kind': 'invalid'}]),
    lambda s: s['action_plan']['interaction_1'].update(following=[{**passive(), 'unknown': True}]),
    lambda s: s['action_plan']['interaction_1'].update(unknown=True),
    lambda s: s['action_plan']['interaction_1'].pop('resolution'),
    lambda s: s['action_plan']['interaction_1'].pop('response'),
    lambda s: s['action_plan']['interaction_1'].pop('link_previous'),
    lambda s: s['action_plan']['interaction_1'].update(carry='old authority'),
    lambda s: s['action_plan']['interaction_1'].update(outcomes=[]),
    lambda s: s['action_plan']['interaction_1']['effect'].update(mode='direct_contact'),
])
def test_missing_obligations_and_ambiguous_mixed_contracts_are_rejected(mutate):
    raw = raw_shot()
    mutate(raw)
    before = deepcopy(raw)
    with pytest.raises(jsonschema.ValidationError):
        assert_wire_valid(raw, slot())
    with pytest.raises(ValueError):
        bind_authored_schedule(raw, slot())
    assert raw == before


def test_total_capacity_uses_existing_limit_without_truncating_content():
    raw = raw_shot({'interaction_1': {**interaction(), 'following': [passive() for _ in range(11)]}})
    current_slot = slot(1)
    assert_wire_valid(raw, current_slot)
    result = bind_authored_schedule(raw, current_slot)
    assert len(result['action_plan']) == 12
    AuthoredShot.model_validate(result)
    raw['action_plan']['opening'] = {'first': passive()}
    # 每个数组合法，但总容量属于本地检查；不增加组合schema或删除动作。
    assert_wire_valid(raw, current_slot)
    before = deepcopy(raw)
    with pytest.raises(ValueError, match='容量'):
        bind_authored_schedule(raw, current_slot)
    assert raw == before


@pytest.mark.parametrize('current_slot', [slot(0), slot(True), slot(13), slot(12, .5),
                                          slot(1, -1), slot(1, float('nan')), {}, None])
def test_invalid_or_over_capacity_slots_stop_before_authoring(current_slot):
    with pytest.raises(ValueError):
        schema_for(current_slot)
    with pytest.raises(ValueError):
        bind_authored_schedule(raw_shot(), current_slot)


@pytest.mark.parametrize('change', [{'id': 'manual'}, {'kind': 'hold'}, {'following': []}, {'extra': ''}])
def test_terminal_hold_cannot_override_program_fields_or_add_units(change):
    hold = {k: v for k, v in passive('hold').items() if k != 'kind'}
    hold.update(change)
    raw = raw_shot({'interaction_1': interaction(), 'result_hold': hold})
    with pytest.raises(jsonschema.ValidationError):
        assert_wire_valid(raw, slot(1, .5))
    with pytest.raises(ValueError):
        bind_authored_schedule(raw, slot(1, .5))


def test_required_obligations_survive_actual_provider_projection():
    from pipeline.llm_options import ark_output_schema
    schema = ark_output_schema(schema_for(slot(2, .5)))
    plan = schema['properties']['shot']['properties']['action_plan']
    for branch in plan['oneOf']:
        assert {'interaction_1', 'interaction_2', 'result_hold'} <= set(branch['required'])
        assert branch['additionalProperties'] is False
    assert 'InteractionUnit' not in schema['$defs'] and 'PassiveUnit' not in schema['$defs']
    for variant in schema['$defs']['ScheduledRequiredInteraction']['oneOf']:
        assert 'id' not in variant['properties'] and 'kind' not in variant['properties']
        assert {'response', 'resolution', 'link_previous'} <= set(variant['required'])
    following = schema['$defs']['ScheduledFollowingUnit']['oneOf'][0]['oneOf']
    assert all('id' not in v['properties'] and 'kind' in v['required'] for v in following)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'shot': {k: v for k, v in raw_shot().items() if k != 'shot_id'}}, schema)


@pytest.mark.parametrize('opening', [[], [passive()]])
def test_first_required_interaction_cannot_overlap_passive_opening(opening):
    from pipeline.llm_options import ark_output_schema
    raw = raw_shot()
    if opening:
        raw['action_plan']['opening'] = {'first': opening[0], 'following': opening[1:]}
    raw['action_plan']['interaction_1'].update(overlap=True, link_previous=bool(opening))
    before = deepcopy(raw)
    for schema in (schema_for(slot()), ark_output_schema(schema_for(slot()))):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}, schema)
    with pytest.raises(ValueError, match='搭接'):
        bind_authored_schedule(raw, slot())
    # 现有编译器对相同展开内容也拒绝；新绑定前移相同约束，不改动作。
    legacy = deepcopy(raw)
    legacy['action_plan'] = [dict(id=f'p_{i}', **unit) for i, unit in enumerate(opening)] + [
        {'id': 'i_1', 'kind': 'interaction', **raw['action_plan']['interaction_1']},
        {'id': 'i_2', 'kind': 'interaction', **raw['action_plan']['interaction_2']}]
    with pytest.raises(ValueError, match='搭接'):
        compile_action_units(AuthoredShot.model_validate(legacy), slot())
    assert raw == before


def test_first_interaction_can_link_to_opening_without_overlapping_it():
    from pipeline.llm_options import ark_output_schema
    raw = raw_shot()
    raw['action_plan']['opening'] = {'first': passive()}
    raw['action_plan']['interaction_1'].update(link_previous=True, overlap=False)
    assert_wire_valid(raw, slot())
    projected = ark_output_schema(schema_for(slot()))
    jsonschema.validate({'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}, projected)
    result = bind_authored_schedule(raw, slot())
    beats = compile_action_units(AuthoredShot.model_validate(result), slot())
    assert result['action_plan'][1]['link_previous'] is True
    assert beats[1]['start_seconds'] == beats[0]['end_seconds']
    assert 'flow_from_previous' in beats[1]
    raw['action_plan'].pop('opening')
    with pytest.raises(ValueError, match='前项'):
        bind_authored_schedule(raw, slot())


def test_following_and_later_required_interactions_keep_legal_overlap():
    from pipeline.llm_options import ark_output_schema
    raw = raw_shot()
    extra = {'kind': 'interaction', **interaction(), 'link_previous': True, 'overlap': True}
    raw['action_plan']['interaction_1']['following'] = [extra]
    raw['action_plan']['interaction_2'].update(link_previous=True, overlap=True)
    schema = ark_output_schema(schema_for(slot()))
    jsonschema.validate({'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}, schema)
    result = bind_authored_schedule(raw, slot())
    beats = compile_action_units(AuthoredShot.model_validate(result), slot())
    assert [u.get('overlap', False) for u in result['action_plan']] == [False, True, True]
    assert beats[1]['start_seconds'] < beats[0]['end_seconds']
    assert beats[2]['start_seconds'] < beats[1]['end_seconds']
