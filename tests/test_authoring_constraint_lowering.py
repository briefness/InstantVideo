"""真实失败保持拒绝；请求语言、解析和编译在离线边界上对齐。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from pipeline.authoring_contract import AuthoringContract, AuthoringContractError
from test_goal_allocation import offline

FIXTURE = Path(__file__).parent / 'fixtures/real_authoring_domain_failure.json'


def captured():
    data = json.loads(FIXTURE.read_text())
    return (json.loads(data['contract_text']), {'response_schema': data['wire_schema']},
            json.loads(json.loads(data['response_text'])['content']))


def test_capture_preserves_original_bytes():
    import hashlib
    data = json.loads(FIXTURE.read_text())
    for key, name in [('contract_text', 'planning_shot_002_contract.json'),
                      ('response_text', 'planning_shot_002_response.json')]:
        assert hashlib.sha256(data[key].encode()).hexdigest() == data['sha256'][name]


def test_original_failure_is_allowed_by_wire_but_only_zero_weight_blocks_matching_branch():
    contract, request, response = captured()
    assert Draft202012Validator(request['response_schema']).is_valid(response)
    before = deepcopy(response)
    with pytest.raises(AuthoringContractError) as error:
        AuthoringContract(contract['schema']).validate(response)
    assert error.value.issues == [
        {'path': '/shot/action_plan/opening/first/weight', 'constraint': 'exclusiveMinimum'}]
    assert response == before


def test_new_authoring_contract_makes_real_invalid_number_unrepresentable_on_wire():
    from pipeline.action_authoring import authored_shot_response_schema
    from pipeline.authoring_schedule import scheduled_shot_schema
    from pipeline.authoring_timing import execution_timing_schema, VERSION
    from test_authoring_schedule import slot
    _, _, response = captured()
    schema = authored_shot_response_schema(['QING_FENG', 'XUE_YIN'], require_outcomes=True,
        scene_entities=['ROOF'], bound_handoffs=True)
    schema = execution_timing_schema(scheduled_shot_schema(schema, slot(), explicit_absence=True), version=VERSION)
    current = AuthoringContract(schema, generation_constraints=True)
    # 单独核对当前类型定义的控制域，避免其他上下文字段造成假阳性。
    weight_schema = current.wire_schema['$defs']['ScheduledFirstPassiveUnit']['properties']['weight']
    weight_contract = {'$defs': current.wire_schema['$defs'], **weight_schema}
    assert Draft202012Validator(weight_contract).is_valid(1.5)
    assert not Draft202012Validator(weight_contract).is_valid(0)
    assert not Draft202012Validator(current.wire_schema).is_valid(response)
    with pytest.raises(AuthoringContractError):
        current.validate(response)


def test_continuous_values_and_historical_contracts_are_not_quantized():
    schema = {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 100}
    contract = AuthoringContract(schema, generation_constraints=True)
    assert contract.validate(1.234567) is None
    assert '$ref' not in contract.wire_schema
    old, _, _ = captured()
    assert AuthoringContract(old['schema']).record() == old


@pytest.mark.parametrize('schema', [
    {'type': 'number', 'minimum': 0, 'maximum': 1, 'x-authoring-step': '0'},
    {'type': 'number', 'minimum': 0, 'maximum': 1, 'x-authoring-step': 'invalid'},
    {'type': 'number', 'minimum': 0, 'maximum': 1, 'x-authoring-step': 'NaN'},
    {'type': 'number', 'minimum': 0, 'x-authoring-step': '0.1'},
    {'type': 'integer', 'minimum': 0, 'maximum': 1000000},
    {'type': 'integer', 'exclusiveMinimum': 0, 'exclusiveMaximum': 1},
])
def test_invalid_or_excessive_domains_stop_before_a_request(schema):
    before = deepcopy(schema)
    with pytest.raises(ValueError):
        AuthoringContract(schema, generation_constraints=True)
    assert schema == before


def test_discrete_domains_are_shared_and_do_not_modify_inputs():
    from pipeline.authoring_controls import bounded_authoring_schema
    schema = {'type': 'object', 'properties': {name: {'type': 'number',
        'exclusiveMinimum': 0, 'maximum': 2, 'x-authoring-step': '0.1'}
        for name in ('relative_budget', 'control_gain')}}
    before = deepcopy(schema)
    lowered = bounded_authoring_schema(schema)
    assert schema == before and len(lowered['$defs']) == 1
    assert bounded_authoring_schema(lowered) == lowered
    contract = AuthoringContract(schema, generation_constraints=True)
    for number in (-1, 0, .1, .3, 1.5, 1.234, 2, 2.1, True, '1'):
        value = {'relative_budget': number, 'control_gain': number}
        wire = Draft202012Validator(contract.wire_schema).is_valid(value)
        assert wire == Draft202012Validator(contract.schema).is_valid(value)
    # 类型约束以外的自由文本仍然需要本地拒绝；不将描述声称为强制语法。
    text_contract = AuthoringContract({'type': 'string', 'minLength': 1}, generation_constraints=True)
    assert Draft202012Validator(text_contract.wire_schema).is_valid('')
    with pytest.raises(AuthoringContractError):
        text_contract.validate('')


def test_fractional_multiples_cannot_make_wire_wider_than_local_validation():
    contract = AuthoringContract({'type': 'number', 'minimum': 0, 'maximum': 2,
        'multipleOf': .1, 'x-authoring-step': '0.1'}, generation_constraints=True)
    for number in (0, .1, .3, .6, 1.2, 1.7, 2, 2.1):
        assert Draft202012Validator(contract.wire_schema).is_valid(number) == (
            Draft202012Validator(contract.schema).is_valid(number))
    assert AuthoringContract(contract.schema, generation_constraints=True).record() == contract.record()


@pytest.mark.parametrize('name', ['shot_durations', 'sample_count', 'capacity'])
def test_integer_domains_lower_exactly_not_by_field_name(name):
    schema = {'type': 'object', 'properties': {name: {
        'type': 'integer', 'minimum': 4, 'exclusiveMaximum': 8}},
        'required': [name], 'additionalProperties': False}
    contract = AuthoringContract(schema, generation_constraints=True)
    for value in (-1, 0, 3, 4, 5, 7, 8, 12, 4.5, True, None, '4'):
        instance = {name: value}
        assert Draft202012Validator(contract.wire_schema).is_valid(instance) == (
            Draft202012Validator(schema).is_valid(instance))


def test_union_diagnostics_do_not_claim_unselected_variants_are_obligations():
    schema = {'oneOf': [
        {'type': 'object', 'properties': {'kind': {'const': 'move'}, 'amount': {
            'type': 'integer', 'minimum': 1}}, 'required': ['kind', 'amount']},
        {'type': 'object', 'properties': {'kind': {'const': 'wait'}, 'duration': {
            'type': 'integer', 'minimum': 1}}, 'required': ['kind', 'duration']}]}
    with pytest.raises(AuthoringContractError) as error:
        AuthoringContract(schema).validate({'kind': 'wait', 'duration': 0})
    assert error.value.issues == [{'path': '/duration', 'constraint': 'minimum'}]
    with pytest.raises(AuthoringContractError):
        AuthoringContract(schema).validate({'kind': 'invalid', 'duration': 2})


def test_ambiguous_union_does_not_invent_a_selected_branch():
    contract = AuthoringContract({'oneOf': [{'const': 'left'}, {'const': 'right'}]})
    with pytest.raises(AuthoringContractError) as error:
        contract.validate('neither')
    assert error.value.issues == [{'path': '/', 'constraint': 'oneOf'}]
    with pytest.raises(AuthoringContractError):
        AuthoringContract({'oneOf': [{'type': 'number'}, {'type': 'integer'}]}).validate(1)


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_actual_sdk_request_domain_and_compilation_agree_across_scenes(monkeypatch, topic):
    import httpx
    from openai import OpenAI
    from pipeline.storyboard import generate_storyboard
    from structured_transport_cases import completion_response
    from test_creation_schedule_flow import scenario_story
    from test_director_intent import Responses
    brief, intent, story, _ = scenario_story(topic)
    author = Responses(intent, story)
    calls = []
    saved = {}
    def receive(request):
        body = json.loads(request.content)
        calls.append(body)
        response = json.loads(author.create(**body).choices[0].message.content)
        # 上游意图使用已有的历史协议夹具；此处差分核验新版创作边界。
        if body['response_format']['json_schema']['name'] != 'director_intent':
            Draft202012Validator(body['response_format']['json_schema']['schema']).validate(response)
        return completion_response(response, body)
    with OpenAI(api_key='offline-key', base_url='https://offline.invalid/v1', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(receive))) as sdk:
        monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: sdk)
        result = generate_storyboard(brief, target_duration=15,
            save_diagnostic=lambda name, value: saved.update({name: deepcopy(value)}))
    assert len(result['shots']) == 3 and len(calls) == 5
    for index, call in enumerate(calls[1:]):
        key = 'identity' if index == 0 else f'shot_{index:03d}'
        record = saved[f'planning_{key}_contract.json']
        assert record['version'] == 'authoring-contract-v2'
        from pipeline.llm_options import ark_output_schema
        assert call['response_format']['json_schema']['schema'] == ark_output_schema(record['schema'])
        assert Draft202012Validator(record['schema']).is_valid(saved[f'planning_{key}.json'])
