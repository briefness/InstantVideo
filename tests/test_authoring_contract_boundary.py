"""完整创作契约与服务端投影的差分测试，不修写真实失败内容。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace as NS

import jsonschema
import pytest
from test_scheduled_production_flow import scheduled_services


def test_explicit_absence_consumes_no_capacity_even_when_all_units_are_required():
    from pipeline.action_authoring import AuthoredShot, authored_shot_response_schema
    from pipeline.authoring_schedule import scheduled_shot_schema, bind_authored_schedule
    from pipeline.authoring_contract import AuthoringContract
    from test_authoring_schedule import slot, raw_shot, interaction
    limit = next(v.max_length for v in AuthoredShot.model_fields['action_plan'].metadata
                 if hasattr(v, 'max_length'))
    current = slot(count=limit)
    base = authored_shot_response_schema(['attacker', 'defender'], require_outcomes=True,
        scene_entities=['Door', 'Dust'], bound_handoffs=True)
    raw = raw_shot({**{f'interaction_{i+1}': interaction() for i in range(limit)}, 'opening': None})
    contract = AuthoringContract(scheduled_shot_schema(base, current, explicit_absence=True))
    contract.validate({'shot': {k:v for k,v in raw.items() if k!='shot_id'}})
    assert len(bind_authored_schedule(raw, current, explicit_absence=True)['action_plan']) == limit


@pytest.mark.parametrize('field,value', [('action', ''), ('action', ' \n\t'), ('result', ''),
    ('weight', 0), ('weight', -1), ('weight', 101), ('weight', '1'), ('weight', True), ('actor', 'unregistered')])
@pytest.mark.parametrize('position', ['opening', 'following', 'result_hold'])
def test_all_action_positions_share_full_contract_without_repair(field, value, position):
    from pipeline.action_authoring import authored_shot_response_schema
    from pipeline.authoring_schedule import scheduled_shot_schema
    from pipeline.authoring_contract import AuthoringContract, AuthoringContractError
    from test_authoring_schedule import slot, raw_shot, passive
    base = authored_shot_response_schema(['attacker', 'defender'], require_outcomes=True,
        scene_entities=['Door', 'Dust'], bound_handoffs=True)
    current = slot(hold=.5)
    raw = raw_shot()
    raw['action_plan']['result_hold'] = {k:v for k,v in passive().items() if k!='kind'}
    bad = {**passive(), field: value}
    if position == 'opening':
        raw['action_plan']['opening'] = {'first': bad}
    elif position == 'following':
        raw['action_plan']['interaction_1']['following'] = [bad]
    else:
        raw['action_plan']['result_hold'] = {k:v for k,v in bad.items() if k!='kind'}
    snapshot = deepcopy(raw)
    contract = AuthoringContract(scheduled_shot_schema(base, current, explicit_absence=True))
    with pytest.raises(AuthoringContractError) as rejected:
        contract.validate({'shot':{k:v for k,v in raw.items() if k!='shot_id'}})
    assert any(v['path'].endswith('/'+field) for v in rejected.value.issues)
    assert raw == snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
async def test_explicit_absence_survives_formal_export_and_resume(scheduled_services, topic):
    from pipeline.orchestrator import VideoPipeline
    from tools.ffmpeg_ops import get_video_info
    brief, services = scheduled_services(topic)
    original = services.author.create
    null_count = []
    def synthetic_response(**kwargs):
        response = original(**kwargs)
        content = json.loads(response.choices[0].message.content)
        plan = content.get('shot', {}).get('action_plan')
        # 仅给独立测试替身标注“无准备段”；真实坏响应从不改写。
        if isinstance(plan, dict) and 'opening' not in plan and 'interaction_1' in plan:
            plan['opening'] = None
            null_count.append(len(services.author.calls))
            response.choices[0].message.content = json.dumps(content)
        return response
    services.author.create = synthetic_response
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], paid_take_budget=3, take_policy='single_pass')
    output = await pipeline.run(brief)
    assert null_count and len(services.author.calls) == 5 and len(services.submissions) == 3
    assert [len(r['action_events']) for r in services.reviews] == [2,3,1]
    video = next(v for v in get_video_info(output)['streams'] if v['codec_type']=='video')
    assert int(video['nb_frames']) == 360
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert len(services.author.calls) == 5 and len(services.submissions) == 3


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *a: pytest.fail('network forbidden'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *a: pytest.fail('network forbidden'))


def captured():
    value = json.loads((Path(__file__).parent / 'fixtures/real_empty_action_failure.json').read_text())
    for item in value['files'].values():
        assert hashlib.sha256(item['text'].encode()).hexdigest() == item['sha256']
    return value


def test_real_response_satisfies_old_wire_but_is_rejected_at_decode_not_compilation(monkeypatch, tmp_path):
    import config
    import asyncio
    from pipeline.orchestrator import VideoPipeline
    data = captured()
    response = json.loads(json.loads(data['files']['planning_shot_003_response.json']['text'])['content'])
    old_request = json.loads(data['files']['planning_shot_003_request.json']['text'])
    jsonschema.validate(response, old_request['response_schema'])
    names = ['planning_intent_response.json', 'planning_identity_response.json'] + [
        f'planning_shot_{i:03d}_response.json' for i in range(1, 4)]
    calls = []
    class Client:
        def __init__(self): self.chat = self.completions = self
        def create(self, **kwargs):
            calls.append(kwargs)
            assert len(calls) <= len(names)
            raw = json.loads(data['files'][names[len(calls)-1]]['text'])
            return NS(choices=[NS(message=NS(content=raw['content'], refusal=None), finish_reason=raw['finish_reason'])])
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: Client())
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(style=data['style_input'], paid_take_budget=0, planning_repair=False)
    with pytest.raises(ValueError, match='创作响应契约'):
        asyncio.run(pipeline.run(data['request']))
    assert len(calls) == 5
    record = json.loads((pipeline.workspace / 'planning_shot_003_contract_error.json').read_text())
    assert {'minLength', 'exclusiveMinimum'} <= {v['constraint'] for v in record['issues']}
    assert all('opening' in v['path'] for v in record['issues'] if v['constraint']=='minLength')
    steps = json.loads((pipeline.workspace/'planning_checkpoint.json').read_text())['steps']
    assert steps[-1]['state'] == 'failed' and steps[-1]['key'] == 'shot_003'
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations


@pytest.mark.parametrize('topic', ['dance', 'parcel', 'rescue', 'product'])
def test_explicit_absence_is_not_an_empty_action(topic):
    from pipeline.authoring_schedule import scheduled_shot_schema, bind_authored_schedule
    from test_authoring_schedule import schema_for, slot, raw_shot, passive
    from pipeline.action_authoring import authored_shot_response_schema
    base = authored_shot_response_schema(['attacker','defender'], require_outcomes=True,
        scene_entities=['Door','Dust'], bound_handoffs=True)
    schema = scheduled_shot_schema(base, slot(), explicit_absence=True)
    raw = raw_shot()
    raw['action_plan']['opening'] = None
    raw['action_plan']['interaction_1']['action'] = topic + ' transfers momentum'
    snapshot = deepcopy(raw)
    jsonschema.validate({'shot':{k:v for k,v in raw.items() if k!='shot_id'}}, schema)
    compiled = bind_authored_schedule(raw, slot(), explicit_absence=True)
    assert len(compiled['action_plan']) == 2 and raw == snapshot
    assert compiled['action_plan'][0]['action'] == snapshot['action_plan']['interaction_1']['action']
    raw['action_plan']['opening']={'first':{**passive(), 'action':'', 'result':'', 'weight':0}}
    with pytest.raises(ValueError):
        bind_authored_schedule(raw, slot(), explicit_absence=True)


def test_contract_keeps_unsupported_constraints_explicit():
    from pipeline.authoring_contract import AuthoringContract, AuthoringContractError
    local = {'type':'object','properties':{'text':{'type':'string','minLength':1},
             'weight':{'type':'number','exclusiveMinimum':0}},'required':['text','weight'], 'additionalProperties':False}
    contract=AuthoringContract(local)
    assert {v['constraint'] for v in contract.record()['local_constraints']} == {'minLength','exclusiveMinimum'}
    assert contract.wire_schema['properties']['text'] == {'type':'string'}
    with pytest.raises(AuthoringContractError): contract.validate({'text':'','weight':0})
    good={'text':'valid action','weight':1.25}
    assert contract.validate(good) is None


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_contract_rejects_non_json_numbers_without_coercion(value):
    from pipeline.authoring_contract import AuthoringContract, AuthoringContractError
    contract = AuthoringContract({'type':'number', 'exclusiveMinimum':0})
    with pytest.raises(AuthoringContractError):
        contract.validate(value)


def test_contract_never_resolves_external_schema(offline):
    from pipeline.authoring_contract import AuthoringContract
    with pytest.raises(ValueError, match='只允许本地引用'):
        AuthoringContract({'properties':{'action':{'$ref':'https://offline.invalid/schema'}}})


@pytest.mark.asyncio
async def test_real_capture_uses_actual_sdk_body_and_failed_resume_never_resubmits(tmp_path, monkeypatch):
    import config
    from pipeline.orchestrator import VideoPipeline
    from pipeline.llm_options import ark_output_schema
    from test_authoring_transport import http_client
    data = captured()
    names = ['planning_intent_response.json', 'planning_identity_response.json'] + [
        f'planning_shot_{i:03d}_response.json' for i in range(1, 4)]
    raw_contents = [json.loads(data['files'][name]['text'])['content'] for name in names]
    values = [json.loads(content) for content in raw_contents]
    sdk, calls = http_client(monkeypatch, values)
    sdk.max_retries = 0
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(style=data['style_input'], paid_take_budget=0, planning_repair=False)
    with sdk:
        with pytest.raises(ValueError, match='创作响应契约'):
            await pipeline.run(data['request'])
        before = {name:(pipeline.workspace/name).read_bytes() for name in (
            'planning_shot_003.json', 'planning_shot_003_response.json', 'planning_checkpoint.json')}
        with pytest.raises(ValueError, match='失败或请求结果不明'):
            await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert len(calls) == 5
    assert all((pipeline.workspace/name).read_bytes() == content for name,content in before.items())
    assert json.loads(before['planning_shot_003.json']) == values[-1]
    assert json.loads(json.loads(before['planning_shot_003_response.json'])['content']) == values[-1]
    record = json.loads((pipeline.workspace/'planning_shot_003_contract.json').read_text())
    request_schema = calls[-1]['response_format']['json_schema']['schema']
    assert request_schema == ark_output_schema(record['schema'])
    assert hashlib.sha256(json.dumps(request_schema, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == record['wire_sha256']
    assert json.loads((pipeline.workspace/'planning_check.json').read_text())['stage'] == 'response_contract'
    assert not (pipeline.workspace/'planning_shot_004_request.json').exists()
    assert not (pipeline.workspace/'storyboard.json').exists()
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations


@pytest.mark.parametrize('failed_file', ['planning_shot_003_contract_error.json', 'planning_check.json'])
def test_contract_diagnostic_failure_preserves_primary_error(monkeypatch, failed_file):
    from pipeline.authoring_contract import AuthoringContractError
    from pipeline.operation_failure import failure_evidence
    from pipeline.storyboard import generate_storyboard
    from test_authoring_transport import http_client
    data = captured()
    names = ['planning_intent_response.json', 'planning_identity_response.json'] + [
        f'planning_shot_{i:03d}_response.json' for i in range(1,4)]
    values = [json.loads(json.loads(data['files'][name]['text'])['content']) for name in names]
    sdk, calls = http_client(monkeypatch, values)
    saved = {}
    def save(name, value):
        if name == failed_file:
            raise OSError('offline disk failure')
        saved[name] = deepcopy(value)
    with sdk, pytest.raises(AuthoringContractError) as rejected:
        generate_storyboard(data['request'], target_duration=24, style=data['style_input'], save_diagnostic=save)
    assert len(calls) == 5
    assert saved['planning_shot_003.json'] == values[-1]
    evidence = failure_evidence(rejected.value)
    assert evidence['category'] == 'validation'
    assert evidence['secondary'] == [{'phase':'save_diagnostic', 'type':'OSError'}]
