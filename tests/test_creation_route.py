"""业务要求不选择内部编译协议；真实失败不改写，服务替身不证明模型语义精度。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.director_intent import director_intent_response_schema, resolve_director_intent
from pipeline.production_plan import build_production_plan
from test_director_intent import Responses, intent_reply


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def capture():
    record = json.loads((Path(__file__).parent / 'fixtures/real_creation_route_failure.json').read_text())
    for source in record['sources'].values():
        assert hashlib.sha256(source['text'].encode()).hexdigest() == source['sha256']
    response = json.loads(record['sources']['planning_intent_response.json']['text'])
    return record, json.loads(response['content'])


def test_original_route_failure_does_not_become_approval_of_the_old_draft():
    from pipeline.models import DirectorIntentSpec
    from pipeline.storyboard import _compile_storyboard_contract, _validate_storyboard_richness
    record, raw = capture()
    assert raw['choreography'] == 'exact'
    old = DirectorIntentSpec.model_validate({**raw, 'source_request': record['request'],
        'style_input': record['style']}).model_dump()
    plan = build_production_plan(old['content_focus'], 24, intent=old)
    assert not any(s.get('authoring_protocol') for s in plan['slots'])
    draft = json.loads(record['sources']['planning_draft_1.json']['text'])
    before = deepcopy(draft)
    compiled = _compile_storyboard_contract(deepcopy(draft), '16:9', '480p', record['style'], production_plan=plan)
    issues, critical = _validate_storyboard_richness(compiled, record['request'])
    assert critical
    assert any('empty_chair_table' in issue for issue in issues)
    assert any('多个主动动作执行者' in issue for issue in issues)
    assert draft == before


def test_real_intent_cannot_route_new_text_creation_to_legacy_single_actor_protocol():
    record, raw = capture()
    client, diagnostics = Responses(raw), {}
    resolved = resolve_director_intent(client, record['request'], style_input=record['style'],
        save_diagnostic=lambda n, v: diagnostics.update({n: v}))
    plan = build_production_plan(resolved['content_focus'], 24, intent=resolved)
    assert all(s.get('authoring_protocol') == 'scene-units-v1' for s in plan['slots'])
    assert resolved['completion_goals'] == raw['completion_goals']
    assert resolved['source_request'] == record['request']
    assert len(plan['slots']) == 4 and sum(s['duration'] for s in plan['slots']) == 24
    assert json.loads(diagnostics['planning_intent_response.json']['content']) == raw
    assert len(client.calls) == 1


def test_fresh_wire_does_not_ask_the_model_to_choose_an_internal_engine():
    schema = director_intent_response_schema('Keep the events fixed; design the performance.')
    for variant in schema.get('oneOf', [schema]):
        assert 'choreography' not in variant['properties']
        assert 'authoring_protocol' not in variant['properties']
        assert 'completion_goals' in variant['required']
        assert variant['additionalProperties'] is False


@pytest.mark.parametrize('reported', ['exact', 'flexible', None])
@pytest.mark.parametrize('brief', [
    'Keep this order: the courier delivers the parcel, the receiver signs, both leave. Design their movements.',
    'A dancer turns clockwise exactly twice, then holds both arms above the head for two seconds.',
    '展示拆装过程：先按住锁扣，再拔出接头。必须用右手，操作顺序不得改变。',
    '只展示灯罩透光材质，不出现人物或额外故事。',
])
def test_user_methods_are_goals_not_protocol_selection(brief, reported):
    raw = intent_reply(brief, 'balanced', 'balanced')
    if reported is not None:
        raw['choreography'] = reported
    resolved = resolve_director_intent(Responses(raw), brief)
    assert resolved['authoring_protocol'] == 'scene-units-v1'
    assert resolved['completion_goals'][0] == {'goal': brief, 'basis': 'user', 'evidence': brief}
    assert resolved['source_request'] == brief


def test_reference_execution_keeps_its_explicit_capability_boundary():
    brief = '用骨架参考视频复现动作路径。'
    raw = {**intent_reply(brief), 'motion_control': 'reference_motion',
           'motion_control_evidence': brief, 'choreography': 'flexible'}
    resolved = resolve_director_intent(Responses(raw), brief)
    plan = build_production_plan(resolved['content_focus'], 15, intent=resolved)
    assert not resolved.get('authoring_protocol')
    assert all(s['choreography'] == 'exact' for s in plan['slots'])
    assert resolved['completion_goals'] == raw['completion_goals']


def test_explicit_reference_setting_selects_capability_without_rewriting_request_interpretation():
    brief = 'A performer demonstrates a movement.'
    raw = intent_reply(brief, 'balanced', 'balanced')
    resolved = resolve_director_intent(Responses(raw), brief, motion_control='reference_motion')
    assert resolved['motion_control'] == 'text_guided' and resolved['motion_control_evidence'] == ''
    assert resolved['choreography'] == 'exact' and not resolved.get('authoring_protocol')
    assert resolved['source_request'] == brief and resolved['completion_goals'] == raw['completion_goals']


def test_real_raw_response_reaches_structured_authoring_at_actual_sdk_boundary(monkeypatch):
    from test_authoring_transport import http_client
    from pipeline.storyboard import generate_storyboard
    record, raw = capture()
    sdk, calls = http_client(monkeypatch, [raw, {}])
    # 空创作响应必须停止；这里只验证原意图的下一步，不替原坏稿补分镜。
    with sdk, pytest.raises(ValueError):
        generate_storyboard(record['request'], style=record['style'], resolution='480p', allow_repair=False)
    assert len(calls) == 2
    assert calls[1]['response_format']['type'] == 'json_schema'
    assert 'scene_entities' in calls[1]['response_format']['json_schema']['schema']['properties']


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', ['family', 'ensemble', 'product'])
async def test_formal_export_does_not_depend_on_model_engine_label(topic, tmp_path, monkeypatch):
    import test_scene_delivery_flow as flow
    original = flow.scene_case
    def reported_exact(value):
        brief, reply, identity, wires, shots = original(value)
        return brief, {**reply, 'choreography': 'exact'}, identity, wires, shots
    monkeypatch.setattr(flow, 'scene_case', reported_exact)
    # 复用完整导出、预算、恢复断言；只改变旧分类标签这一个变量。
    await flow.test_scene_creation_reaches_final_media_and_reuses_completed_delivery(
        tmp_path, monkeypatch, topic, active_performance=False)
