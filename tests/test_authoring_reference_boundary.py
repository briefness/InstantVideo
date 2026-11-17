"""创作/执行实体边界的差分回放；原始失败不修稿，不访问外部服务。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from pipeline.action_authoring import authored_shot_response_schema
from pipeline.authoring_context import SceneLocation, contextual_shot_schema
from pipeline.authoring_schedule import scheduled_shot_schema
from pipeline.llm_options import ark_output_schema
from pipeline.storyboard_authoring import _bind_scene_entities
from test_scheduled_production_flow import scheduled_services


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def captured():
    record = json.loads((Path(__file__).parent / 'fixtures/real_spatial_reference_failure.json').read_text())
    for item in record['sources'].values():
        assert hashlib.sha256(item['text'].encode()).hexdigest() == item['sha256']
    return {name: json.loads(item['text']) for name, item in record['sources'].items()}


def test_actual_capture_is_rejected_by_wire_and_local_boundary_without_rewrite():
    data = captured()
    identity = data['planning_identity.json']
    names = [c['name'] for c in identity['characters']]
    catalog = identity['scene_entities']
    base = contextual_shot_schema(authored_shot_response_schema(names, require_outcomes=True,
        scene_entities=catalog, bound_handoffs=True),
        [SceneLocation.model_validate(s) for s in identity['locations']], catalog)
    before = deepcopy(data)
    for i, (authored, compiled) in enumerate(zip(data['planning_authoring.json']['shots'],
                                               data['planning_compiled.json']['shots']), 1):
        response = json.loads(data[f'planning_shot_{i:03d}_response.json']['content'])
        wire = scheduled_shot_schema(base, compiled['production_slot'])
        for schema in (wire, ark_output_schema(wire)):
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(response, schema)
        with pytest.raises(ValueError, match='camera.screen_positions.*未注册'):
            _bind_scene_entities(authored, names, catalog)
    assert data == before


def test_actual_capture_stops_formal_flow_before_requesting_next_shot(monkeypatch):
    from test_authoring_transport import http_client
    from pipeline.storyboard import generate_storyboard
    from pipeline.authoring_contract import AuthoringContractError
    data = captured()
    before = deepcopy(data)
    request = data['planning_compiled.json']['production_plan']['director_intent']['source_request']
    values = [json.loads(data['planning_intent_response.json']['content']), data['planning_identity.json'],
              json.loads(data['planning_shot_001_response.json']['content'])]
    sdk, calls = http_client(monkeypatch, values, annotate_legacy_intent=True)
    with sdk, pytest.raises(AuthoringContractError) as caught:
        generate_storyboard(request, target_duration=15)
    assert {'path': '/shot/camera/screen_positions', 'constraint': 'additionalProperties'} in caught.value.issues
    assert {'path': '/shot/action_plan/interaction_1', 'constraint': 'additionalProperties'} in caught.value.issues
    assert len(calls) == 3 and data == before


@pytest.mark.parametrize('field,value', [
    ('position', 'work_area'), ('blocking', 'work_area'), ('blocking', 'Door'),
    ('actor', 'Door'), ('target', 'unknown'), ('result', 'unknown'),
])
def test_all_reference_kinds_share_registry_and_fail_atomically(field, value):
    from test_action_results import annotated_story
    shot = deepcopy(annotated_story()['shots'][0])
    if field == 'position':
        shot['camera']['screen_positions'][value] = 'center'
    elif field == 'blocking':
        shot['blocking'][value] = deepcopy(next(iter(shot['blocking'].values())))
    elif field in ('actor', 'target'):
        shot['action_plan'][1][field] = value
    else:
        shot['action_plan'][1]['resolution']['scene_changes'][0]['entity'] = value
    before = deepcopy(shot)
    with pytest.raises(ValueError, match='未注册'):
        _bind_scene_entities(shot, ['attacker', 'defender'], ['Door', 'Dust'])
    assert shot == before


@pytest.mark.parametrize('object_id', ['Parcel', 'Door', 'Rescue_Rope', 'properties', 'A/B~C'])
def test_registered_staging_object_gets_same_scope_as_action_object(object_id):
    from test_action_results import annotated_story
    from pipeline.action_authoring import compile_authored_storyboard
    from test_action_authoring import plan_for
    from pipeline.participants import structured_entity_reference_issues
    raw = annotated_story()
    for shot in raw['shots']:
        shot['camera']['screen_positions'][object_id] = 'lower center beside the working hand'
        shot['blocking']['attacker']['eyeline_target'] = 'toward the opening in the distance'
        _bind_scene_entities(shot, ['attacker', 'defender'], ['Door', 'Dust', object_id])
        assert object_id in shot['key_props']
    compiled = compile_authored_storyboard(raw, plan_for())
    for shot in compiled['shots']:
        assert not structured_entity_reference_issues(shot, ['attacker', 'defender'])
        assert shot['camera']['screen_positions'][object_id] == 'lower center beside the working hand'
        assert shot['blocking']['attacker']['eyeline_target'] == 'toward the opening in the distance'


def test_dynamic_maps_distinguish_character_owners_objects_and_free_prose():
    from test_authoring_context import response, context
    raw = response()
    locations, catalog = context()
    schema = contextual_shot_schema(authored_shot_response_schema(['attacker', 'defender'],
        require_outcomes=True, scene_entities=catalog, bound_handoffs=True), locations, catalog)
    for wire in (schema, ark_output_schema(schema)):
        jsonschema.validate({'shot': raw}, wire)
        for field, entity in [('screen_positions', 'unknown'), ('blocking', 'unknown'), ('blocking', 'Door')]:
            bad = deepcopy(raw)
            container = bad['camera'][field] if field == 'screen_positions' else bad[field]
            container[entity] = 'center' if field == 'screen_positions' else {}
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate({'shot': bad}, wire)
        valid = deepcopy(raw)
        valid['camera']['screen_positions']['Door'] = 'far right'
        valid['blocking']['attacker']['facing_target'] = 'toward the bright horizon'
        jsonschema.validate({'shot': valid}, wire)


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
async def test_spatial_only_entity_survives_formal_planning_preview_and_resume(scheduled_services, topic):
    import config
    from pipeline.orchestrator import VideoPipeline
    from pipeline.film_rhythm import planned_interaction_seconds
    from pipeline.provider_prompt import compile_generation_preview
    from test_creation_schedule_flow import scenario_story
    from test_director_intent import Responses
    brief, services = scheduled_services(topic)
    _, intent, raw, _ = scenario_story(topic)
    raw['scene_entities'].append('Spatial_Anchor')
    for shot in raw['shots']:
        shot['camera']['screen_positions']['Spatial_Anchor'] = 'lower right beside the path'
        assert 'Spatial_Anchor' not in shot['key_props']
        for unit in shot['action_plan']:
            unit['weight'] = .01 if unit['kind'] == 'interaction' else 3.
    before = deepcopy(raw)
    services.author = Responses(intent, raw)
    run = VideoPipeline(plan_only=True, paid_take_budget=0, take_policy='single_pass')
    await run.run(brief)
    frozen = json.loads((run.workspace / 'storyboard.json').read_text())
    preview_bytes = (run.workspace / 'generation_preview.json').read_bytes()
    preview = json.loads(preview_bytes)
    # 请求使用独立审查后的执行合同；storyboard.json 保留审查前原稿用于复核。
    admitted = json.loads((run.workspace / 'result_admission.json').read_text())['execution_storyboard']
    assert preview == compile_generation_preview(admitted, model=config.SEEDANCE_MODEL)
    assert planned_interaction_seconds(frozen['shots']) >= 10.5 - 1e-6
    for source, shot, row in zip(raw['shots'], frozen['shots'], preview['shots'], strict=True):
        assert 'Spatial_Anchor' in shot['key_props']
        assert shot['camera']['screen_positions']['Spatial_Anchor'] == 'lower right beside the path'
        assert 'lower right beside the path' in row['profiles']['normal']['prompt']
        assert all(not p['preflight_issues'] for p in row['profiles'].values())
        assert [(u['action'], u['result']) for u in source['action_plan']] == [
            (b['action'], b['visible_result']) for b in shot['action_beats']]
    assert len(services.author.calls) == 5 and len(services.planning) == 1
    assert not services.submissions and raw == before
    resumed = VideoPipeline.from_workspace(run.workspace)
    resumed.plan_only = True
    await resumed.run()
    assert len(services.author.calls) == 5 and len(services.planning) == 1
    assert (run.workspace / 'generation_preview.json').read_bytes() == preview_bytes
    assert not resumed.run_workspace.manifest.paid_take_budget.reservations
