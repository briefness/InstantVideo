"""创建合同不能把消费方必需事实当作可省略的历史默认值。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from pipeline.action_authoring import authored_shot_response_schema
from pipeline.authoring_context import SceneLocation, contextual_shot_schema, bind_authored_context
from pipeline.authoring_schedule import scheduled_shot_schema, bind_authored_schedule
from pipeline.llm_options import ark_output_schema
from pipeline.storyboard_authoring import StoryIdentity, _bind_scene_entities
from test_creation_schedule_flow import scenario_story
from authoring_cases import scheduled_shot_fixture, contextual_shot_fixture
from test_scheduled_production_flow import scheduled_services


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def capture():
    record = json.loads((Path(__file__).parent / 'fixtures/real_identity_omission_failure.json').read_text())
    for item in record['sources'].values():
        assert hashlib.sha256(item['text'].encode()).hexdigest() == item['sha256']
    return {name: json.loads(item['text']) for name, item in record['sources'].items()}


def test_real_identity_response_is_not_repaired_and_schema_matches_local_contract():
    raw = capture()['planning_identity.json']
    before = deepcopy(raw)
    assert set(raw['story_arc']) == {'resolution'}
    for schema in (StoryIdentity.model_json_schema(), ark_output_schema(StoryIdentity.model_json_schema())):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(raw, schema)
    with pytest.raises(ValueError, match='story_arc.goal'):
        StoryIdentity.model_validate(raw)
    assert raw == before


@pytest.mark.parametrize('field', ['goal', 'stakes', 'turning_point', 'resolution'])
@pytest.mark.parametrize('value', [None, '', ' \n\t '])
def test_story_facts_are_required_nonblank_without_changing_stored_story_model(field, value):
    from pipeline.models import StoryArcSpec
    _, _, raw, _ = scenario_story('货物交接')
    identity = {k: v for k, v in raw.items() if k != 'shots'}
    if value is None:
        identity['story_arc'].pop(field)
    else:
        identity['story_arc'][field] = value
    with pytest.raises(ValueError, match='story_arc.' + field):
        StoryIdentity.model_validate(identity)
    # 历史模型保留原有空默认值，不能把读取旧存档变成新创作任务。
    assert StoryArcSpec.model_validate({}).model_dump() == dict.fromkeys(
        ['goal', 'stakes', 'turning_point', 'resolution'], '')


def case(topic):
    brief, intent, raw, plan = scenario_story(topic)
    names = [c['name'] for c in raw['characters']]
    locations = [SceneLocation.model_validate(s) for s in raw['locations']]
    catalog = raw['scene_entities']
    base = contextual_shot_schema(authored_shot_response_schema(names, require_outcomes=True,
        scene_entities=catalog, bound_handoffs=True), locations, catalog)
    wires = [scheduled_shot_fixture(contextual_shot_fixture(s), slot)
             for s, slot in zip(raw['shots'], plan['slots'])]
    for shot in wires:
        shot.pop('shot_id', None)
    return brief, intent, raw, plan, names, locations, catalog, base, wires


@pytest.mark.parametrize('boundary', ['start_state', 'end_state'])
@pytest.mark.parametrize('field', ['location', 'subject', 'action_phase', 'camera'])
def test_shot_schema_declares_existing_required_continuity_fields(boundary, field):
    _, _, _, plan, _, _, _, base, wires = case('绳索救援')
    shot = deepcopy(wires[0])
    shot[boundary].pop(field)
    schema = scheduled_shot_schema(base, plan['slots'][0])
    for wire in (schema, ark_output_schema(schema)):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({'shot': shot}, wire)


def test_formal_sdk_rejects_original_identity_before_any_shot(monkeypatch):
    from test_authoring_transport import http_client
    from pipeline.storyboard import generate_storyboard
    data = capture()
    intent = json.loads(data['planning_intent_response.json']['content'])
    sdk, calls = http_client(monkeypatch, [intent, data['planning_identity.json']])
    request = '制作约15秒的紫禁之巅决战，分三段独立视频再拼接。两大高手在紫禁城屋脊决战，武侠电影风格和电影级质感。轻功入场要帅；武打动作要多、节奏快，有借势、变招和连续的攻防流动，不要机械地一来一回或每招停顿复位。最后一方被另一方帅气击杀，以克制、非血腥的画面清楚呈现死亡结果。重点保持三段人物身份、服饰和兵器一致，前后动作与空间连续，运镜服务于动作可读性。项目自行规划脚本、分镜与时间分配。'
    with sdk, pytest.raises(ValueError, match='story_arc.goal'):
        generate_storyboard(request, target_duration=15)
    assert len(calls) == 2
    schema = calls[1]['response_format']['json_schema']['schema']
    ref = schema['properties']['story_arc']['$ref'].split('/')[-1]
    assert set(schema['$defs'][ref]['required']) == {'goal', 'stakes', 'turning_point', 'resolution'}


def paths(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield prefix + (key,)
            yield from paths(child, prefix + (key,))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from paths(child, prefix + (i,))


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_schema_valid_omissions_survive_whole_film_compilation(topic):
    from pipeline.action_authoring import compile_authored_storyboard
    from pipeline.storyboard import _compile_storyboard_contract, _validate_storyboard_richness
    from pipeline.film_rhythm import planned_interaction_seconds
    brief, _, raw, plan, names, locations, catalog, base, wires = case(topic)
    schema = scheduled_shot_schema(base, plan['slots'][0])
    checked = 0
    for path in paths(wires[0]):
        variant = deepcopy(wires[0])
        parent = variant
        for part in path[:-1]:
            parent = parent[part]
        del parent[path[-1]]
        if not jsonschema.Draft202012Validator(schema).is_valid({'shot': variant}):
            continue
        before = deepcopy(variant)
        shot = bind_authored_context(bind_authored_schedule({'shot_id': 1, **variant}, plan['slots'][0]), locations, catalog)
        _bind_scene_entities(shot, names, catalog)
        story = deepcopy(raw)
        story['shots'][0] = shot
        compiled = _compile_storyboard_contract(compile_authored_storyboard(story, plan),
            '16:9', '480p', 'cinematic', production_plan=plan)
        issues, critical = _validate_storyboard_richness(compiled, user_request=brief, require_narrative_contract=True)
        assert not critical, (path, issues)
        assert planned_interaction_seconds(compiled['shots']) >= 10.5 - 1e-6
        assert [(u['action'], u['result']) for u in shot['action_plan']] == [
            (b['action'], b['visible_result']) for b in compiled['shots'][0]['action_beats']]
        assert variant == before
        checked += 1
    assert checked >= 20


@pytest.mark.parametrize('boundary', ['start_state', 'end_state'])
@pytest.mark.parametrize('field', ['location', 'subject', 'action_phase', 'camera'])
@pytest.mark.parametrize('value', [None, '', ' \t\n'])
def test_provider_ignoring_required_states_stops_before_next_shot(monkeypatch, boundary, field, value):
    from test_authoring_transport import http_client
    from pipeline.storyboard import generate_storyboard
    brief, intent, raw, _, _, _, _, _, wires = case('货物交接')
    if value is None:
        wires[0][boundary].pop(field)
    else:
        wires[0][boundary][field] = value
    before = deepcopy(wires[0])
    values = [{k: v for k, v in intent.items() if k != 'source_request'},
              {k: v for k, v in raw.items() if k != 'shots'}, {'shot': wires[0]}]
    sdk, calls = http_client(monkeypatch, values)
    with sdk, pytest.raises(ValueError, match=boundary + '.' + field):
        generate_storyboard(brief, target_duration=15)
    assert len(calls) == 3 and wires[0] == before


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
async def test_minimal_creative_facts_reach_formal_preview_and_resume(scheduled_services, topic):
    import config
    from pipeline.orchestrator import VideoPipeline
    from pipeline.film_rhythm import planned_interaction_seconds
    from pipeline.provider_prompt import compile_generation_preview
    from test_director_intent import Responses
    brief, services = scheduled_services(topic)
    _, intent, raw, _ = scenario_story(topic)
    raw.pop('music_style', None)
    raw.pop('mood', None)
    for shot in raw['shots']:
        for boundary in ('start_state', 'end_state'):
            shot[boundary] = {k: shot[boundary][k] for k in ('location', 'subject', 'action_phase', 'camera')}
        for unit in shot['action_plan']:
            unit['weight'] = .01 if unit['kind'] == 'interaction' else 3.
    before = deepcopy(raw)
    services.author = Responses(intent, raw)
    run = VideoPipeline(plan_only=True, paid_take_budget=0, take_policy='single_pass')
    await run.run(brief)
    admitted = json.loads((run.workspace / 'result_admission.json').read_text())['execution_storyboard']
    preview_bytes = (run.workspace / 'generation_preview.json').read_bytes()
    assert json.loads(preview_bytes) == compile_generation_preview(admitted, model=config.SEEDANCE_MODEL)
    assert planned_interaction_seconds(admitted['shots']) >= 10.5 - 1e-6
    for source, shot in zip(raw['shots'], admitted['shots'], strict=True):
        assert [(u['action'], u['result']) for u in source['action_plan']] == [
            (b['action'], b['visible_result']) for b in shot['action_beats']]
    resumed = VideoPipeline.from_workspace(run.workspace)
    resumed.plan_only = True
    await resumed.run()
    assert (run.workspace / 'generation_preview.json').read_bytes() == preview_bytes
    assert raw == before and len(services.author.calls) == 5 and len(services.planning) == 1
    assert not services.submissions and not resumed.run_workspace.manifest.paid_take_budget.reservations


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
async def test_minimal_creation_reaches_sdk_and_export_with_independent_expectations(
        scheduled_services, monkeypatch, topic):
    """检查原稿事实与实际 SDK 参数，不用另一次提示词编译充当预期答案。"""
    from types import SimpleNamespace
    import config
    from pipeline.orchestrator import VideoPipeline
    from test_creation_schedule_flow import SCENARIOS
    from test_director_intent import Responses
    from tools.seedance_api import SeedanceAPI
    from tools.ffmpeg_ops import get_video_info

    brief, services = scheduled_services(topic)
    _, intent, raw, _ = scenario_story(topic)
    for shot in raw['shots']:
        for boundary in ('start_state', 'end_state'):
            shot[boundary] = {k: shot[boundary][k]
                              for k in ('location', 'subject', 'action_phase', 'camera')}
        for unit in shot['action_plan']:
            unit['weight'] = .01 if unit['kind'] == 'interaction' else 3.
    original = deepcopy(raw)
    services.author = Responses(intent, raw)
    polls = []

    def submit(**kwargs):
        services.submissions.append(deepcopy(kwargs))
        return SimpleNamespace(id=f'offline-{len(services.submissions)}')

    def poll(*, task_id):
        polls.append(task_id)
        return SimpleNamespace(status='succeeded', content=SimpleNamespace(video_url=task_id))

    api = object.__new__(SeedanceAPI)
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(
        tasks=SimpleNamespace(create=submit, get=poll)))
    api.download_video = services.download_video
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: api)

    run = VideoPipeline(plan_only=True, resolution='480p', platforms=['youtube'],
                        paid_take_budget=3, take_policy='single_pass')
    await run.run(brief)
    assert not services.submissions and not run.run_workspace.manifest.paid_take_budget.reservations
    preview_bytes = (run.workspace / 'generation_preview.json').read_bytes()
    preview = json.loads(preview_bytes)
    admitted = json.loads((run.workspace / 'result_admission.json').read_text())['execution_storyboard']
    resumed = VideoPipeline.from_workspace(run.workspace)
    output = await resumed.run()
    assert len(services.submissions) == len(services.reviews) == len(polls) == 3
    assert len(services.author.calls) == 5 and len(services.planning) == 1
    assert len(services.entries) == 2

    total_interaction = 0.
    for source, shot, row, request in zip(original['shots'], admitted['shots'],
                                        preview['shots'], services.submissions, strict=True):
        # 区间并集独立计算：不能把搭接的动作时长相加来凑全片占比。
        cursor, occupied = 0., 0.
        intervals = sorted((b['start_seconds'], b['end_seconds'])
                           for b in shot['action_beats'] if b['activity'] == 'interaction')
        for start, end in intervals:
            occupied += max(0., end - max(cursor, start))
            cursor = max(cursor, end)
        total_interaction += occupied
        assert row['allocated_interaction_seconds'] == pytest.approx(occupied, abs=.001)
        text = request['content'][0]['text']
        pacing = f'Allocate about {occupied:g}s of this 5s shot'
        assert pacing in text and pacing in row['profiles']['normal']['prompt']
        actions = [u for u in source['action_plan'] if u['kind'] == 'interaction']
        positions = [text.index(u['action']) for u in actions]
        assert positions == sorted(positions)
        for unit in actions:
            assert unit['result'] in text
        for field in ('primary_movement', 'end_framing'):
            assert source['camera'][field] in text
        for position in source['camera']['screen_positions'].values():
            assert position in text
        for character in original['characters']:
            assert character['name'] in text and character['description'] in text
        assert request['model'] == preview['model'] == config.SEEDANCE_MODEL
        assert request['duration'] == 5 and request['resolution'] == '480p'
        assert request['ratio'] == '16:9'
        references = request['content'][1:]
        if source['shot_id'] == 1:
            assert not references
        else:
            assert references[0]['role'] == 'first_frame'
            assert references[0]['image_url']['url'].startswith('data:image/')
    assert total_interaction >= 10.5 - 1e-6
    assert SCENARIOS[topic][-1][1] in services.submissions[-1]['content'][0]['text']
    video = next(s for s in get_video_info(output)['streams'] if s['codec_type'] == 'video')
    assert int(video['nb_frames']) == 360 and float(video['duration']) == pytest.approx(15, abs=1 / 24)
    assert len(resumed.run_workspace.manifest.paid_take_budget.reservations) == 3
    counts = [len(v) for v in (services.author.calls, services.planning, services.submissions,
                              services.reviews, services.entries, polls)]
    assert await VideoPipeline.from_workspace(run.workspace).run() == output
    assert counts == [len(v) for v in (services.author.calls, services.planning, services.submissions,
                                      services.reviews, services.entries, polls)]
    assert raw == original and (run.workspace / 'generation_preview.json').read_bytes() == preview_bytes
