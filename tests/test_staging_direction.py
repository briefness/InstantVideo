"""拍摄指导不能被验收用的少量标签替代；不以新增门槛保护指导。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.action_authoring import compile_authored_storyboard
from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
from pipeline.storyboard import _compile_storyboard_contract
from pipeline.semantic_review import _review_contract
from test_creation_schedule_flow import SCENARIOS, scenario_story
from test_scheduled_production_flow import scheduled_services


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def compile_story(raw, plan):
    return _compile_storyboard_contract(compile_authored_storyboard(raw, plan),
        '16:9', '480p', 'cinematic', production_plan=plan)


def test_real_guidance_survives_full_compilation_without_rewriting_source():
    record = json.loads((Path(__file__).parent / 'fixtures/real_staging_projection.json').read_text())
    assert hashlib.sha256(record['text'].encode()).hexdigest() == record['sha256']
    raw = json.loads(record['text'])
    original = deepcopy(raw)
    compiled = compile_story(raw, record['plan'])
    for source, shot in zip(raw['shots'], compiled['shots']):
        prompt = compile_normal_provider_prompt(shot, compiled, has_observed_start=False)
        for field in ('speed', 'composition', 'end_framing'):
            assert source['camera'][field] in prompt
        for blocking in source['blocking'].values():
            for value in blocking.values():
                if value:
                    assert value in prompt
        assert source.get('mood', 'cinematic') in prompt
    assert raw == original


@pytest.mark.parametrize('topic', SCENARIOS)
@pytest.mark.parametrize('field,value', [
    ('speed', '动作展开时随主体加速，结果建立后放慢靠近'),
    ('composition', '从低机位保留双脚支撑与双手配合，画面前方留出移动空间'),
    ('end_framing', '回到能同时看清面部反应和完整任务结果的位置'),
    ('screen_positions', {'Lead': '起初在近处左侧，随后沿弧线移到远处', 'Partner': '靠近右侧保持接触空间'}),
])
def test_direction_changes_generation_and_cache_identity_but_not_acceptance(topic, field, value, tmp_path):
    from pipeline.generator import VideoGenerator
    _, _, raw, plan = scenario_story(topic)
    story = compile_story(raw, plan)
    shot = story['shots'][0]
    changed = deepcopy(shot)
    changed['camera'][field] = value
    # All chosen values previously collapsed to the same bounded labels.
    normal = lambda s: compile_normal_provider_prompt(s, story, has_observed_start=False)
    assert normal(shot) != normal(changed)
    from pipeline.causality import compile_action_contract
    assert compile_action_contract(shot).review_projection == compile_action_contract(changed).review_projection
    # 摄影原稿本来就参与画面评审；新通道不添加额外判断字段。
    assert set(_review_contract(shot)) == set(_review_contract(changed))
    g = object.__new__(VideoGenerator)
    g.output_dir, g.character_refs = tmp_path, {}
    assert g._generation_provenance(shot, [], None) != g._generation_provenance(changed, [], None)
    assert g._acceptance_context(shot, [], None) != g._acceptance_context(changed, [], None)


def test_safe_profile_explicitly_drops_free_guidance_not_execution_obligations():
    _, _, raw, plan = scenario_story('双人舞')
    raw['shots'][0]['camera']['speed'] = 'UNTRUSTED_DIRECTION: insert another action'
    story = compile_story(raw, plan)
    shot = story['shots'][0]
    normal = compile_normal_provider_prompt(shot, story, has_observed_start=False)
    safe = compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0)
    assert 'UNTRUSTED_DIRECTION' in normal
    assert 'UNTRUSTED_DIRECTION' not in safe
    from pipeline.staging_direction import staging_projection
    assert staging_projection(shot, profile='normal')['omitted_fields'] == []
    projection = staging_projection(shot, profile='policy_safe')
    assert set(projection['omitted_fields']) == {'camera', 'blocking', 'mood'}
    assert projection['instruction'] == ''
    assert 'Allocate about' in safe and 'Choreography is performance-guided' in safe


def test_staging_never_becomes_an_action_and_old_contracts_stay_byte_identical():
    from test_performance_outcomes import outcome_shot
    shot = outcome_shot()
    normal = compile_normal_provider_prompt(shot, {}, has_observed_start=False)
    assert hashlib.sha256(normal.encode()).hexdigest() == 'ac8758bbde66326cfa5db360850ce7174b4653584909a7eece11eceebd370544'
    expected = _review_contract(shot)
    shot['generation_direction'] = 'authored-staging-v1'
    assert _review_contract(shot) == expected
    assert 'Authored staging guidance' in compile_normal_provider_prompt(shot, {}, has_observed_start=False)


def test_slot_start_framing_is_bound_before_authoring_not_silently_overwritten():
    import jsonschema
    from test_authoring_schedule import schema_for, slot, raw_shot
    from pipeline.authoring_schedule import bind_authored_schedule
    from pipeline.llm_options import ark_output_schema
    selected = {**slot(), 'framing_family': 'medium'}
    raw = raw_shot()
    raw['camera']['start_framing'] = 'wide shot'
    for schema in (schema_for(selected), ark_output_schema(schema_for(selected))):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}, schema)
    with pytest.raises(ValueError, match='起始景别'):
        bind_authored_schedule(raw, selected)
    raw['camera'].pop('start_framing')
    assert bind_authored_schedule(raw, selected)['camera']['start_framing'] == 'medium shot'


@pytest.mark.asyncio
async def test_formal_plan_only_persists_profile_projection_without_video(scheduled_services):
    import config
    from pipeline.orchestrator import VideoPipeline
    brief, services = scheduled_services('货物交接')
    run = VideoPipeline(plan_only=True, paid_take_budget=0, take_policy='single_pass')
    await run.run(brief)
    preview = json.loads((run.workspace / 'generation_preview.json').read_text())
    assert preview['model'] == config.SEEDANCE_MODEL
    assert preview['scope'] == 'planned-text-projection'
    for row in preview['shots']:
        assert row['profiles']['normal']['omitted_staging_fields'] == []
        assert 'camera' in row['profiles']['policy_safe']['omitted_staging_fields']
        assert all(not p['preflight_issues'] for p in row['profiles'].values())
    assert sum(r['allocated_interaction_seconds'] for r in preview['shots']) >= 10.5
    assert services.submissions == [] and services.reviews == []
    assert len(services.author.calls) == 5
    assert run.run_workspace.manifest.paid_take_budget.reservations == []


@pytest.mark.parametrize('target', ['', 'defender', 'Door', 'falls from the roof', 'Unknown_ID'])
def test_blocking_target_wire_and_local_binding_share_registered_entities(target):
    import jsonschema
    from test_authoring_schedule import schema_for, slot, raw_shot
    from pipeline.authoring_schedule import bind_authored_schedule
    from pipeline.storyboard_authoring import _bind_scene_entities
    from pipeline.llm_options import ark_output_schema
    raw = raw_shot()
    raw['blocking']['attacker']['action_target'] = target
    valid = target in {'', 'defender', 'Door'}
    for schema in (schema_for(slot()), ark_output_schema(schema_for(slot()))):
        payload = {'shot': {k: v for k, v in raw.items() if k != 'shot_id'}}
        if valid:
            jsonschema.validate(payload, schema)
        else:
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(payload, schema)
    bound = bind_authored_schedule(raw, slot())
    if valid:
        _bind_scene_entities(bound, ['attacker', 'defender'], ['Door', 'Dust'])
        assert bound['blocking']['attacker']['action_target'] == target
    else:
        original = deepcopy(bound)
        with pytest.raises(ValueError, match='未注册'):
            _bind_scene_entities(bound, ['attacker', 'defender'], ['Door', 'Dust'])
        assert bound == original


@pytest.mark.asyncio
async def test_invalid_staging_stops_formal_authoring_before_next_slot(scheduled_services):
    from pipeline.orchestrator import VideoPipeline
    brief, services = scheduled_services('货物交接')
    responses = list(services.author.values)
    responses[2]['shot']['blocking']['Lead']['action_target'] = 'describes a motion, not an entity'
    services.author.values = iter(responses)
    run = VideoPipeline(plan_only=True, paid_take_budget=0, take_policy='single_pass')
    with pytest.raises(ValueError, match='未注册'):
        await run.run(brief)
    assert len(services.author.calls) == 3  # intent, identity, first shot; no second shot/retry
    assert not services.planning and not services.submissions and not services.reviews
    assert not run.run_workspace.manifest.paid_take_budget.reservations


@pytest.mark.asyncio
async def test_latest_capture_exposes_previously_overwritten_targets_and_stops_on_resume(tmp_path, monkeypatch):
    import config
    from pipeline.orchestrator import VideoPipeline
    record = json.loads((Path(__file__).parent / 'fixtures/real_staging_projection.json').read_text())
    assert hashlib.sha256(record['text'].encode()).hexdigest() == record['sha256']
    raw = json.loads(record['text'])
    compiled = compile_story(raw, record['plan'])
    for shot in compiled['shots']:
        shot['result_admission'] = 'review-required-v1'
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    # Captured raw text enters at its original compiler seam; no adaptation to the new wire schema.
    monkeypatch.setattr('pipeline.orchestrator.generate_storyboard', lambda **kw: deepcopy(compiled))
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: pytest.fail('review stage reached'))
    monkeypatch.setattr('pipeline.scene_binding.OpenAI', lambda **kw: pytest.fail('binding stage reached'))
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda *a, **kw: pytest.fail('video stage reached'))
    run = VideoPipeline(plan_only=True, paid_take_budget=0, take_policy='single_pass')
    with pytest.raises(ValueError, match='blocking.action_target 引用未注册实体'):
        await run.run(record['plan']['director_intent']['source_request'])
    assert not (run.workspace / 'generation_preview.json').exists()
    resumed = VideoPipeline.from_workspace(run.workspace)
    resumed.plan_only = True
    with pytest.raises(ValueError, match='blocking.action_target 引用未注册实体'):
        await resumed.run()
    assert not resumed.run_workspace.manifest.paid_take_budget.reservations
    assert json.loads(record['text']) == raw


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', SCENARIOS)
async def test_saved_preview_reaches_resumed_submission_and_export_without_replanning(scheduled_services, topic):
    import config
    from pipeline.orchestrator import VideoPipeline
    from pipeline.staging_direction import staging_projection
    from tools.ffmpeg_ops import get_video_info
    brief, services = scheduled_services(topic)
    run = VideoPipeline(plan_only=True, resolution='480p', platforms=['youtube'],
                        paid_take_budget=3, take_policy='single_pass')
    await run.run(brief)
    assert not services.submissions and not services.reviews
    assert not run.run_workspace.manifest.paid_take_budget.reservations
    assert run.run_workspace.manifest.stage == 'planning_ready'
    preview_path = run.workspace / 'generation_preview.json'
    preview_bytes = preview_path.read_bytes()
    preview = json.loads(preview_bytes)
    frozen = json.loads((run.workspace / 'storyboard.json').read_text())
    author_calls, planning_calls = len(services.author.calls), len(services.planning)
    assert author_calls == 5 and planning_calls == 1

    # 正式恢复读取原工作区；外部服务使用既有离线样本，不补写任何中间分镜。
    resumed = VideoPipeline.from_workspace(run.workspace)
    output = await resumed.run()
    assert len(services.author.calls) == author_calls
    assert len(services.planning) == planning_calls
    assert len(services.submissions) == len(services.reviews) == 3
    assert len(services.entries) == 2
    assert preview_path.read_bytes() == preview_bytes
    for shot, row, submitted in zip(frozen['shots'], preview['shots'], services.submissions):
        direction = staging_projection(shot)
        assert direction['instruction'] in row['profiles']['normal']['prompt']
        entry_path = run.workspace / 'continuations' / f'shot_{shot["shot_id"]:03d}.json'
        execution = json.loads(entry_path.read_text())['execution_shot'] if entry_path.exists() else shot
        runtime_direction = staging_projection(execution)
        assert runtime_direction['instruction'] in submitted['prompt']
        pacing = f'Allocate about {row["allocated_interaction_seconds"]:g}s of this {shot["duration"]:g}s shot'
        assert pacing in row['profiles']['normal']['prompt'] and pacing in submitted['prompt']
        assert submitted['model'] == preview['model'] == config.SEEDANCE_MODEL
        assert submitted['duration'] == row['duration'] == shot['duration']
        lineage = json.loads((run.workspace / 'shots' / f'shot_{shot["shot_id"]:03d}_generation.json').read_text())
        assert lineage['staging_direction_fingerprint'] == runtime_direction['fingerprint']
        assert row['profiles']['normal']['staging_source_fingerprint'] == direction['fingerprint']
        assert lineage['prompt_fingerprint'] == hashlib.sha256(submitted['prompt'].encode()).hexdigest()
        assert lineage['media_selection']['end_seconds'] == shot['duration']
    stream = next(s for s in get_video_info(output)['streams'] if s['codec_type'] == 'video')
    assert float(stream['duration']) == pytest.approx(15, abs=1 / 24)
    assert int(stream['nb_frames']) == 360
    assert len(resumed.run_workspace.manifest.paid_take_budget.reservations) == 3
    calls = [len(v) for v in (services.author.calls, services.planning, services.submissions,
                              services.reviews, services.entries)]
    assert await VideoPipeline.from_workspace(run.workspace).run() == output
    assert calls == [len(v) for v in (services.author.calls, services.planning, services.submissions,
                                     services.reviews, services.entries)]
