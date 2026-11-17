"""Generation-side replay: captured requests plus explicitly offline scene annotations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.action_results import required_outcomes
from pipeline.causality import compile_action_contract
from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
from pipeline.result_admission import apply_result_admission
from test_full_production_flow import production_services


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        pytest.fail('offline generation replay attempted a network connection')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)


def captured_execution(*, recompile=True):
    sources = {}
    for name in ('real_review_scope_failure.json', 'real_generation_admission.json'):
        record = json.loads((Path(__file__).parent / 'fixtures' / name).read_text())
        for path, source in record['sources'].items():
            assert hashlib.sha256(source['text'].encode()).hexdigest() == source['sha256']
            sources[path] = json.loads(source['text'])
    admitted = apply_result_admission(sources['storyboard.json'], sources['result_admission.json']['units'])
    assert admitted == sources['result_admission.json']['execution_storyboard']
    if recompile:
        from pipeline.action_authoring import compile_authored_storyboard
        from pipeline.storyboard import _compile_storyboard_contract
        previous = admitted
        plan = previous['production_plan']
        compiled = _compile_storyboard_contract(
            compile_authored_storyboard(sources['planning_authoring.json'], plan),
            previous['aspect_ratio'], previous['resolution'], previous['style'], production_plan=plan)
        admitted = apply_result_admission(compiled, sources['result_admission.json']['units'])
        for authored, old, new in zip(sources['planning_authoring.json']['shots'], previous['shots'], admitted['shots'], strict=True):
            for unit, before, after in zip(authored['action_plan'], old['action_beats'], new['action_beats'], strict=True):
                # 旧缺省曾把空目标回填成另一交互的目标；新版以原稿归属为准，其余字段保持。
                expected = {**before, 'actor': unit['actor'], 'target': unit.get('target', '')}
                assert {k: v for k, v in expected.items() if k not in ('result_directions', 'result_functions')} == {
                    k: v for k, v in after.items() if k not in ('result_directions', 'result_functions')}
            for field in ('start_state', 'end_state', 'camera', 'blocking'):
                assert old[field] == new[field]
            assert new['generation_execution'] == 'scene-instructions-v2'
    return sources, admitted


def normal(shot, story=None):
    return compile_normal_provider_prompt(shot, story or {}, has_observed_start=False)


def test_captured_preview_is_the_submitted_text_and_exposes_spatial_loss():
    sources, story = captured_execution()
    submitted = sources['generation_preview.json']['shots'][0]['profiles']['normal']['prompt']
    lineage = sources['shots/shot_001_generation.json']
    assert hashlib.sha256(submitted.encode()).hexdigest() == lineage['prompt_fingerprint']
    assert lineage['model_used'] == 'doubao-seedance-2.0-fast'
    assert lineage['reference_fingerprints'] == []
    beat = story['shots'][0]['action_beats'][1]
    assert beat['action'] in submitted  # No claim that the requested result was entirely omitted.
    assert beat['interaction_geometry']['source'] not in submitted
    assert beat['interaction_geometry']['effect_region'] not in submitted
    assert story['shots'][0]['production_slot']['creative_brief'] in submitted


def test_original_four_shots_lower_to_local_actions_with_their_own_spatial_context():
    _, story = captured_execution()
    original = deepcopy(story)
    for shot in story['shots']:
        prompt = normal(shot, story)
        assert shot['production_slot']['creative_brief'] not in prompt
        for index, beat in enumerate(shot['action_beats'], 1):
            assert beat['action'] in prompt
            if beat['activity'] != 'interaction':
                continue
            section = prompt.split(f'Beat {index}:', 1)[1].split(f'Beat {index + 1}:', 1)[0]
            for field in ('source', 'effect_region', 'actor_screen_position', 'target_screen_position'):
                assert beat['interaction_geometry'][field] in section
            for claim in required_outcomes(beat):
                assert claim['state'] in section
        assert shot['start_state']['prop_state'] in prompt
    assert story == original


@pytest.mark.parametrize('source,region', [
    ('sender supporting palm', 'loading bench inside the yellow transfer zone'),
    ('dancer joined hands', 'the shared turning axis above the stage'),
    ('operator insertion tool', 'panel socket on the upper equipment rail'),
    ('rescuer secured rope', 'anchored guide channel above the safe ledge'),
])
def test_event_space_is_preserved_as_execution_guidance_not_added_review_obligations(source, region):
    _, story = captured_execution()
    shot = story['shots'][0]
    baseline = compile_action_contract(shot).review_projection
    shot['action_beats'][1]['interaction_geometry'].update(source=source, effect_region=region)
    prompt = normal(shot, story)
    assert source in prompt and region in prompt
    assert compile_action_contract(shot).review_projection == baseline
    # Safe recompilation still excludes free direction from a rejected request.
    safe = compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0)
    assert source not in safe and region not in safe


def test_authored_motion_in_composition_is_not_promoted_to_an_initial_position_lock():
    from pipeline.visual_invariants import compile_visual_invariants
    _, story = captured_execution()
    shot = story['shots'][0]
    shot['camera']['screen_positions'] = {'XiaoAn': 'starts left and crosses to the right during the action'}
    prompt = normal(shot, story)
    assert shot['camera']['screen_positions']['XiaoAn'] in prompt
    assert 'begin with these screen positions' not in prompt
    assert 'screen positions: XiaoAn=left right' not in prompt
    assert 'initial_screen_positions' not in compile_visual_invariants(shot).review_projection
    assert shot['start_state']['prop_state'] in prompt


def test_only_admitted_scene_execution_can_stop_rebroadcasting_the_whole_film():
    _, story = captured_execution()
    shot = story['shots'][0]
    brief = shot['production_slot']['creative_brief']
    assert brief not in normal(shot, story)
    shot['result_admission'] = 'review-required-v1'
    for beat in shot['action_beats']:
        beat.pop('result_directions', None)
        beat.pop('result_functions', None)
    assert brief in normal(shot, story)


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_other_scene_sequences_keep_authored_actions_but_do_not_reassign_the_full_story(topic):
    from test_creation_schedule_flow import scenario_story
    from test_staging_direction import compile_story
    from test_action_result_flow import annotated_verdict
    _, _, raw, plan = scenario_story(topic)
    story = compile_story(raw, plan)
    for shot in story['shots']:
        shot['result_admission'] = 'review-required-v1'
    story = apply_result_admission(story, annotated_verdict(story)['units'])
    original = deepcopy(story)
    for shot in story['shots']:
        prompt = normal(shot, story)
        assert shot['production_slot']['creative_brief'] not in prompt
        for beat in shot['action_beats']:
            assert beat['action'] in prompt
        for other in story['shots']:
            if other['shot_id'] != shot['shot_id']:
                for beat in other['action_beats']:
                    if beat['activity'] == 'interaction':
                        assert beat['action'] not in prompt
    assert story == original


@pytest.mark.parametrize('brief,prop,action,result,visual', [
    ('合作搬运包裹', 'Parcel', 'guides the parcel across the support', 'parcel rests in the receiver hands',
     'A warehouse loading bench with bright side lighting'),
    ('装配面板', 'Panel', 'seats the panel in the receiver rail', 'panel sits securely in the rail',
     'An assembly bench with a clear view of the equipment rail'),
])
@pytest.mark.asyncio
async def test_formal_sdk_requests_and_complete_export_keep_local_event_space(
        production_services, monkeypatch, brief, prop, action, result, visual):
    # 原有正式流程夹具只替换外部服务，执行真实规划编译、续拍、SDK组包与FFmpeg整片导出。
    from test_generation_information_flow import test_formal_fast_chain_reaches_real_sdk_payload_and_export as run_flow
    await run_flow(production_services, monkeypatch, brief, prop, action, result, visual)
    _, services = production_services
    assert len(services.submissions) == 3
    for request in services.submissions:
        prompt = request['content'][0]['text']
        assert 'physical source: hands' in prompt
        assert 'action region: shared workspace' in prompt
        assert 'Whole-film brief and current shot function' not in prompt
        assert visual in prompt and action in prompt and result in prompt


@pytest.mark.asyncio
async def test_captured_plan_reaches_fast_sdk_without_editing_the_four_shots():
    from types import SimpleNamespace
    from pipeline.provider_prompt import compile_generation_preview
    from tools.seedance_api import SeedanceAPI
    sources, story = captured_execution()
    source = deepcopy(story)
    model = sources['shots/shot_001_generation.json']['model_used']
    preview = compile_generation_preview(story, model=model)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id=f'offline-{len(calls)}')
    def get(*, task_id):
        return SimpleNamespace(status='succeeded', content=SimpleNamespace(video_url=task_id))
    api = object.__new__(SeedanceAPI)
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(tasks=SimpleNamespace(create=create, get=get)))
    for shot, row in zip(story['shots'], preview['shots']):
        assert all(not profile['preflight_issues'] for profile in row['profiles'].values())
        text = row['profiles']['normal']['prompt']
        response = await api.generate(text, duration=shot['duration'], ratio=story['aspect_ratio'],
                                      resolution=story['resolution'], model=model)
        assert response['status'] == 'succeeded'
        assert calls[-1]['content'] == [{'type': 'text', 'text': text}]
        assert calls[-1]['model'] == model
        assert calls[-1]['duration'] == shot['duration']
    # This checks planned text delivery; later real shots still require accepted runtime references.
    assert story == source and len(calls) == 4


def test_execution_scope_is_compiler_owned_and_historical_sources_are_not_migrated():
    from pipeline.models import validate_storyboard
    from pipeline.staging_direction import staging_projection
    _, old = captured_execution(recompile=False)
    _, new = captured_execution()
    assert all('generation_execution' not in s for s in validate_storyboard(old)['shots'])
    assert all(s['generation_execution'] == 'scene-instructions-v2' for s in validate_storyboard(new)['shots'])
    for before, after in zip(old['shots'], new['shots']):
        assert staging_projection(before)['fingerprint'] != staging_projection(after)['fingerprint']
    from pipeline.action_authoring import AuthoredShot
    sources, _ = captured_execution()
    forged = {**sources['planning_authoring.json']['shots'][0], 'generation_execution': 'local-events-v1'}
    with pytest.raises(ValueError):
        AuthoredShot.model_validate(forged)


@pytest.mark.parametrize('version', ['', False, {}, 'local-events-v3'])
def test_unknown_execution_projection_cannot_reach_generation(version):
    _, story = captured_execution()
    shot = story['shots'][0]
    shot['generation_execution'] = version
    with pytest.raises(ValueError, match='未知的逐镜执行投影'):
        normal(shot, story)
