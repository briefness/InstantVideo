"""参考像素、世界状态和摄影目标分离；不调用云端或改写真实失败记录。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from test_goal_allocation import offline
from test_provider_materials import original  # noqa: F401
from test_view_execution import view_case


@pytest.mark.parametrize('observation', ['固定机位，全程保持宽景取景', 'プラン全景', '', 'medium shot'])
def test_actual_pixels_not_free_text_labels_control_view_availability(tmp_path, observation):
    from pipeline.reference_view import camera_view_context
    shot, previous, tail, _, _ = view_case(tmp_path)
    shot['camera']['start_framing'] = 'medium shot'
    previous['observed_end_state']['camera'] = observation
    context = camera_view_context(shot, previous, tail)
    assert context is not None
    assert context['camera_plan'] == shot['camera']
    assert context['execution_capability']['mode'] == 'first_frame_pixels'


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_original_video_view_keeps_separate_identity_bindings(tmp_path, original, topic):
    from pipeline import provider_materials as media
    from pipeline.reference_view import camera_view_context, bind_view, select_view
    from pipeline.generator import VideoGenerator
    from test_creation_schedule_flow import scenario_story
    from test_staging_direction import compile_story
    _, _, raw, plan = scenario_story(topic)
    story = compile_story(raw, plan)
    shot = deepcopy(story['shots'][1])
    shot.update(continuity_from_previous='intentional_cut', completion_policy='goal-allocation-v1')
    shot['camera'].update(start_framing='medium shot', axis_change='hold', composition='two shot')
    tail, video = original(2, kind='selected_state', character=None, timestamp=5.95, box=None)
    assert cv2.imwrite(tail, np.zeros((496, 864, 3), dtype=np.uint8))
    media.bind_derivation(tail, str(video), kind='selected_state', timestamp=5.95)
    previous = {'observed_end_state': {'camera': '宽景'}}
    context = camera_view_context(shot, previous, Path(tail))
    assert context is not None
    assert context['execution_capability']['mode'] == 'original_video_guidance'
    window = {'left': .1, 'top': .1, 'scale': .75,
              'regions': {e: [.2, .2, .7, .7] for e in context['region_entities']}}
    decision = select_view({'regions': window['regions']}, context)
    assert decision['mode'] == 'original_video_guidance'
    shot['reference_view'] = bind_view(decision['window'], context, hashlib.sha256(Path(tail).read_bytes()).hexdigest())
    gen = VideoGenerator(output_dir=str(tmp_path), review_content=False)
    gen.character_refs = {name: original(1, character=name)[0] for name in shot['characters']}
    refs, role = gen._build_image_refs(shot, tail, previous)
    assert role == 'reference_image'
    assert refs[1:] == list(gen.character_refs.values())
    material = media.material_plan(refs, role, live=True)
    assert len(material['content']) == 2
    assert [b['source_task_id'] for b in material['bindings']] == ['offline-2'] + ['offline-1'] * len(shot['characters'])
    assert all(c['role'] == 'reference_video' for c in material['content'])
    assert '建立新镜头取景' in material['bindings'][0]['label']
    assert 'data:image' not in json.dumps(material)
    before = deepcopy((shot['camera'], shot['action_beats']))
    assert gen._build_image_refs(shot, tail, previous) == (refs, role)
    assert (shot['camera'], shot['action_beats']) == before


def test_real_failure_new_planning_receives_pixels_and_old_resume_is_unchanged(tmp_path):
    from pipeline.continuation import prepare_continuation
    from pipeline.generator import VideoGenerator
    from test_director_intent import Responses
    root = Path(__file__).parents[1] / 'output/20261109_184525_100966'
    if not root.exists():
        pytest.skip('private original record unavailable; portable cross-scene cases still run')
    stored = json.loads((root / 'continuations/shot_002.json').read_text())
    manifest = json.loads((root / 'run_manifest.json').read_text())
    story = json.loads((root / 'result_admission.json').read_text())['execution_storyboard']
    shot = story['shots'][1]
    previous = {**story['shots'][0], 'observed_end_state': manifest['shots']['1']['observed_end_state']}
    VideoGenerator._reconcile_start_state(shot, previous)
    tail = root / 'shots/shot_001_lastframe.jpg'
    assert 'camera_view' not in stored['input']
    records = [root / 'run_manifest.json', root / 'continuations/shot_002.json', tail]
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in records]
    # 原记录只读重放，不对旧拒绝稿增加窗口或重新裁切；新模型响应只用明确替身。
    old = tmp_path / 'old'
    (old / 'continuations').mkdir(parents=True)
    shutil.copyfile(root / 'continuations/shot_002.json', old / 'continuations/shot_002.json')
    sdk = Responses()
    resumed = prepare_continuation(shot, previous, str(tail), old, client=sdk, allow_create=False)
    assert resumed == stored['execution_shot'] and sdk.calls == []
    new_sdk = Responses({'compatible': True, 'reason': 'Explicit offline annotation; no view proposal.',
                         'entry_action': None, 'view_window': None})
    fresh = prepare_continuation(shot, previous, str(tail), tmp_path / 'new', client=new_sdk)
    call = new_sdk.calls[0]
    context = json.loads(call['messages'][-1]['content'])
    assert context['camera_view']['execution_capability']['mode'] == 'original_video_guidance'
    assert call['messages'][1]['content'][0]['type'] == 'image_url'
    assert 'crop these exact pixels as the next first frame' not in call['messages'][0]['content']
    assert fresh['camera'] == shot['camera'] and fresh['action_beats'] == shot['action_beats']
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in records] == hashes


@pytest.mark.parametrize('mutation', ['missing_receipt', 'changed_source', 'changed_capability'])
def test_planned_reference_route_cannot_change_at_materialization(tmp_path, original, mutation):
    from pipeline import provider_materials as media
    from pipeline.reference_view import camera_view_context, bind_view, materialize_view
    from pipeline.readiness import GenerationReadinessError
    shot, previous, _, _, _ = view_case(tmp_path)
    tail, video = original(kind='selected_state', character=None, box=None)
    assert cv2.imwrite(tail, np.zeros((496, 864, 3), dtype=np.uint8))
    media.bind_derivation(tail, str(video), kind='selected_state', timestamp=5.95)
    context = camera_view_context(shot, previous, Path(tail))
    window = {'left': .1, 'top': .1, 'scale': .75,
              'regions': {e: [.2, .2, .7, .7] for e in context['region_entities']}}
    shot['reference_view'] = bind_view(window, context, hashlib.sha256(Path(tail).read_bytes()).hexdigest())
    if mutation == 'missing_receipt':
        media.derivation_path(tail).unlink()
    elif mutation == 'changed_source':
        video.write_bytes(b'changed source')
    else:
        shot['reference_view']['execution_capability']['mode'] = 'first_frame_pixels'
    with pytest.raises((ValueError, GenerationReadinessError)):
        materialize_view(shot, tail, tmp_path)
    assert not (tmp_path / 'reference_views').exists()


def test_actual_rejected_video_still_fails_camera_with_actions_preserved(tmp_path):
    from tools.camera_measurement import measure_video_composition, composition_verdict
    root = Path(__file__).parents[1] / 'output/20261109_184525_100966'
    video = root / 'shots/shot_002_rejected_1.mp4'
    if not video.exists():
        pytest.skip('private original media unavailable')
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    assert digest == 'f75d89bc384f75e078d69b85d4432186fb83d60625bc6ec733c62707e9966f7f'
    record = json.loads((root / f'semantic_reviews/{digest}.json').read_text())
    measurement = measure_video_composition(root / 'shots/shot_001_lastframe.jpg', str(video), 0.)
    assert measurement['status'] == 'unchanged'
    assert composition_verdict(record['observation_response']['composition_change_valid'], measurement) is False
    assert record['review']['accepted'] is False and record['review']['primary_action_completed'] is True
    assert record['review']['identity_continuity_valid'] is True
    assert hashlib.sha256(video.read_bytes()).hexdigest() == digest


@pytest.mark.asyncio
async def test_view_lowering_reaches_sdk_and_missing_identity_stops(tmp_path, original):
    from types import SimpleNamespace
    from pipeline import provider_materials as media
    from pipeline.generator import VideoGenerator
    from pipeline.reference_view import camera_view_context, bind_view
    from test_seedance_api_resume import api_with_fake_tasks
    shot, previous, _, _, _ = view_case(tmp_path)
    tail, video = original(2, kind='selected_state', character=None, box=None)
    assert cv2.imwrite(tail, np.zeros((496, 864, 3), dtype=np.uint8))
    media.bind_derivation(tail, str(video), kind='selected_state', timestamp=5.95)
    context = camera_view_context(shot, previous, Path(tail))
    window = {'left': .1, 'top': .1, 'scale': .75,
              'regions': {e: [.2, .2, .7, .7] for e in context['region_entities']}}
    shot['reference_view'] = bind_view(window, context, hashlib.sha256(Path(tail).read_bytes()).hexdigest())
    gen = VideoGenerator(output_dir=str(tmp_path), review_content=False)
    gen.character_refs = {name: original(1, character=name)[0] for name in shot['characters']}
    refs, role = gen._build_image_refs(shot, tail, previous)
    material = media.material_plan(refs, role, live=True)
    api, tasks = api_with_fake_tasks()
    requests = []
    def create(**request):
        requests.append(deepcopy(request))
        return SimpleNamespace(id='offline-view-task')
    tasks.create = create
    prompt = media.material_prompt('Image 1 supplies state; Images 2-3 supply identities.', material)
    result = await api.generate(prompt=prompt, image_urls=refs, image_role=role,
                                provider_materials=material, timeout=1)
    assert result['status'] == 'succeeded' and len(requests) == 1
    assert requests[0]['content'][1:] == material['content']
    assert len(requests[0]['content']) == 3
    assert 'data:image' not in json.dumps(requests[0])
    Path(next(iter(gen.character_refs.values()))).unlink()
    with pytest.raises(ValueError, match='身份参考不可用'):
        gen._build_image_refs(shot, tail, previous)
    assert len(requests) == 1
