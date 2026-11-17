"""素材的可用范围、观察位置和新片段操作不能混为一谈。全程离线。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline import provider_materials as media
from test_provider_materials import original
from test_goal_allocation import offline


def terminal_state(original):
    state, video = original(kind='selected_state', character=None, timestamp=5.95, box=None)
    media.bind_derivation(state, str(video), kind='selected_state', timestamp=5.95,
                          selected_end_seconds=6.08)
    return state


@pytest.mark.parametrize('characters', [('Yun', 'Zhao'), ('mother', 'daughter'),
                                       ('dancer', 'partner'), ('rescuer', 'hiker')])
def test_same_source_identity_and_state_compile_one_temporal_operand(original, characters):
    state = terminal_state(original)
    refs = [state, *(original(character=c)[0] for c in characters)]
    plan = media.material_plan(refs, 'reference_image', execution={'boundary': 'intentional_cut'})
    operation = plan['operation']
    assert operation['task'] == 'extend_after'
    assert operation['source']['media_label'] == '视频1'
    assert operation['camera_boundary'] == 'cut_to_current_shot'
    assert operation['transport_control'] == 'prompt_dispatch'
    assert len(plan['content']) == 1  # 不复制、裁剪或删除原视频。
    assert [b['usage'] for b in plan['bindings']] == ['selected_state', 'identity', 'identity']
    prompt = media.material_prompt('Image 1 provides state. Images 2-3 provide identities.', plan)
    assert prompt.startswith('向后延长视频1')
    assert '原始素材引用' not in prompt
    assert media.plan_evidence(plan)['operation'] == operation


def test_identity_source_is_not_a_temporal_predecessor(original):
    image, _ = original()
    plan = media.material_plan([image], 'reference_image', execution={'boundary': 'intentional_cut'})
    assert plan['operation']['task'] == 'reference_generation'
    assert plan['operation']['source'] is None
    assert not media.material_prompt('Image 1 appears in a new place.', plan).startswith('向后延长')


def test_interior_observation_is_not_the_original_end(original):
    state, _ = original(kind='selected_state', character=None, timestamp=1.9, box=None)
    plan = media.material_plan([state], 'reference_image', execution={'boundary': 'intentional_cut'})
    assert plan['operation']['task'] == 'continue_from_observation'
    assert plan['operation']['transport_control'] == 'semantic_reference'
    assert not media.material_prompt('Image 1 state.', plan).startswith('向后延长')


@pytest.mark.asyncio
async def test_sdk_rejects_operation_or_dispatch_drift_before_create(original):
    from test_seedance_api_resume import api_with_fake_tasks
    state = terminal_state(original)
    plan = media.material_plan([state], 'reference_image', execution={'boundary': 'seamless'})
    prompt = media.material_prompt('Continue the movement.', plan)
    for tamper in ('operation', 'prompt'):
        api, tasks = api_with_fake_tasks()
        submitted = deepcopy(plan)
        if tamper == 'operation':
            submitted['operation']['source']['media_label'] = '视频2'
        result = await api.generate(prompt=prompt if tamper == 'operation' else 'Reference the old video.',
            image_urls=[state], image_role='reference_image', provider_materials=submitted)
        assert result['status'] == 'failed'
        assert result['error_type'] == 'reference_validation'
        assert tasks.created == 0


def test_real_failure_records_distinguish_reenactment_from_reused_download():
    root = Path(__file__).parents[1] / 'output/20261112_184640_228865'
    if not root.exists():
        pytest.skip('本机真实原片不可用')
    request = json.loads((root / 'submission_requests/shot_002_attempt_001.json').read_text())
    assert '向后延长视频1' not in request['prompt']
    assert '5.950秒' in request['prompt']
    digests = []
    for name in ('shot_001', 'shot_002_rejected_1'):
        digest = hashlib.sha256((root / f'shots/{name}.mp4').read_bytes()).hexdigest()
        receipt = json.loads((root / f'shots/{name}_provider_output.json').read_text())
        assert receipt['video_sha256'] == digest
        digests.append(digest)
    assert digests[0] != digests[1]
    review = json.loads((root / f'semantic_reviews/{digests[1]}.json').read_text())
    assert review['review']['accepted'] is False


@pytest.mark.parametrize('end', [None, 2., 6.08])
def test_selected_end_survives_a_camera_view_without_becoming_a_crop(original, tmp_path, end):
    state, video = original(kind='selected_state', character=None, timestamp=1.9, box=None)
    media.bind_derivation(state, str(video), kind='selected_state', timestamp=1.9,
                          selected_end_seconds=end)
    view = tmp_path / 'view.jpg'
    view.write_bytes(b'offline local view')
    media.bind_view(str(view), state, {'left': .2, 'top': .1, 'scale': .7})
    plan = media.material_plan([str(view)], 'reference_image', execution={'boundary': 'intentional_cut'})
    assert plan['operation']['task'] == ('extend_after' if end == 6.08 else 'continue_from_observation')
    assert plan['content'][0]['video_url']['url'] == 'https://example.com/original-1.mp4'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, -1, 100, '6'])
def test_invalid_selection_cannot_become_temporal_authority(original, bad):
    state, video = original(kind='selected_state', character=None, timestamp=5.95, box=None)
    with pytest.raises(media.GenerationReadinessError, match='结束位置'):
        media.bind_derivation(state, str(video), kind='selected_state', timestamp=5.95,
                              selected_end_seconds=bad)


def test_different_identity_video_does_not_steal_the_continuation_operand(original):
    identity, _ = original(1)
    state, video = original(2, kind='selected_state', character=None, timestamp=5.95, box=None)
    media.bind_derivation(state, str(video), kind='selected_state', timestamp=5.95, selected_end_seconds=6.08)
    plan = media.material_plan([identity, state], 'reference_image', execution={'boundary': 'seamless'})
    assert plan['operation']['source']['media_label'] == '视频2'
    assert plan['operation']['camera_boundary'] == 'continuous_camera'
    assert media.material_prompt('Images 1-2', plan).startswith('向后延长视频2')


def test_two_state_sources_cannot_silently_pick_one(original):
    first = terminal_state(original)
    second, _ = original(2, kind='selected_state', character=None, timestamp=5.95, box=None)
    with pytest.raises(media.GenerationReadinessError, match='多个动作承接'):
        media.material_plan([first, second], 'reference_image', execution={'boundary': 'intentional_cut'})


@pytest.mark.asyncio
async def test_actual_failure_source_reaches_sdk_as_extension_and_keeps_cut(tmp_path, monkeypatch):
    """原片重建正式媒体边界；SDK 为替身，不推断新请求会产生合格画面。"""
    import shutil
    import time
    import config
    from pipeline.generator import VideoGenerator
    from pipeline.provider_operation import execution_intent, operation_prefix
    from test_seedance_api_resume import api_with_fake_tasks
    fixture = json.loads((Path(__file__).parent / 'fixtures/real_reference_replay_failure.json').read_text())
    root = Path(__file__).parents[1] / 'output' / fixture['run']
    if not root.exists():
        pytest.skip('本机真实原片不可用')
    monkeypatch.setattr(config, 'ARK_API_KEY_SEEDANCE', 'explicit-offline-replay-key')
    gen = VideoGenerator(str(tmp_path), review_content=False)
    source = tmp_path / 'shots/shot_001.mp4'
    shutil.copyfile(root / 'shots/shot_001.mp4', source)
    media.capture_output(str(source), {'provider_task_id': 'offline-replay', 'provider_output': {
        'task_id': 'offline-replay', 'created_at': int(time.time()), 'model': 'doubao-seedance-2.0-fast',
        'video_url': 'https://example.com/unchanged-original.mp4',
        'last_frame_url': 'https://example.com/official-tail.png'}})
    state = gen._extract_local_tail_frame(1, str(source), fixture['first_shot'])
    selected = media._read(media.derivation_path(state))
    assert selected['selected_end_seconds'] == 6
    assert selected['timestamp'] == 5.95
    shot = fixture['execution_shot']
    # 使用真实取景窗口，但独立测试来源，不能冒充原运行的签名产物。
    view = tmp_path / 'view.jpg'
    view.write_bytes(b'local observation, never submitted')
    window = shot['reference_view']['window']
    media.bind_view(str(view), state, window)
    plan = media.material_plan([str(view)], 'reference_image', execution=execution_intent(shot))
    assert plan['operation']['task'] == 'extend_after'
    assert plan['operation']['source']['source_sha256'] == fixture['materials']['bindings'][0]['source_sha256']
    from pipeline.camera_contract import compile_camera
    camera, _ = compile_camera(shot)
    prompt = media.material_prompt('. '.join(camera), plan)
    assert 'tracking' in prompt and 'medium shot' in prompt
    api, tasks = api_with_fake_tasks()
    calls = []
    from types import SimpleNamespace
    tasks.create = lambda **request: (calls.append(request) or SimpleNamespace(id='offline-task'))
    await api.generate(prompt=prompt, duration=6, image_urls=[str(view)], image_role='reference_image',
                       provider_materials=plan, model='doubao-seedance-2.0-fast', timeout=1)
    assert len(calls) == 1 and calls[0]['content'][0]['text'].startswith(operation_prefix(plan['operation']))
    assert calls[0]['content'][1:] == plan['content']
    assert 'data:image' not in json.dumps(calls)
    for name, digest in fixture['file_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
