"""官方原始产物复用：本地派生图不继承供应商信任。"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import config
from pipeline.generator import VideoGenerator
from tools.seedance_api import SeedanceAPI
from pipeline import provider_materials as media
from pipeline.readiness import GenerationReadinessError


@pytest.fixture
def original(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'ARK_API_KEY_SEEDANCE', 'offline-fixture-key')
    monkeypatch.setattr(media.time, 'time', lambda: 1000000.)
    monkeypatch.setattr('tools.ffmpeg_ops.get_video_info', lambda _: {'streams': [{
        'codec_type': 'video', 'duration': '6.08', 'width': 864, 'height': 496,
        'avg_frame_rate': '24/1', 'codec_name': 'h264'}]})
    def make(number=1, kind='identity', character='traveler', timestamp=3., box=(.1,.1,.4,.9)):
        video = tmp_path / f'shot-{number}.mp4'
        if not video.exists():
            video.write_bytes(f'unchanged original {number}'.encode())
            result = dict(provider_task_id=f'offline-{number}', provider_output=dict(
                task_id=f'offline-{number}', created_at=999990, model='doubao-seedance-2.0-fast',
                video_url=f'https://example.com/original-{number}.mp4',
                last_frame_url=f'https://example.com/official-{number}.png'))
            media.capture_output(str(video), result)
        image = tmp_path / f'{number}-{kind}-{character}.jpg'
        image.write_bytes(f'observation {kind} {character}'.encode())
        media.bind_derivation(str(image), str(video), kind=kind, timestamp=timestamp,
                              character=character, box=box)
        return str(image), video
    return make


@pytest.mark.parametrize('characters', [('mother','daughter'), ('dancer','partner'), ('rescuer','hiker')])
def test_same_original_retains_separate_identities_and_state_without_crops(original, characters):
    tail, _ = original(kind='selected_state', character=None, timestamp=5.95, box=None)
    first, video = original(character=characters[0])
    second, _ = original(character=characters[1], box=(.6,.1,.9,.9))
    plan = media.material_plan([tail, first, second], 'reference_image', live=True)
    assert plan['content'] == [{'type': 'video_url', 'video_url': {'url': 'https://example.com/original-1.mp4'}, 'role': 'reference_video'}]
    assert len(plan['bindings']) == 3
    prompt = media.material_prompt('Image 1 controls state; Images 2-3 control identities. Run together.', plan)
    assert '5.950秒' in prompt and '3.000秒' in prompt
    assert all(name in prompt for name in characters)
    assert 'Images 2-3' not in prompt and 'Run together.' in prompt
    assert video.read_bytes() == b'unchanged original 1'


def test_original_tail_can_be_strict_first_frame(original):
    tail, _ = original(kind='official_tail', character=None, box=None)
    plan = media.material_plan([tail], 'first_frame', live=True)
    assert plan['content'][0] == {'type':'image_url', 'image_url': {'url':'https://example.com/official-1.png'}, 'role':'first_frame'}


def test_presented_state_and_next_crop_keep_original_coordinates(original, tmp_path):
    tail, video = original(kind='selected_state', character=None, timestamp=1.9, box=[.2,.2,.8,.8])
    view = tmp_path / 'next-view.jpg'
    view.write_bytes(b'local view of already presented state')
    media.bind_view(str(view), tail, dict(left=.25, top=.25, scale=.5))
    record = media._read(media.derivation_path(str(view)))
    assert record['box'] == pytest.approx([.35,.35,.65,.65])
    assert 'view_window' not in record
    assert record['video'] == str(video.resolve())
    plan = media.material_plan([str(view)], 'reference_image', live=True)
    assert plan['content'] == [{'type': 'video_url', 'video_url': {
        'url': 'https://example.com/original-1.mp4'}, 'role': 'reference_video'}]


def test_selected_state_cannot_claim_official_pixel_exact_tail(original):
    tail, _ = original(kind='selected_state', character=None, box=None)
    with pytest.raises(GenerationReadinessError, match='first_frame'):
        media.material_plan([tail], 'first_frame', live=True)


def test_new_camera_view_preserves_source_location_without_uploading_crop(original, tmp_path):
    tail, _ = original(kind='selected_state', character=None, box=None, timestamp=5.95)
    view = tmp_path / 'view.jpg'
    view.write_bytes(b'cropped preview')
    media.bind_view(str(view), tail, dict(left=.1, top=.2, scale=.5))
    plan = media.material_plan([str(view)], 'reference_image', live=True)
    assert plan['content'][0]['video_url']['url'] == 'https://example.com/original-1.mp4'
    assert '5.950秒' in plan['bindings'][0]['label'] and '建立新镜头取景' in plan['bindings'][0]['label']


@pytest.mark.parametrize('target', ['image', 'video', 'receipt', 'credentials', 'endpoint'])
def test_modified_or_cross_scope_material_never_inherits_trust(original, monkeypatch, target):
    image, video = original()
    if target in ('image','video'):
        Path(image if target == 'image' else video).write_bytes(b'changed')
    elif target == 'receipt':
        p = media.receipt_path(video)
        data = json.loads(p.read_text()); data['created_at'] += 1
        p.write_text(json.dumps(data))
    elif target == 'credentials':
        monkeypatch.setattr(config, 'ARK_API_KEY_SEEDANCE', 'different-offline-key')
    else:
        monkeypatch.setattr(config, 'ARK_BASE_URL_SEEDANCE', 'https://different.example.com/api/v3')
    with pytest.raises(GenerationReadinessError):
        media.material_plan([image], 'reference_image', live=True)


def test_expiry_stops_new_submission_but_does_not_invalidate_old_evidence(original, monkeypatch):
    image, _ = original()
    before = media.plan_evidence(media.material_plan([image], 'reference_image'))
    monkeypatch.setattr(media.time, 'time', lambda: 999990 + 24*3600)
    assert media.plan_evidence(media.material_plan([image], 'reference_image')) == before
    with pytest.raises(GenerationReadinessError, match='有效期'):
        media.material_plan([image], 'reference_image', live=True)


def test_original_video_capacity_never_solved_by_trimming_or_removing_refs(original):
    images = [original(i)[0] for i in range(1,4)]
    with pytest.raises(GenerationReadinessError, match='合计15秒'):
        media.material_plan(images, 'reference_image', live=True)


@pytest.mark.asyncio
async def test_actual_sdk_content_contains_originals_not_observation_bytes(original):
    from test_seedance_api_resume import api_with_fake_tasks
    image, _ = original()
    plan = media.material_plan([image], 'reference_image', live=True)
    api, tasks = api_with_fake_tasks()
    submitted = []
    tasks.create = lambda **kwargs: (submitted.append(kwargs) or SimpleNamespace(id='offline-task'))
    await api.generate(prompt=media.material_prompt('Image 1 runs.', plan), image_urls=[image],
                       image_role='reference_image', provider_materials=plan, timeout=1,
                       model='doubao-seedance-2.0-fast')
    assert len(submitted) == 1 and submitted[0]['model'] == 'doubao-seedance-2.0-fast'
    assert submitted[0]['content'][1:] == plan['content']
    assert 'data:image' not in json.dumps(submitted)


@pytest.mark.asyncio
async def test_untracked_project_crop_is_blocked_at_sdk_before_create(tmp_path):
    from test_seedance_api_resume import api_with_fake_tasks
    path = tmp_path / 'character_refs/old.jpg'
    path.parent.mkdir(); path.write_bytes(b'legacy crop')
    api, tasks = api_with_fake_tasks()
    with pytest.raises(GenerationReadinessError, match='历史派生素材'):
        await api.generate(prompt='continue', image_urls=[str(path)], image_role='reference_image')
    assert tasks.created == 0


@pytest.mark.parametrize('method', ['_preserve_rejected_take', '_preserve_incompatible_take'])
def test_archived_take_keeps_its_receipt_and_next_candidate_does_not_inherit_it(original, tmp_path, method):
    _, video = original()
    generator = VideoGenerator(str(tmp_path), review_content=False)
    receipt = media.receipt_path(video).read_bytes()
    archived = Path(getattr(generator, method)(1, str(video)))
    assert archived.is_file() and media.receipt_path(archived).read_bytes() == receipt
    assert not media.receipt_path(video).exists()
    media.move_original(archived, video)
    assert media.receipt_path(video).read_bytes() == receipt
    assert not media.receipt_path(archived).exists()


def test_legacy_source_is_not_assigned_some_other_candidates_receipt(original, tmp_path):
    _, video = original()
    legacy = tmp_path / 'legacy.mp4'; legacy.write_bytes(b'legacy')
    target = tmp_path / 'candidate.mp4'
    media.receipt_path(target).write_bytes(media.receipt_path(video).read_bytes())
    with pytest.raises(GenerationReadinessError, match='来源'):
        media.move_original(legacy, target)
    assert legacy.is_file() and not target.exists()


@pytest.mark.asyncio
async def test_real_privacy_code_never_retries_with_changed_references(tmp_path):
    generator = VideoGenerator(str(tmp_path), review_content=False)
    image = tmp_path / 'identity.jpg'
    image.write_bytes(b'local identity observation')
    generator.character_refs['person'] = str(image)
    failure = {'status': 'failed', 'error_type': 'privacy', 'error_locus': 'input_reference',
               'error_code': 'InputImageSensitiveContentDetected.PrivacyInformation',
               'error': "The request failed because the input image 'content[1]' may contain real person."}
    generator.api = SimpleNamespace(supports_last_frame=True, generate=AsyncMock(return_value=failure))
    shot = dict(shot_id=1, duration=6, scene_description='station',
                prompt_en='The traveler enters the station.', characters=['person'])
    result = await generator._generate_single_shot(shot, None, None,
                dict(shots=[shot], resolution='480p', aspect_ratio='16:9'))
    assert result.status == 'failed'
    assert generator.api.generate.call_count == 1
    assert generator.api.generate.call_args.kwargs['image_urls'] == [str(image)]


def test_latest_raw_failure_did_not_preserve_original_output_receipt():
    root = Path(__file__).parents[1] / 'output/20261104_030303_322135'
    if not root.exists():
        pytest.skip('original local run unavailable')
    manifest = json.loads((root / 'run_manifest.json').read_text())
    assert manifest['shots']['2']['provider_error_code'] == 'InputImageSensitiveContentDetected.PrivacyInformation'
    assert manifest['shots']['2']['attempts'] == 2
    assert not (root / 'shots/shot_001_provider_output.json').exists()


@pytest.mark.asyncio
async def test_poll_preserves_official_output_metadata():
    api = object.__new__(SeedanceAPI)
    task = SimpleNamespace(status='succeeded', model='doubao-seedance-2-0-fast-260128',
        created_at=1793718000, content=SimpleNamespace(video_url='https://example.com/original.mp4',
        last_frame_url='https://example.com/official.png', audio_url=None))
    api.ark_client = SimpleNamespace(content_generation=SimpleNamespace(
        tasks=SimpleNamespace(get=lambda **_: task)))
    result = await api.poll_task('cgt-original', timeout=1)
    assert result['provider_output']['created_at'] == task.created_at
    assert result['provider_output']['last_frame_url'] == task.content.last_frame_url
    assert result['provider_output']['model'] == task.model
