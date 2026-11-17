"""双来源连续性离线回放；测试通过不等于供应商人像审核通过。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from pipeline import provider_materials as media
from test_provider_materials import original  # noqa: F401 -- 复用原始产物测试夹具
from test_seedance_api_resume import api_with_fake_tasks


FAILURE = json.loads((Path(__file__).parent / 'fixtures/real_original_video_privacy_failure.json').read_text())


class RecordedPrivacyRejection(Exception):
    """只回放已记录的 400，不调用供应商审核。"""
    status_code = FAILURE['http_status']
    request_id = FAILURE['request_id']
    body = {'error': FAILURE['error']}
    code = FAILURE['error']['code']
    message = FAILURE['error']['message']


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('offline replay must never use network')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


@pytest.mark.asyncio
@pytest.mark.parametrize('characters,action', [
    (('dancer', 'partner'), 'Both dancers turn together, then separate.'),
    (('rescuer', 'hiker'), 'The rescuer secures the rope while the hiker reaches the ledge.'),
    (('mechanic', 'driver'), 'The mechanic hands over the key and the driver opens the car.'),
])
@pytest.mark.parametrize('state_first', [True, False])
async def test_dual_source_identity_and_state_survive_sdk_refusal(original, characters, action, state_first):
    identities = [original(character=name)[0] for name in characters]
    old_tail, _ = original(kind='selected_state', character=None, timestamp=5.95, box=None)
    new_tail, _ = original(2, kind='selected_state', character=None, timestamp=5.95, box=None)
    api, tasks = api_with_fake_tasks()
    submitted = []
    acknowledged = []

    def create(**request):
        submitted.append(deepcopy(request))
        if len(submitted) == 2:
            raise RecordedPrivacyRejection()
        return SimpleNamespace(id='offline-accepted')

    tasks.create = create
    for number, tail in enumerate((old_tail, new_tail)):
        refs = [tail, *identities] if state_first else [*identities, tail]
        plan = media.material_plan(refs, None, live=True)
        before = deepcopy(plan)
        result = await api.generate(
            prompt=media.material_prompt('Images 1-3. ' + action, plan),
            image_urls=refs, image_role=None, provider_materials=plan,
            model=FAILURE['model'], timeout=1, on_submitted=acknowledged.append)
        assert plan == before
        assert submitted[number]['content'][1:] == plan['content']
        assert len(plan['content']) == number + 1
        assert len(plan['bindings']) == 3
        assert plan['video_duration'] <= 15
        assert all(item['role'] == 'reference_video' for item in plan['content'])
        assert {binding['source_task_id'] for binding in plan['bindings']} == (
            {'offline-1'} if number == 0 else {'offline-1', 'offline-2'})
        assert action in submitted[number]['content'][0]['text']
        assert all(name in submitted[number]['content'][0]['text'] for name in characters)
        assert 'data:image' not in json.dumps(submitted[number])
        assert result['status'] == ('succeeded' if number == 0 else 'failed')
    # 已经受理过的同一源 URL 仍在被拒请求中；不把批量拒绝归因给任一单独来源。
    assert submitted[0]['content'][1] in submitted[1]['content'][1:]
    assert result['error_type'] == 'privacy'
    assert result['error_code'] == FAILURE['error']['code']
    assert result['submission_diagnostic']['request_id'] == FAILURE['body_request_id']
    assert result['submission_diagnostic']['sdk_request_id'] == FAILURE['request_id']
    assert len(submitted) == 2
    assert acknowledged == ['offline-accepted']
    assert tasks.polled == ['offline-accepted']


@pytest.mark.asyncio
async def test_actual_requests_match_original_receipts_through_sdk_boundary(monkeypatch):
    root = Path(__file__).parents[1] / 'output' / FAILURE['run']
    if not root.is_dir():
        pytest.skip('private original run unavailable; portable cross-scene replay runs separately')
    paths = [root / 'run_manifest.json']
    paths += sorted((root / 'submission_requests').glob('*.json'))
    paths += sorted((root / 'submission_results').glob('*.json'))
    paths += sorted((root / 'shots').glob('*.mp4'))
    paths += sorted((root / 'shots').glob('*_provider_output.json'))
    before = {str(p): media._hash(p) for p in paths}
    receipts = []
    for number in (1, 2):
        video = root / f'shots/shot_{number:03d}.mp4'
        assert media._hash(video) == FAILURE['source_hashes'][number - 1]
        receipts.append(media._read(media.receipt_path(video)))
    # 重放提交当时的有效期；不刷新地址、篡改回执或在云端重新提交。
    monkeypatch.setattr(media.time, 'time', lambda: max(p['created_at'] for p in receipts) + 600)
    identities = [str(root / 'character_refs/Aunt_Lin_309d18096e26.jpg'),
                  str(root / 'character_refs/An.jpg')]
    api, tasks = api_with_fake_tasks()
    submitted = []

    def create(**request):
        # 仅摘要进入断言，pytest 失败时也不暴露真实签名地址。
        submitted.append({
            'content_hash': hashlib.sha256(media._canonical(request['content'][1:])).hexdigest(),
            'prompt_hash': hashlib.sha256(request['content'][0]['text'].encode()).hexdigest(),
            'options': {k: v for k, v in request.items() if k != 'content'},
        })
        if len(submitted) == 2:
            raise RecordedPrivacyRejection()
        return SimpleNamespace(id='offline-replay-accepted')

    tasks.create = create
    for shot in (2, 3):
        recorded = media._read(root / f'submission_requests/shot_{shot:03d}_attempt_001.json')
        refs = [str(root / f'shots/shot_{shot-1:03d}_lastframe.jpg'), *identities]
        plan = media.material_plan(refs, recorded['provenance']['image_role'], live=True)
        assert media.plan_evidence(plan) == recorded['provenance']['provider_materials']
        assert plan['video_duration'] == pytest.approx(6.041667 * (shot - 1))
        assert [b['source_task_id'] for b in plan['bindings']] == [
            receipts[shot-2]['task_id'], receipts[0]['task_id'], receipts[0]['task_id']]
        options = {k: v for k, v in recorded.items() if k not in ('prompt', 'provenance')}
        result = await api.generate(prompt=recorded['prompt'], **options, image_urls=refs,
            image_role=recorded['provenance']['image_role'], provider_materials=plan, timeout=1)
        assert submitted[-1]['content_hash'] == FAILURE['content_hashes'][str(shot)]
        assert submitted[-1]['prompt_hash'] == hashlib.sha256(recorded['prompt'].encode()).hexdigest()
        assert submitted[-1]['options'] == {**options, 'watermark': False}
        assert result['status'] == ('succeeded' if shot == 2 else 'failed')
    manifest = json.loads((root / 'run_manifest.json').read_text())
    assert manifest['shots']['3']['provider_error_code'] == result['error_code'] == FAILURE['error']['code']
    assert result['error_type'] == 'privacy'
    assert len(submitted) == 2 and tasks.polled == ['offline-replay-accepted']
    assert before == {str(p): media._hash(p) for p in paths}


@pytest.mark.asyncio
async def test_real_sdk_serializes_recorded_dual_video_request_unchanged(monkeypatch):
    import httpx
    from volcenginesdkarkruntime import Ark
    import config
    from tools.seedance_api import SeedanceAPI

    root = Path(__file__).parents[1] / 'output' / FAILURE['run']
    if not root.is_dir():
        pytest.skip('private original run unavailable')
    recorded = media._read(root / 'submission_requests/shot_003_attempt_001.json')
    source = media._read(root / 'shots/shot_002_provider_output.json')
    monkeypatch.setattr(media.time, 'time', lambda: source['created_at'] + 600)
    refs = [str(root / name) for name in ['shots/shot_002_lastframe.jpg',
        'character_refs/Aunt_Lin_309d18096e26.jpg', 'character_refs/An.jpg']]
    plan = media.material_plan(refs, recorded['provenance']['image_role'], live=True)
    sent = []

    def transport(request):
        payload = json.loads(request.content)
        sent.append({
            'method': request.method, 'path': request.url.path,
            'content_hash': hashlib.sha256(media._canonical(payload['content'][1:])).hexdigest(),
            'prompt_hash': hashlib.sha256(payload['content'][0]['text'].encode()).hexdigest(),
            'options': {k: v for k, v in payload.items() if k != 'content'},
        })
        return httpx.Response(400, json={'error': FAILURE['error']},
                              headers={'x-request-id': FAILURE['request_id']})

    api = object.__new__(SeedanceAPI)
    # 真实 SDK 序列化 + 内存 HTTP 替身；不使用真实凭据或网络。
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        api.ark_client = Ark(api_key='offline-serialization-only', max_retries=0,
                            base_url=config.ARK_BASE_URL_SEEDANCE, http_client=client)
        options = {k: v for k, v in recorded.items() if k not in ('prompt', 'provenance')}
        result = await api.generate(prompt=recorded['prompt'], **options, image_urls=refs,
            image_role=recorded['provenance']['image_role'], provider_materials=plan, timeout=1)
    assert sent == [{
        'method': 'POST', 'path': '/api/plan/v3/contents/generations/tasks',
        'content_hash': FAILURE['content_hashes']['3'],
        'prompt_hash': hashlib.sha256(recorded['prompt'].encode()).hexdigest(),
        'options': {**options, 'watermark': False, **dict.fromkeys([
            'safety_identifier', 'callback_url', 'service_tier', 'execution_expires_after',
            'priority', 'draft', 'camera_fixed', 'seed', 'frames', 'tools'])},
    }]
    assert result['error_code'] == FAILURE['error']['code']
    assert result['error_type'] == 'privacy'
    assert result['submission_diagnostic']['http_status'] == 400
    assert result['submission_diagnostic']['request_id'] == FAILURE['request_id']
