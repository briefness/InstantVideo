"""独立模型请求与最终采用的观察来源分别绑定；离线替身不证明画质。"""
from copy import deepcopy
import json

import pytest

from test_contribution_witness import frames, opposite_motion, offline
from test_director_intent import Responses
from test_support_evidence_boundary import scene
from pipeline.observation_support import apply_support, support_fingerprint, support_input, verify_support
from pipeline.support_witness import CURRENT_POLICY


def inputs(requirement='Courier hands the parcel to the recipient.'):
    catalog, raw = scene(requirement)
    catalog['evidence_policy'] = CURRENT_POLICY
    return catalog, raw


def response():
    return {'decisions': {'g': {'requirement': 'fulfilled', 'issue': 'none',
        'finding': 'The required transfer is visibly completed.',
        'evidence': ['current_clip:1', 'current_clip:2']}}}


def arguments(tmp_path, frames, client):
    return dict(client=client, model='offline', cache_dir=tmp_path,
        video_hash=frames['manifest']['sources'][0]['video_sha256'], visual_evidence=frames)


@pytest.mark.parametrize('requirement', [
    'Courier hands the parcel to the recipient.',
    'Rescuer brings the swimmer onto the river bank.',
    'Two dancers exchange positions on the stage.',
])
@pytest.mark.parametrize('field', ['terminal', 'scene', 'camera', 'samples', 'geometry', 'events'])
def test_blind_request_cannot_rebind_proof_to_changed_delivery_facts(tmp_path, frames, requirement, field):
    catalog, raw = inputs(requirement)
    client = Responses(response())
    args = arguments(tmp_path, frames, client)
    checked, proof = verify_support(catalog, raw, **args)
    original = deepcopy(raw)
    assert checked['terminal_observation'] == original['terminal_observation']
    assert proof['source_observation_hash'] == support_fingerprint(original)
    if field == 'terminal':
        raw['terminal_observation']['prop_state'] = 'The source still retains the transferred object.'
    elif field == 'scene':
        raw['end_state_context']['location'] = 'unrelated airport'
    elif field == 'camera':
        raw['end_state_context']['camera'] = 'different framing'
    elif field == 'samples':
        raw['observations']['g']['sample_indices'] = [1]
    elif field == 'geometry':
        raw['action_geometry_valid'] = False
    else:
        raw['action_event_evidence'] = [{'event_id': 1, 'causal_order_valid': False}]
    # 模型输入继续独立；来源变化由程序拒绝，不能补发一次模型调用掩盖它。
    assert support_input(catalog, raw) == support_input(catalog, original)
    with pytest.raises(ValueError, match='原始观察来源不一致'):
        apply_support(catalog, raw, proof, visual_manifest=frames['manifest'])
    with pytest.raises(ValueError, match='原始观察来源不一致'):
        verify_support(catalog, raw, **args)
    assert len(client.calls) == 1


def test_request_remains_blind_and_unchanged_source_resumes_without_call(tmp_path, frames):
    catalog, raw = inputs()
    client = Responses(response())
    args = arguments(tmp_path, frames, client)
    result, proof = verify_support(catalog, raw, **args)
    assert verify_support(catalog, deepcopy(raw), **args) == (result, proof)
    payload = json.loads(client.calls[0]['messages'][-1]['content'])
    assert 'observed_context' not in payload and 'source_observation_hash' not in payload
    assert set(payload['cases']['g']) == {'requirement', 'requirement_context'}
    saved = json.loads(next((tmp_path/'support').glob('*.json')).read_text())
    assert saved['source_observation_hash'] == support_fingerprint(raw)
    assert len(client.calls) == 1


def test_unbound_v2_record_is_not_silently_upgraded_or_requeried(tmp_path, frames):
    catalog, raw = inputs()
    client = Responses(response())
    args = arguments(tmp_path, frames, client)
    _, proof = verify_support(catalog, raw, **args)
    del proof['source_observation_hash']
    with pytest.raises(ValueError, match='证据格式无效'):
        apply_support(catalog, raw, proof, visual_manifest=frames['manifest'])
    # 模拟补丁前保存的证明；保持原始独立输入与响应，不伪造新的来源绑定。
    path = next((tmp_path/'support').glob('*.json'))
    saved = json.loads(path.read_text())
    saved['proof'] = proof
    saved.pop('source_observation_hash')
    from pipeline.run_state import RunWorkspace
    RunWorkspace._atomic_write_json(path, saved)
    with pytest.raises(ValueError, match='证据格式无效'):
        verify_support(catalog, raw, **args)
    assert len(client.calls) == 1
