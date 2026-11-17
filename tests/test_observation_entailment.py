"""真实矛盾观察不可变回放；离线标签不冒充新视觉判断。"""
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.result_observations import decode_observations


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def captured():
    root = Path(__file__).parents[1] / 'output/20261104_141157_732393'
    if not root.is_dir():
        pytest.skip('original local run unavailable')
    digest = '059f89bb7b9ef4d2c0dbe582544ab9476a520c636d7d375a301e87d2589e0001'
    assert hashlib.sha256((root / 'shots/shot_001.mp4').read_bytes()).hexdigest() == digest
    record = json.loads((root / f'semantic_reviews/{digest}.json').read_text())
    return record


def test_recorded_description_and_success_are_inconsistent_but_legacy_decoder_accepts():
    record = captured()
    value = record['observation_response']
    key = '/intent/completion_goals/1@0:28'
    assert value['observations'][key]['status'] == 'satisfied'
    assert '椅面上' in value['observations'][key]['observation']
    assert '桌边的空椅旁' in record['action_contract']['review_projection']['completion_contract']['criteria'][key]['action']
    decoded = decode_observations(value,
        record['action_contract']['review_projection']['completion_contract'],
        record['observation_schema'], len(record['sample_timestamps']))
    assert decoded['accepted'] is True  # 原始缺陷：只看status，未检查事实是否支持要求。


def test_real_positive_claim_is_rejected_by_independent_supported_evidence(tmp_path):
    from pipeline.observation_support import verify_support
    from test_director_intent import Responses
    record = captured()
    catalog = record['action_contract']['review_projection']['completion_contract']
    raw = record['observation_response']
    rows = {key: {'verdict': 'supported', 'requirement_quote': catalog['criteria'][key]['action'],
                  'observation_quote': value['observation'], 'reason': 'Explicit offline annotation.'}
            for key, value in raw['observations'].items()}
    key = '/intent/completion_goals/1@0:28'
    rows[key].update(verdict='contradicted', reason='The observed chair seat is not the required table edge.')
    client = Responses({'decisions': {k: row['verdict'] for k, row in rows.items()}})
    result, proof = verify_support(catalog, raw, client=client, model='offline',
        cache_dir=tmp_path, video_hash=record['video_hash'])
    assert result['observations'][key]['status'] == 'violated'
    assert raw['observations'][key]['status'] == 'satisfied'
    payload = json.loads(client.calls[0]['messages'][-1]['content'])
    assert 'status' not in json.dumps(payload)
    assert proof['decisions'][key]['verdict'] == 'contradicted'
    assert len(client.calls) == 1
    assert verify_support(catalog, raw, client=client, model='offline',
        cache_dir=tmp_path, video_hash=record['video_hash']) == (result, proof)
    assert len(client.calls) == 1


@pytest.mark.parametrize('requirement,observation,verdict', [
    ('Courier hands parcel to recipient A.', 'Courier gives parcel to B.', 'contradicted'),
    ('Rescuer pulls the swimmer onto the bank.', 'Swimmer remains in the water.', 'contradicted'),
    ('Dancer turns clockwise.', 'Dancer turns anticlockwise.', 'contradicted'),
    ('Open valve after sealing container.', 'Valve opens before container is sealed.', 'contradicted'),
    ('Two actors enter.', 'Only one actor enters.', 'contradicted'),
    ('Courier delivers the parcel to A.', 'A takes possession of the package.', 'supported'),
    ('Performer sits down.', 'Performer sits, then stands again.', 'supported'),
])
def test_cross_scene_independent_annotations(requirement, observation, verdict):
    from pipeline.observation_support import apply_support
    catalog = {'criteria': {'goal': {'kind': 'completion', 'action': requirement}}}
    raw = {'observations': {'goal': {'status': 'satisfied', 'observation': observation, 'sample_indices': [1]}}}
    proof = {'decisions': {'goal': {'verdict': verdict, 'requirement_quote': requirement,
        'observation_quote': observation, 'reason': 'Independent offline test label, not model accuracy evidence.'}}}
    result = apply_support(catalog, raw, proof)
    assert result['observations']['goal']['status'] == ('satisfied' if verdict == 'supported' else 'violated')
    assert raw['observations']['goal']['status'] == 'satisfied'


@pytest.mark.parametrize('fault', ['unknown', 'invented_quote', 'missing', 'extra', 'negative_overwrite'])
def test_unusable_support_never_upgrades_acceptance(fault):
    from pipeline.observation_support import apply_support
    catalog = {'criteria': {'goal': {'kind': 'completion', 'action': 'A receives the parcel.'}}}
    raw = {'observations': {'goal': {'status': 'satisfied', 'observation': 'A holds the parcel.', 'sample_indices': [1]}}}
    row = {'verdict': 'supported', 'requirement_quote': 'receives the parcel',
           'observation_quote': 'holds the parcel', 'reason': 'offline annotation'}
    proof = {'decisions': {'goal': row}}
    if fault == 'unknown':
        row['verdict'] = 'unknown'
    elif fault == 'invented_quote':
        row['observation_quote'] = 'B holds the parcel'
    elif fault == 'missing':
        proof['decisions'].clear()
    elif fault == 'extra':
        proof['decisions']['foreign'] = row
    else:
        raw['observations']['goal']['status'] = 'violated'
    with pytest.raises(ValueError):
        apply_support(catalog, raw, proof)


def test_failed_support_call_is_not_retried(tmp_path):
    from pipeline.observation_support import verify_support
    from test_director_intent import Responses
    catalog = {'criteria': {'goal': {'kind': 'completion', 'action': 'A receives parcel.'}}}
    raw = {'observations': {'goal': {'status': 'satisfied', 'observation': 'A holds parcel.', 'sample_indices': [1]}}}
    client = Responses({'decisions': {}})
    for _ in range(2):
        with pytest.raises(ValueError):
            verify_support(catalog, raw, client=client, model='offline', cache_dir=tmp_path, video_hash='offline')
    assert len(client.calls) == 1


def test_original_video_formal_review_rejects_contradiction_and_reuses_evidence(tmp_path):
    from types import SimpleNamespace
    from copy import deepcopy
    from pipeline.semantic_review import SemanticTakeReviewer
    from test_full_production_flow import completion
    from functional_observation_cases import annotated_support_response
    record = captured()
    root = Path(__file__).parents[1] / 'output/20261104_141157_732393'
    story = json.loads((root / 'result_admission.json').read_text())['execution_storyboard']
    calls = []
    def create(**request):
        calls.append(request)
        if isinstance(request['messages'][-1]['content'], list):
            return completion(deepcopy(record['observation_response']))
        proof = annotated_support_response(request)
        proof['decisions']['/intent/completion_goals/1@0:28'] = 'contradicted'
        return completion(proof)
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-original',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    for _ in range(2):
        result = reviewer.review(str(root / 'shots/shot_001.mp4'), story['shots'][0],
                                 identity_crop_entities=['XiaoAn', 'LinAyi'])
        assert not result.accepted and not result.primary_action_completed
    assert len(calls) == 2
    saved = json.loads((reviewer.cache_dir / f"{record['video_hash']}.json").read_text())
    assert saved['observation_response'] == record['observation_response']
    assert saved['support_evidence']['decisions']['/intent/completion_goals/1@0:28']['verdict'] == 'contradicted'
