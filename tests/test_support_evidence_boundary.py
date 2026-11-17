"""证据传递不依赖题材；离线回放不冒充新的模型判断。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.observation_support import support_input, verify_support, apply_support


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    def forbidden(*_, **__):
        pytest.fail('offline evidence replay only')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)


def captured():
    root = Path(__file__).parents[1] / 'output/20261109_193601_192365'
    support = json.loads(next((root / 'semantic_reviews/support').glob('*.json')).read_text())
    record = json.loads(next((root / 'semantic_reviews/responses').glob('*/*.json')).read_text())
    raw = json.loads(record['attempts'][0]['choices'][0]['content'])
    from pipeline.causality import compile_action_contract
    story = json.loads((root / 'result_admission.json').read_text())['execution_storyboard']
    catalog = compile_action_contract(story['shots'][0]).review_projection['completion_contract']
    assert hashlib.sha256((root / 'shots/shot_001.mp4').read_bytes()).hexdigest() == record['video_hash']
    return root, support, catalog, raw


def scene(requirement='Worker hands package to recipient.', observation='Recipient receives package.'):
    return ({'criteria': {'g': {'kind': 'completion', 'action': requirement}}}, {
        'observations': {'g': {'status': 'satisfied', 'observation': observation, 'sample_indices': [1, 2]}},
        'end_state_context': {'location': 'loading bay', 'subject': 'worker and recipient',
            'camera': 'wide shot', 'screen_direction': 'worker left', 'lighting': 'daylight'},
        'terminal_observation': {'sample_index': 2, 'status': 'observed', 'action_phase': 'aftermath',
            'pose_and_gaze': 'Both stand.', 'prop_state': 'Recipient holds package.', 'open_motion': ''},
    })


def test_original_observed_context_is_not_lost_or_promoted_to_whole_clip():
    _, saved, catalog, raw = captured()
    before = deepcopy(raw)
    assert '雨夜' not in json.dumps([c['observation'] for c in saved['input']['payload']['cases'].values()], ensure_ascii=False)
    assert '雨夜' in raw['end_state_context']['lighting']
    payload = support_input(catalog, raw)
    context = payload['observed_context']
    assert context['ending_scene'] == {'source': '/end_state_context', 'scope': 'ending_context',
                                      'facts': raw['end_state_context']}
    assert context['ending_snapshot'] == {'source': '/terminal_observation', 'scope': 'ending_sample',
        'sample_indices': [9], 'facts': {k: v for k, v in raw['terminal_observation'].items()
                                       if k not in ('status', 'sample_index')}}
    assert payload['cases'] == saved['input']['payload']['cases']  # 要求、动作、范围均未改。
    assert 'status' not in json.dumps(payload)
    assert raw == before


@pytest.mark.parametrize('requirement,location,lighting,observation', [
    ('Courier delivers at a station at night.', 'railway platform', 'night lamps', 'Recipient takes parcel.'),
    ('Rescuer brings swimmer onto the river bank at dawn.', 'river bank', 'dawn', 'Swimmer reaches land.'),
    ('Dancers exchange places on stage under a red spotlight.', 'stage', 'red spotlight', 'Dancers exchange places.'),
    ('Technician closes the cabinet in a factory.', 'factory', 'daylight', 'Technician closes the cabinet.'),
])
def test_cross_scene_evidence_reaches_formal_support_without_planned_facts(tmp_path, requirement, location, lighting, observation):
    from test_director_intent import Responses
    catalog, raw = scene(requirement, observation)
    raw['end_state_context'].update(location=location, lighting=lighting)
    raw['accepted'] = True
    raw['expected_state'] = {'lighting': 'invented planned light'}
    client = Responses({'decisions': {'g': 'supported'}})
    result, proof = verify_support(catalog, raw, client=client, model='offline', cache_dir=tmp_path, video_hash='fixture')
    payload = json.loads(client.calls[0]['messages'][-1]['content'])
    assert payload['observed_context']['ending_scene']['facts'] == raw['end_state_context']
    assert 'invented planned light' not in json.dumps(payload)
    assert 'accepted' not in payload and 'status' not in json.dumps(payload)
    assert result == raw
    assert proof['protocol'] == payload['protocol']
    assert apply_support(catalog, raw, proof) == raw


@pytest.mark.parametrize('change', ['scene', 'snapshot', 'samples', 'requirement', 'missing_context'])
def test_saved_decision_cannot_be_rebound_to_other_evidence(tmp_path, change):
    from test_director_intent import Responses
    catalog, raw = scene()
    client = Responses({'decisions': {'g': 'supported'}})
    _, proof = verify_support(catalog, raw, client=client, model='offline', cache_dir=tmp_path, video_hash='fixture')
    if change == 'scene': raw['end_state_context']['location'] = 'other room'
    if change == 'snapshot': raw['terminal_observation']['prop_state'] = 'Worker still holds package.'
    if change == 'samples': raw['observations']['g']['sample_indices'] = [1]
    if change == 'requirement': catalog['criteria']['g']['action'] += ' Then sign.'
    if change == 'missing_context': del raw['end_state_context']
    with pytest.raises(ValueError, match='输入绑定'):
        apply_support(catalog, raw, proof)
    assert len(client.calls) == 1


@pytest.mark.parametrize('fault', ['scene_type', 'foreign_field', 'snapshot_index', 'unknown_snapshot', 'snapshot_fact'])
def test_malformed_context_stops_before_any_model_call(tmp_path, fault):
    catalog, raw = scene()
    if fault == 'scene_type': raw['end_state_context'] = []
    if fault == 'foreign_field': raw['end_state_context']['expected'] = 'planned result'
    if fault == 'snapshot_index': raw['terminal_observation']['sample_index'] = True
    if fault == 'unknown_snapshot': raw['terminal_observation']['status'] = 'unknown'
    if fault == 'snapshot_fact': raw['terminal_observation']['prop_state'] = {'expected': 'delivered'}
    with pytest.raises(ValueError):
        verify_support(catalog, raw, client=None, model='offline', cache_dir=tmp_path, video_hash='fixture')
    assert not (tmp_path / 'support').exists()


def test_projection_does_not_alias_source_evidence():
    catalog, raw = scene()
    original = deepcopy(raw)
    payload = support_input(catalog, raw)
    payload['cases']['g']['sample_indices'].append(3)
    payload['observed_context']['ending_scene']['facts']['lighting'] = 'changed'
    assert raw == original


def test_latest_unknown_stays_unknown_through_formal_review_and_resume(tmp_path):
    from pipeline.semantic_review import SemanticTakeReviewer, SemanticReviewUnavailableError
    from test_full_production_flow import completion
    root, saved, _, raw = captured()
    story = json.loads((root / 'result_admission.json').read_text())['execution_storyboard']
    calls = []
    def create(**request):
        calls.append(request)
        content = request['messages'][-1]['content']
        if isinstance(content, list):
            return completion(deepcopy(raw))
        payload = json.loads(content)
        assert payload['observed_context']['ending_scene']['facts'] == raw['end_state_context']
        return completion(json.loads(saved['response']['choices'][0]['content']))
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-latest',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    for _ in range(2):
        with pytest.raises(SemanticReviewUnavailableError, match='语义不确定'):
            reviewer.review(str(root / 'shots/shot_001.mp4'), story['shots'][0],
                            identity_crop_entities=['XiaoAn', 'AuntLin'])
    assert len(calls) == 2
    evidence = json.loads(next(reviewer.cache_dir.glob('*.evidence.json')).read_text())
    assert evidence['failure_kind'] == 'semantic_uncertainty'
    assert evidence['response'] == raw
