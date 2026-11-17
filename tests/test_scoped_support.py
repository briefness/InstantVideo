"""真实跨镜目标核对回放；禁止网络，不改原片或原判定。"""
from copy import deepcopy
import json
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def captured():
    root = Path(__file__).parents[1] / 'output/20261104_230344_312370'
    support = json.loads(next((root / 'semantic_reviews/support').glob('*.json')).read_text())
    response = json.loads(next((root / 'semantic_reviews/responses').glob('*/*.json')).read_text())
    raw = json.loads(response['attempts'][0]['choices'][0]['content'])
    catalog = {'criteria': {k: row['requirement_context']
                           for k, row in support['input']['payload']['cases'].items()}}
    return root, support, catalog, raw


def test_captured_local_check_does_not_receive_future_execution_as_its_scope():
    from pipeline.observation_support import support_input
    _, _, catalog, raw = captured()
    original = deepcopy(catalog)
    payload = support_input(catalog, raw)
    for case in payload['cases'].values():
        scope = case['requirement_context'].get('execution_scope')
        if scope:
            assert all(unit['shot_id'] == scope['shot_id'] for unit in scope['units'])
    assert catalog == original


def test_captured_semantic_unknown_is_preserved_as_a_returned_judgment(tmp_path):
    from pipeline.observation_support import verify_support
    from test_director_intent import Responses
    _, support, catalog, raw = captured()
    judgment = json.loads(support['response']['choices'][0]['content'])
    client = Responses(judgment)
    with pytest.raises(ValueError, match='语义不确定'):
        verify_support(catalog, raw, client=client, model='offline', cache_dir=tmp_path, video_hash='original')
    saved = json.loads(next((tmp_path / 'support').glob('*.json')).read_text())
    assert saved['state'] == 'returned'
    assert any(row['verdict'] == 'unknown' for row in saved['proof']['decisions'].values())
    with pytest.raises(ValueError, match='语义不确定'):
        verify_support(catalog, raw, client=client, model='offline', cache_dir=tmp_path, video_hash='original')
    assert len(client.calls) == 1


def test_real_footage_formal_review_preserves_unknown_and_shared_scope(tmp_path):
    from types import SimpleNamespace
    from pipeline.semantic_review import SemanticTakeReviewer, SemanticReviewUnavailableError
    from test_full_production_flow import completion
    root, support, _, raw = captured()
    story = json.loads((root / 'result_admission.json').read_text())['execution_storyboard']
    calls, wire = [], {}
    def create(**request):
        calls.append(request)
        content = request['messages'][-1]['content']
        if isinstance(content, list):
            wire.update(json.JSONDecoder().raw_decode(content[0]['text'].split('Shot contract:\n')[1])[0])
            return completion(deepcopy(raw))
        payload = json.loads(content)
        for key, case in payload['cases'].items():
            assert case['requirement_context'] == wire['completion_contract']['criteria'][key]
        return completion(json.loads(support['response']['choices'][0]['content']))
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-real-replay',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    for _ in range(2):
        with pytest.raises(SemanticReviewUnavailableError, match='语义不确定'):
            reviewer.review(str(root / 'shots/shot_001.mp4'), story['shots'][0],
                            identity_crop_entities=['AN', 'LIN_AUNT'])
    assert len(calls) == 2  # 一次视觉回放、一次核对回放，恢复不再次调用。
    evidence = json.loads(next(reviewer.cache_dir.glob('*.evidence.json')).read_text())
    assert evidence['failure_kind'] == 'semantic_uncertainty'
    assert evidence['response'] == raw
    assert not list(reviewer.cache_dir.glob('[0-9a-f]' * 64 + '.json'))


@pytest.mark.asyncio
async def test_original_failed_run_still_stops_without_any_provider(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    from pipeline.orchestrator import VideoPipeline
    from pipeline.semantic_review import SemanticReviewUnavailableError
    root, _, _, _ = captured()
    copy = tmp_path / 'original-copy'
    shutil.copytree(root, copy)
    def forbidden(*_, **__):
        pytest.fail('recorded uncertainty does not authorize any provider call')
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=forbidden)))
    for module in ('storyboard', 'planning_review', 'continuation', 'semantic_review', 'scene_binding'):
        monkeypatch.setattr(f'pipeline.{module}.OpenAI', lambda **_: client)
    monkeypatch.setattr('pipeline.generator.SeedanceAPI', lambda: SimpleNamespace(
        generate=forbidden, query_task=forbidden, download_video=forbidden))
    with pytest.raises(SemanticReviewUnavailableError, match='语义不确定'):
        await VideoPipeline.from_workspace(str(copy)).run()
    ledger = json.loads((copy / 'run_manifest.json').read_text())['paid_take_budget']['reservations']
    assert len(ledger) == 1 and ledger[0]['provider_task_id'] == 'cgt-20261104230902-2k4mw'
    assert not (copy / 'final.mp4').exists()
    assert (copy / 'shots/shot_001.mp4').read_bytes() == (root / 'shots/shot_001.mp4').read_bytes()


@pytest.mark.parametrize('goal,first,last', [
    ('Deliver a parcel and obtain the signature.', 'Courier hands parcel to receiver.', 'Receiver signs.'),
    ('Disconnect power then replace the fuse.', 'Technician disconnects power.', 'Technician replaces the fuse.'),
    ('Exchange places and finish together.', 'Dancers exchange places.', 'Dancers finish together.'),
    ('Show the lamp profile then the glowing shade.', 'Lamp rotates to profile.', 'Lamp shade glows.'),
])
@pytest.mark.parametrize('verdict', ['supported', 'contradicted', 'unknown'])
def test_cross_scene_local_verdict_does_not_complete_or_discard_later_work(tmp_path, goal, first, last, verdict):
    from pipeline.observation_support import verify_support, SupportJudgmentError
    from pipeline.assessment_scope import local_assessment
    from test_director_intent import Responses
    units = [dict(shot_id=s, beat_index=0, actor='performer', target='recipient', action=a)
             for s, a in [(8, first), (3, last)]]
    catalog = {'criteria': {'joint': {'kind': 'completion', 'action': goal, 'beat_id': 1,
        'beat_ids': [1], 'execution_scope': {'shot_id': 8, 'units': units}}}}
    raw = {'observations': {'joint': {'status': 'satisfied', 'observation': first,
                                    'sample_indices': [1]}}}
    original = deepcopy(catalog)
    client = Responses({'decisions': {'joint': verdict}})
    kwargs = dict(client=client, model='offline', cache_dir=tmp_path, video_hash='fixture')
    if verdict == 'unknown':
        with pytest.raises(SupportJudgmentError):
            verify_support(catalog, raw, **kwargs)
    else:
        result, _ = verify_support(catalog, raw, **kwargs)
        assert result['observations']['joint']['status'] == ('satisfied' if verdict == 'supported' else 'violated')
    payload = json.loads(client.calls[0]['messages'][-1]['content'])
    criterion = payload['cases']['joint']['requirement_context']
    assert criterion['kind'] == 'contribution' and 'action' not in criterion
    assert criterion['goal_context']['action'] == goal
    assert last not in json.dumps(payload)
    assert criterion['execution_scope']['units'] == units[:1]
    later = deepcopy(catalog)
    later['criteria']['joint']['execution_scope']['shot_id'] = 3
    assert local_assessment(later)['criteria']['joint']['execution_scope']['units'] == units[1:]
    assert catalog == original  # 后镜任务和全片源目标仍在冻结合同中。


@pytest.mark.parametrize('fault', ['missing_local', 'duplicate', 'wrong_beats', 'bad_identity', 'empty', 'null'])
def test_invalid_scope_never_reaches_support_model(tmp_path, fault):
    from pipeline.observation_support import verify_support
    from test_director_intent import Responses
    _, _, catalog, raw = captured()
    criterion = next(c for c in catalog['criteria'].values() if 'execution_scope' in c)
    scope = criterion['execution_scope']
    if fault == 'missing_local': scope['units'] = scope['units'][1:]
    if fault == 'duplicate': scope['units'].append(deepcopy(scope['units'][0]))
    if fault == 'wrong_beats': criterion['beat_ids'] = [99]
    if fault == 'bad_identity': scope['shot_id'] = True
    if fault == 'empty': criterion['execution_scope'] = {}
    if fault == 'null': criterion['execution_scope'] = None
    client = Responses()
    with pytest.raises(ValueError):
        verify_support(catalog, raw, client=client, model='offline', cache_dir=tmp_path, video_hash='invalid')
    assert not client.calls


@pytest.mark.parametrize('fault', ['transport', 'truncated', 'invalid', 'uncertain', 'conflict'])
def test_transport_protocol_and_semantic_failures_keep_distinct_records(tmp_path, fault):
    from types import SimpleNamespace
    from pipeline.observation_support import verify_support, SupportJudgmentError
    from test_full_production_flow import completion
    calls = []
    def create(**request):
        calls.append(request)
        if fault == 'transport':
            raise TimeoutError('offline timeout')
        response = completion({'decisions': {} if fault == 'invalid' else {
            'goal': 'unknown' if fault == 'uncertain' else 'supported'}})
        if fault == 'truncated': response.choices[0].finish_reason = 'length'
        return response
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    catalog = {'criteria': {'goal': {'kind': 'completion', 'action': 'Recipient receives parcel.'}}}
    raw = {'observations': {'goal': {'status': 'violated' if fault == 'conflict' else 'satisfied',
                                   'observation': 'Recipient holds parcel.', 'sample_indices': [1]}}}
    for model in ['offline-first', 'offline-changed']:
        with pytest.raises(ValueError) as raised:
            verify_support(catalog, raw, client=client, model=model, cache_dir=tmp_path, video_hash='same')
        assert isinstance(raised.value, SupportJudgmentError) == (fault in ['uncertain', 'conflict'])
    record = json.loads(next((tmp_path / 'support').glob('*.json')).read_text())
    assert len(calls) == 1
    assert record['state'] == ('returned' if fault in ['uncertain', 'conflict'] else 'failed')
    if fault in ['uncertain', 'conflict']:
        assert record['assessment']['code'] == ('semantic_uncertainty' if fault == 'uncertain' else 'evidence_conflict')
    elif fault == 'truncated':
        assert record['failure_code'] == 'output_limit'


def test_other_shot_realization_changes_do_not_change_the_local_question():
    from pipeline.assessment_scope import local_assessment
    _, _, catalog, _ = captured()
    original = local_assessment(catalog)
    for criterion in catalog['criteria'].values():
        scope = criterion.get('execution_scope')
        if scope:
            for unit in scope['units']:
                if unit['shot_id'] != scope['shot_id']:
                    unit['action'] = 'Different staging of another shot, not a local requirement.'
    assert local_assessment(catalog) == original
