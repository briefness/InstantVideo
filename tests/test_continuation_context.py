"""续拍传输必须保留全部来源；离线响应不证明视频质量。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_full_production_flow import completion
from test_goal_allocation import offline

ROOT = Path(__file__).parents[1] / 'output/20261117_142918_048411'


def captured():
    story = json.loads((ROOT / 'result_admission.json').read_text())['execution_storyboard']
    manifest = json.loads((ROOT / 'run_manifest.json').read_text())
    previous = {**story['shots'][0], 'observed_end_state': manifest['shots']['1']['observed_end_state'],
                '_accepted_video_path': str(ROOT / 'shots/shot_001.mp4')}
    return story, previous, str(ROOT / 'shots/shot_001_lastframe.jpg')


def planner(calls):
    def create(**request):
        calls.append(request)
        context = json.loads(request['messages'][-1]['content'])
        return completion({'compatible': True, 'reason': 'Explicit offline compatibility fixture.',
            'entry_action': None, 'view_window': None,
            'observed_motion': {'summary': 'Explicit offline motion annotation.',
                'sample_indices': [1, len(context['temporal_evidence']['sources'][0]['frames'])]},
            'spatial_guidance': dict.fromkeys(context['spatial_context']['entities'])})
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_real_failure_reaches_request_without_losing_source_or_changing_actions(tmp_path):
    from pipeline.continuation import prepare_continuation
    story, previous, tail = captured()
    shot = story['shots'][1]
    original = deepcopy(shot)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        ROOT / 'result_admission.json', ROOT / 'run_manifest.json', ROOT / 'shots/shot_001.mp4')}
    calls = []
    client = planner(calls)
    result = prepare_continuation(shot, previous, tail, tmp_path, client=client)
    assert len(calls) == 1
    assert len(calls[0]['messages'][-1]['content']) <= 64000
    assert result['action_beats'] == original['action_beats'] and shot == original
    record = json.loads((tmp_path / 'continuations/shot_002.json').read_text())
    from pipeline.continuation_context import expand_context
    assert expand_context(json.loads(calls[0]['messages'][-1]['content'])) == record['input']
    assert prepare_continuation(shot, previous, tail, tmp_path, client=client, allow_create=False) == result
    assert len(calls) == 1
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in before.items())


@pytest.mark.parametrize('actions', [
    [('DancerA', 'DancerB', 'Turn together while maintaining hand contact'),
     ('DancerB', 'DancerA', 'Carry shared momentum into the next travelling step')],
    [('Robot', 'Panel', 'Align panel with the fixture'),
     ('Robot', 'Panel', 'Seat the aligned panel in the fixture')],
    [('Rescuer', 'Patient', 'Secure the harness before lifting'),
     ('Rescuer', 'Patient', 'Lift along the anchored guide into the safe zone')],
])
def test_shared_sources_scale_with_unique_evidence_and_preserve_every_relationship(actions):
    from pipeline.continuation_context import VERSION, request_context, expand_context, serialized
    units = [dict(shot_id=i+1, beat_index=0, actor=a, target=t, action=text)
             for i, (a, t, text) in enumerate(actions)]
    def source(n):
        return {'request_encoding': VERSION, 'planned_shot': {'action_beats': [
            {'completion_intent': {'kind': 'allocated', 'requirements': [
                {'reference': f'goal:{i}', 'start': 0, 'end': 10, 'scope': deepcopy(units)}]}}
            for i in range(n)]}, 'completion_contract': {'criteria': {
                'joint': {'execution_scope': {'shot_id': 2, 'units': deepcopy(units)}}}},
            'spatial_context': {'work': [{'requirements': [{'scope': deepcopy(units)}]}]}}
    small, large = source(1), source(100)
    before = deepcopy(large)
    wire = request_context(large)
    assert expand_context(wire) == large == before
    assert len(wire['source_catalog']['units']) == 2
    assert len(wire['source_catalog']['scopes']) == 1
    assert wire['source_catalog'] == request_context(small)['source_catalog']
    for _, _, text in actions:
        assert serialized(wire).count(text) == 1
    assert len(serialized(wire)) < len(serialized(large))


def scoped_context():
    from pipeline.continuation_context import VERSION
    units = [dict(shot_id=i+1, beat_index=0, actor='A', target='B', action=f'unchanged work {i}')
             for i in range(2)]
    return {'request_encoding': VERSION, 'first': {'scope': units},
            'second': {'execution_scope': {'shot_id': 2, 'units': deepcopy(units)}}}


@pytest.mark.parametrize('fault', ['conflicting_text', 'duplicate', 'empty', 'reserved', 'version', 'nonfinite'])
def test_invalid_source_never_becomes_a_silent_summary(fault):
    from pipeline.continuation_context import request_context, context_diagnostics
    source = scoped_context()
    if fault == 'conflicting_text':
        source['second']['execution_scope']['units'][0]['action'] = 'different source'
    elif fault == 'duplicate':
        source['first']['scope'].append(deepcopy(source['first']['scope'][0]))
    elif fault == 'empty':
        source['first']['scope'] = []
    elif fault == 'reserved':
        source['first'] = {'scope_ref': 'unbound'}
    elif fault == 'version':
        source['request_encoding'] = 'unrecognized'
    else:
        source['untrusted_number'] = float('nan')
    with pytest.raises(ValueError):
        context_diagnostics(source, request_context(source))


@pytest.mark.parametrize('fault', ['missing_unit', 'missing_scope', 'wrong_identity', 'duplicate', 'unused', 'extra'])
def test_corrupt_directory_cannot_be_expanded_as_complete_evidence(fault):
    from pipeline.continuation_context import request_context, expand_context
    wire = request_context(scoped_context())
    cat = wire['source_catalog']
    if fault == 'missing_unit':
        cat['units'].pop('1:0')
    elif fault == 'missing_scope':
        cat['scopes'].clear()
    elif fault == 'wrong_identity':
        cat['units']['1:0']['shot_id'] = 3
    elif fault == 'duplicate':
        cat['scopes']['scope:1'].append('1:0')
    elif fault == 'unused':
        cat['scopes']['unused'] = ['1:0']
    else:
        wire['first']['scope']['untrusted'] = True
    with pytest.raises(ValueError):
        expand_context(wire)


def test_legacy_wire_format_is_not_silently_migrated():
    from pipeline.continuation_context import request_context, expand_context
    source = scoped_context()
    source.pop('request_encoding')
    assert request_context(source) == expand_context(source) == source
    assert 'source_catalog' not in request_context(source)


def test_real_unique_capacity_failure_records_sizes_without_calling_or_discarding(tmp_path):
    from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
    story, previous, tail = captured()
    shot = story['shots'][1]
    # 独立大字段不能伪装成可去重的来源，守卫仍须停止而不是裁剪。
    shot['unrelated_unique_direction'] = 'x' * 64001
    before = deepcopy(shot)
    calls = []
    with pytest.raises(ContinuationUnavailableError, match='容量不足.*未调用模型'):
        prepare_continuation(shot, previous, tail, tmp_path, client=planner(calls))
    assert not calls and shot == before
    record = json.loads((tmp_path / 'continuations/context_diagnostics/shot_002.json').read_text())
    assert record['request_chars'] > record['limit_chars'] == 64000
    assert record['unique_units'] == 15 and record['unique_scopes'] == 2
    assert 'x' * 100 not in json.dumps(record)
    assert not (tmp_path / 'continuations/shot_002.json').exists()


def test_new_cache_binds_nonlocal_evidence_and_rejects_changed_transport(tmp_path):
    from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
    story, previous, tail = captured()
    calls = []
    shot = story['shots'][1]
    client = planner(calls)
    prepare_continuation(shot, previous, tail, tmp_path, client=client)
    # 所有副本保持一致地改变另一个镜头的原文，也不能借旧指纹通过。
    for beat in shot['action_beats']:
        for requirement in beat['completion_intent']['requirements']:
            for unit in requirement.get('scope', []):
                if unit['shot_id'] != shot['shot_id']:
                    unit['action'] += ' changed'
    with pytest.raises(ContinuationUnavailableError, match='不匹配'):
        prepare_continuation(shot, previous, tail, tmp_path, client=client, allow_create=False)
    assert len(calls) == 1
