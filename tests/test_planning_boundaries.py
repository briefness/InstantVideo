"""冻结真实漏判；接缝判定是独立标注，不冒充模型准确率。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.planning_review import ensure_planning_semantics
from pipeline.run_state import RunWorkspace
from test_director_intent import Responses


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline replay attempted network'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline replay attempted network'))


def capture():
    record = json.loads((Path(__file__).parent / 'fixtures/real_boundary_replay_failure.json').read_text())
    for source in record['sources'].values():
        assert hashlib.sha256(source['text'].encode()).hexdigest() == source['sha256']
    return tuple(json.loads(record['sources'][n]['text']) for n in
                 ('planning_compiled.json', 'planning_semantic_review.json'))


def test_original_global_approval_does_not_prove_individual_boundaries(tmp_path, monkeypatch):
    story, saved = capture()
    assert saved['verdict']['continuity_valid'] is True
    assert '接住杯子' in story['shots'][1]['end_state']['action_phase']
    assert '刚伸到杯身旁' in story['shots'][2]['start_state']['action_phase']
    client = Responses(deepcopy(saved['verdict']))
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: client)
    with pytest.raises(ValueError, match='接缝'):
        ensure_planning_semantics(story, tmp_path,
            lambda n, v: RunWorkspace._atomic_write_json(tmp_path / n, v))
    assert len(client.calls) == 1


def annotated_boundaries(story, *, rejected=None):
    from pipeline.planning_boundaries import boundary_review_context
    context = boundary_review_context(story)
    return {key: {
        'state_relation': 'contradicted' if key == rejected else 'supported',
        'action_progression': 'contradicted' if key == rejected else 'supported',
        'previous_evidence': [next(p for p in pair['previous'] if p.endswith('/end_state/action_phase'))],
        'next_evidence': [next(p for p in pair['next'] if p.endswith('/start_state/action_phase'))],
        'reason': '独立离线标注：前镜已完成接杯，后镜回到尚未接杯' if key == rejected else '独立协议测试标注',
    } for key, pair in context['boundaries'].items()}


def test_actual_draft_with_pairwise_negative_evidence_stops_and_resume_never_reasks(tmp_path, monkeypatch):
    from planning_protocol_cases import strict_intent_wire
    story, saved = capture()
    original = deepcopy(story)
    verdict = deepcopy(saved['verdict'])
    verdict.pop('continuity_valid')
    verdict['boundaries'] = annotated_boundaries(story, rejected='2->3')
    verdict = strict_intent_wire(verdict, story)  # Explicit synthetic boundary/intent labels, original capture unchanged.
    client = Responses(verdict)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: client)
    for _ in range(2):
        with pytest.raises(ValueError, match='接缝 2->3'):
            ensure_planning_semantics(story, tmp_path,
                lambda n, v: RunWorkspace._atomic_write_json(tmp_path / n, v))
    assert len(client.calls) == 1 and story == original
    evidence = json.loads((tmp_path / 'planning_semantic_evidence.json').read_text())
    edge = evidence['boundaries']['2->3']['source_fields']
    assert edge['previous']['/shots/1/end_state/action_phase'] == story['shots'][1]['end_state']['action_phase']
    assert edge['next']['/shots/2/start_state/action_phase'] == story['shots'][2]['start_state']['action_phase']


def test_old_approval_cache_is_preserved_but_cannot_authorize_new_review_contract(tmp_path, monkeypatch):
    story, saved = capture()
    RunWorkspace._atomic_write_json(tmp_path / 'planning_semantic_review.json', saved)
    before = (tmp_path / 'planning_semantic_review.json').read_bytes()
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: pytest.fail('old cache must not be retried'))
    with pytest.raises(ValueError, match='旧连续性审查缺少当前接缝证据'):
        ensure_planning_semantics(story, tmp_path, lambda *_: pytest.fail('old evidence overwritten'))
    assert (tmp_path / 'planning_semantic_review.json').read_bytes() == before


@pytest.mark.parametrize('cache', [[], True, 'invalid'])
def test_invalid_cache_shape_stops_before_protocol_access_or_model_call(tmp_path, monkeypatch, cache):
    story, _ = capture()
    (tmp_path / 'planning_semantic_review.json').write_text(json.dumps(cache))
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: pytest.fail('invalid cache triggered a call'))
    with pytest.raises(ValueError, match='缓存协议无效'):
        ensure_planning_semantics(story, tmp_path, lambda *_: pytest.fail('invalid cache overwritten'))


@pytest.mark.parametrize('relation', ['supported', 'contradicted', 'undetermined'])
@pytest.mark.parametrize('field', ['state_relation', 'action_progression'])
def test_both_dimensions_of_every_edge_are_required(relation, field):
    from pipeline.planning_boundaries import boundary_review_context, decode_boundary_review, boundary_response_schema
    import jsonschema
    story, _ = capture()
    values = annotated_boundaries(story)
    values['2->3'][field] = relation
    context = boundary_review_context(story)
    schema = boundary_response_schema({'properties': {'continuity_valid': {'type': 'boolean'}},
                                      'required': ['continuity_valid']}, context)
    jsonschema.validate({'boundaries': values}, schema)
    result, evidence = decode_boundary_review({'boundaries': values, 'reason': 'original'}, context)
    assert result['continuity_valid'] is (relation == 'supported')
    assert set(evidence) == {'1->2', '2->3', '3->4'}


@pytest.mark.parametrize('invalid', ['missing_edge', 'unknown_edge', 'wrong_side', 'duplicate', 'blank', 'coerced'])
def test_incomplete_or_fabricated_boundary_evidence_is_rejected(invalid):
    from pipeline.planning_boundaries import boundary_review_context, decode_boundary_review
    story, _ = capture()
    values = annotated_boundaries(story)
    if invalid == 'missing_edge':
        values.pop('2->3')
    elif invalid == 'unknown_edge':
        values['4->5'] = deepcopy(values['2->3'])
    elif invalid == 'wrong_side':
        values['2->3']['previous_evidence'] = values['2->3']['next_evidence']
    elif invalid == 'duplicate':
        values['2->3']['previous_evidence'] *= 2
    elif invalid == 'blank':
        values['2->3']['reason'] = ' '
    else:
        values['2->3']['state_relation'] = True
    with pytest.raises(ValueError):
        decode_boundary_review({'boundaries': values}, boundary_review_context(story))


def test_batch_boundary_uses_previous_batch_tail_and_does_not_review_future_edge():
    from pipeline.planning_boundaries import boundary_review_context
    story, _ = capture()
    story['review_scope'] = {'previous_shot': story['shots'][1], 'next_shot': story['shots'][3]}
    story['shots'] = story['shots'][2:3]
    context = boundary_review_context(story)
    assert list(context['boundaries']) == ['2->3']
    edge = context['boundaries']['2->3']
    assert '/review_scope/previous_shot/end_state/action_phase' in edge['previous']
    assert '/shots/0/start_state/action_phase' in edge['next']
    assert not any('/next_shot' in p for side in ('previous', 'next') for p in edge[side])
    story['review_scope'] = {}
    assert boundary_review_context(story) is None
    story['production_plan']['slots'] = [{'shot_id': 1}]
    assert boundary_review_context(story) is None  # 缺省标记的历史契约不变。


@pytest.mark.parametrize('topic', ['family', 'ensemble', 'product'])
def test_author_receives_whole_film_progress_and_remaining_slots_without_new_calls(monkeypatch, topic):
    from pipeline.storyboard import generate_storyboard
    from test_ordinary_scene_contract import scene_case
    brief, reply, identity, wires, _ = scene_case(topic)
    client = Responses(reply, identity, *wires)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    story = generate_storyboard(brief, allow_repair=False)
    assert len(client.calls) == 6
    for index, call in enumerate(client.calls[2:]):
        context = json.loads(call['messages'][-1]['content'])
        progress = context['story_progress']
        assert [s['shot_id'] for s in progress['remaining_slots']] == list(range(index + 1, 5))
        assert [s['shot_id'] for s in progress['completed']] == list(range(1, index + 1))
        assert progress['total_duration'] == sum(s['duration'] for s in story['shots'])
        assert progress['current_boundary'] == (wires[index - 1]['shot']['end_state'] if index else None)
        assert [s['state_after'] for s in progress['completed']] == [w['shot']['story_state_after'] for w in wires[:index]]


@pytest.mark.parametrize('reject_boundary', [False, True])
def test_full_batched_review_covers_every_cut_once_and_caches_failure(tmp_path, monkeypatch, reject_boundary):
    from types import SimpleNamespace
    from test_action_result_flow import annotated_verdict
    from planning_protocol_cases import annotated_outcome
    from test_full_production_flow import completion
    story, _ = capture()
    original = story['shots'][0]
    story['shots'] = [{**deepcopy(original), 'shot_id': i} for i in range(1, 15)]
    calls = []
    def create(**kwargs):
        payload = json.loads(kwargs['messages'][-1]['content'])
        calls.append(payload)
        batch = payload['storyboard']
        response = Responses(annotated_outcome(annotated_verdict(batch), batch)).create(**kwargs)
        raw = json.loads(response.choices[0].message.content)
        if reject_boundary and '6->7' in raw.get('boundaries', {}):
            raw['boundaries']['6->7'].update(action_progression='contradicted', reason='离线标注：批次交界重复已完成事件')
        return completion(raw)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: client)
    save = lambda n, v: RunWorkspace._atomic_write_json(tmp_path / n, v)
    for _ in range(2):
        if reject_boundary:
            with pytest.raises(ValueError, match='接缝 6->7'):
                ensure_planning_semantics(story, tmp_path, save)
        else:
            result = ensure_planning_semantics(story, tmp_path, save)
            assert len(result['shots']) == 14
    assert len(calls) == (2 if reject_boundary else 3)
    edges = [key for p in calls for key in p['boundary_review']['boundaries']]
    assert len(edges) == len(set(edges))
    if not reject_boundary:
        assert edges == [f'{i}->{i + 1}' for i in range(1, 14)]
    else:
        assert not (tmp_path / 'planning_semantic_review.json').exists()
