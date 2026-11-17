"""容量来自证据目录；真实响应只读重放，不修改历史审查或调用外部服务。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from pipeline import goal_allocation as allocation
from pipeline.completion_intent import source_goals, validate_goal_coverage
from pipeline.planning_review_protocol import review_targets
from pipeline.result_admission import apply_result_admission
from test_goal_allocation import evidence, offline


def capture():
    fixture = json.loads((Path(__file__).parent / 'fixtures/real_goal_evidence_capacity.json').read_text())
    for record in fixture['files'].values():
        assert hashlib.sha256(record['text'].encode()).hexdigest() == record['sha256']
    return tuple(json.loads(fixture['files'][n]['text'])
                 for n in ('storyboard.json', 'planning_semantic_review.json'))


def bind(story, judgments):
    targets = review_targets(story)
    units = [{'shot_id': t['shot_id'], 'beat_index': t['beat_index']} for t in targets.values()]
    raw = allocation.decode_allocation({'units': units, 'goal_allocation': judgments},
        source_goals(story), targets, storyboard=story)
    return apply_result_admission(story, raw['units'])


def allocation_schema(story):
    return allocation.response_schema({'type': 'object', 'properties': {}, 'required': []},
                                      source_goals(story), review_targets(story))


def test_real_fifteen_unit_response_is_accepted_by_source_bound_request():
    story, review = capture()
    goals = review['verdict']['goal_allocation']
    assert [len(v['parts'][0]['units']) for v in goals.values()] == [2, 11, 2, 15]
    jsonschema.validate({'goal_allocation': goals}, allocation_schema(story))


def test_real_fifteen_unit_response_reaches_execution_without_evidence_loss():
    from pipeline.causality import compile_action_contract
    from pipeline.assessment_scope import local_assessment
    from pipeline.generation_instructions import local_goal_instructions
    story, review = capture()
    original = deepcopy((story, review))
    bound = bind(story, review['verdict']['goal_allocation'])
    validate_goal_coverage(bound)
    for shot in bound['shots']:
        catalog = compile_action_contract(shot).review_projection['completion_contract']
        criterion = next(c for c in catalog['criteria'].values()
                         if c.get('source', {}).get('reference') == '/intent/completion_goals/3')
        assert len(criterion['execution_scope']['units']) == 15
        local = local_assessment(catalog)
        for c in local['criteria'].values():
            if 'execution_scope' in c:
                assert all(u['shot_id'] == shot['shot_id'] for u in c['execution_scope']['units'])
        assert all(isinstance(text, str) for text in local_goal_instructions(catalog))
    assert (story, review) == original


@pytest.mark.parametrize('goal,action', [
    ('Dancers preserve formation throughout the performance.', 'Dancers continue their formation.'),
    ('Packages remain in custody throughout the transfer.', 'Courier carries the sealed parcel.'),
    ('The lamp is shown from different sides without changing shape.', 'The lamp rotates to its next side.'),
])
@pytest.mark.parametrize('size', [12, 13, 24, 25, 120])
def test_cross_scene_source_bound_capacity(goal, action, size):
    story = {'production_plan': {'director_intent': {'authoring_protocol': 'scene-units-v1',
        'source_request': goal, 'completion_goals': [{'goal': goal, 'basis': 'user', 'evidence': goal}]}},
        'shots': [{'shot_id': 900 - i, 'result_admission': 'review-required-v1',
            'production_slot': {'choreography': 'flexible'}, 'action_beats': [
                {'actor': 'subject', 'activity': 'transition', 'action': action, 'visible_result': action}]}
            for i in range(size)]}
    refs = list(review_targets(story))
    judgments = {'/intent/completion_goals/0': {'parts': [evidence(goal, *refs)]}}
    jsonschema.validate({'goal_allocation': judgments}, allocation_schema(story))
    bound = bind(story, judgments)
    validate_goal_coverage(bound)
    assert len(bound['shots'][0]['action_beats'][0]['completion_intent']['requirements'][0]['scope']) == size


@pytest.mark.parametrize('keys,message', [
    ([], '非空'), (None, '非空'), ([{'key': '1:0'}], '字符串'),
    (['1:0', '1:0'], '重复'), (['999:0'], '当前审查动作目录'),
    (['2:0', '1:0'], '时间线顺序'),
])
def test_source_errors_are_distinct_and_never_silently_repaired(keys, message):
    story, review = capture()
    original = deepcopy(review)
    judgments = deepcopy(review['verdict']['goal_allocation'])
    judgments['/intent/completion_goals/3']['parts'][0]['units'] = keys
    with pytest.raises(ValueError, match=message) as error:
        bind(story, judgments)
    assert '/intent/completion_goals/3/parts/0' in str(error.value)
    assert review == original


@pytest.mark.parametrize('mutation', ['missing_unit', 'duplicate_unit', 'wrong_identity', 'wrong_order'])
def test_decoder_binds_review_catalog_and_originals_before_assigning_evidence(mutation):
    story, review = capture()
    targets = review_targets(story)
    units = [{'shot_id': t['shot_id'], 'beat_index': t['beat_index']} for t in targets.values()]
    if mutation == 'missing_unit':
        units.pop()
    elif mutation == 'duplicate_unit':
        units.append(units[0])
    elif mutation == 'wrong_identity':
        targets['1:0']['shot_id'] = 99
    else:
        targets = dict(reversed(list(targets.items())))
    with pytest.raises(ValueError, match='目录'):
        allocation.decode_allocation({'units': units, 'goal_allocation': review['verdict']['goal_allocation']},
            source_goals(story), targets, storyboard=story)


def test_legacy_real_cache_keeps_original_failure_and_never_calls_model(tmp_path, monkeypatch):
    from pipeline.planning_review import ensure_planning_semantics
    from pipeline.run_state import RunWorkspace
    story, review = capture()
    assert review['protocol_version'] == allocation.SCOPED_VERSION
    path = tmp_path / 'planning_semantic_review.json'
    RunWorkspace._atomic_write_json(path, review)
    before = path.read_bytes()
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: pytest.fail('no retry'))
    for _ in range(2):
        with pytest.raises(ValueError, match='历史协议目标证据容量超限（15/12）'):
            ensure_planning_semantics(story, tmp_path, lambda *_: pytest.fail('do not alter evidence'))
    assert path.read_bytes() == before


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'reordered', 'changed_action', 'lost_owner'])
def test_merge_rejects_evidence_loss_even_when_goal_words_remain_covered(mutation):
    from pipeline.planning_review_batches import merge_review_batches
    story, review = capture()
    bound = bind(story, review['verdict']['goal_allocation'])
    shots = deepcopy(bound['shots'])
    if mutation == 'missing':
        shots.pop(0)
    elif mutation == 'duplicate':
        shots.append(shots[0])
    elif mutation == 'reordered':
        shots.reverse()
    elif mutation == 'changed_action':
        shots[0]['action_beats'][0]['action'] = 'invented'
    else:
        shots[0]['action_beats'][0]['completion_intent']['requirements'] = []
    with pytest.raises(ValueError, match='目录|重复|承接'):
        merge_review_batches(story, shots)


def test_new_formal_review_replays_original_bytes_once_and_reuses_cache(tmp_path, monkeypatch):
    from pipeline.planning_review import ensure_planning_semantics
    from pipeline.run_state import RunWorkspace
    from test_director_intent import Responses
    story, review = capture()
    sdk = Responses(review['verdict'])
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **_: sdk)
    save = lambda name, data: RunWorkspace._atomic_write_json(tmp_path / name, data)
    admitted = ensure_planning_semantics(story, tmp_path, save)
    assert len(sdk.calls) == 1
    schema = sdk.calls[0]['response_format']['json_schema']['schema']
    jsonschema.validate(review['verdict'], schema)
    stored = json.loads((tmp_path / 'planning_semantic_review.json').read_text())
    assert stored['verdict'] == review['verdict']
    assert stored['protocol_version'] == allocation.VERSION
    assert ensure_planning_semantics(story, tmp_path, save) == admitted
    assert len(sdk.calls) == 1


def test_request_subset_does_not_replace_local_evidence_validation():
    from pipeline.llm_options import ark_output_schema
    story, review = capture()
    judgments = deepcopy(review['verdict']['goal_allocation'])
    part = judgments['/intent/completion_goals/3']['parts'][0]
    part['units'].append(part['units'][0])
    # 适配器不支持 uniqueItems，因此云端格式约束接受，不代表本地证据可接受。
    jsonschema.validate({'goal_allocation': judgments}, ark_output_schema(allocation_schema(story)))
    with pytest.raises(ValueError, match='重复'):
        bind(story, judgments)


@pytest.mark.parametrize('size', [1, 2, 7, 12, 15, 24, 120])
def test_selection_capacity_is_derived_from_catalog_not_another_fixed_number(size):
    from pipeline.planning_evidence import selection_schema, validate_selection
    targets = {f'{i + 1}:0': {} for i in range(size)}
    schema = selection_schema(targets)
    assert schema['maxItems'] == size
    assert validate_selection(list(targets), targets, label='test') == list(targets)
    with pytest.raises(ValueError, match='当前审查动作目录'):
        validate_selection([*targets, 'outside:0'], targets, label='test')


@pytest.mark.parametrize('size', [0, 121])
def test_resource_bounds_do_not_increase_with_untrusted_evidence(size):
    from pipeline.planning_evidence import selection_schema, validate_selection
    targets = {str(i): {} for i in range(size)}
    with pytest.raises(ValueError, match='单批范围'):
        selection_schema(targets)
    with pytest.raises(ValueError, match='单批范围'):
        validate_selection(list(targets), targets, label='test')
