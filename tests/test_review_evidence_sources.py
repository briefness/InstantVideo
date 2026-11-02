"""Actual source confusion stays rejected; annotations exercise the new protocol."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline import planning_review_protocol as protocol
from pipeline.planning_review import PlanVerdict, ensure_planning_semantics
from pipeline.planning_review_sources import review_evidence_sources
from test_planning_review_protocol import synthetic_wire, save_to, sdk_reviewer, first_claim_unit


def record():
    return json.loads((Path(__file__).parent / 'fixtures/real_user_source_failure.json').read_text())['sources']


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **kw: pytest.fail('external connection'))


def test_real_cached_failure_remains_failed_and_byte_identical(tmp_path, monkeypatch):
    import config
    from pipeline.models import validate_storyboard
    monkeypatch.setattr(config, 'LLM_MODEL', 'doubao-seed-2.1-lite')
    raw = record()
    story = validate_storyboard(raw['storyboard.json']['value'])
    save_to(tmp_path)('planning_semantic_review.json', raw['planning_semantic_review.json']['value'])
    path = tmp_path / 'planning_semantic_review.json'
    before = path.read_bytes()
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: pytest.fail('retry'))
    for _ in range(2):
        with pytest.raises(ValueError, match='用户结果依据'):
            ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
    assert path.read_bytes() == before
    assert not (tmp_path / 'result_admission.json').exists()


def test_provider_request_cannot_use_story_as_a_user_source():
    import jsonschema
    from pipeline.planning_review_sources import review_evidence_sources
    story = record()['storyboard.json']['value']
    sources = review_evidence_sources(story)
    targets = protocol.review_targets(story)
    schema = protocol.compile_review_response_schema(PlanVerdict.model_json_schema(), targets, evidence_sources=sources)
    unit = schema['properties']['units']['properties']['3:0']
    ref = unit['properties']['result_admission']['properties']['decisions']['properties']['outcomes/0']['$ref']
    variants = schema['$defs'][ref.rsplit('/', 1)[-1]]['oneOf']
    user = next(v for v in variants if v['properties']['basis'].get('const') == 'user')
    choices = user['properties']['reference']['enum']
    assert choices and all(path.startswith('/request/segments/') for path in choices)
    assert all(sources['fields'][path] in story['production_plan']['director_intent']['source_request'] for path in choices)
    bad = {'basis': 'user', 'reference': '/shots/2/action_beats/0/visible_result',
           'composition': 'single_category', 'reason': 'claims its own script is a requirement'}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, user)


@pytest.mark.parametrize('composition', ['mixed_categories', 'uncertain'])
def test_known_mixed_or_uncertain_claim_never_becomes_approved(tmp_path, monkeypatch, composition):
    from planning_protocol_cases import sourced_wire, annotated_outcome
    story = record()['storyboard.json']['value']
    wire = sourced_wire(annotated_outcome(synthetic_wire(story), story), story)
    decision = next(iter(first_claim_unit(wire)['result_admission']['decisions'].values()))
    decision['composition'] = composition
    sdk, calls = sdk_reviewer(monkeypatch, wire)
    with sdk:
        for _ in range(2):
            with pytest.raises(ValueError, match='必要结果与表演细节'):
                ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
    assert len(calls) == 1 and not (tmp_path / 'result_admission.json').exists()
    assert json.loads((tmp_path / 'planning_semantic_review.json').read_text())['verdict'] == wire


@pytest.mark.parametrize('change', [
    {'basis': 'user', 'reference': '/shots/0/action_beats/0/visible_result'},
    {'basis': 'user', 'reference': '/request/segments/999'},
    {'basis': 'continuity', 'reference': '/shots/0/action_beats/0/action'},
    {'basis': 'response', 'reference': '/shots/2/action_beats/0/action'},
    {'basis': 'direction', 'reference': '/request/segments/0'},
    {'basis': []}, {'reference': {'path': '/request/segments/0'}},
    {'composition': True}, {'composition': 'true'}, {'composition': 'approved'},
    {'reason': '  '}, {'reason': 'x' * 1601}, {'reason': None},
    {'evidence': 'legacy free text'}, {'minimal': True},
])
def test_decoder_rejects_wrong_source_or_shape_without_mutating_input(change):
    from planning_protocol_cases import sourced_wire
    story = record()['storyboard.json']['value']
    raw = sourced_wire(synthetic_wire(story), story)
    decision = next(iter(first_claim_unit(raw)['result_admission']['decisions'].values()))
    decision.update(change)
    before = deepcopy(raw)
    with pytest.raises(ValueError):
        protocol.decode_review_response(raw, protocol.review_targets(story),
                                        evidence_sources=review_evidence_sources(story))
    assert raw == before


def test_user_reference_resolves_original_text_and_keeps_separate_reason():
    from planning_protocol_cases import sourced_wire
    story = record()['storyboard.json']['value']
    sources = review_evidence_sources(story)
    raw = sourced_wire(synthetic_wire(story), story)
    key = next(k for k, u in raw['units'].items() if u.get('result_admission'))
    path, decision = next(iter(raw['units'][key]['result_admission']['decisions'].items()))
    reference = sources['allowed'][key]['user'][0]
    decision.update(basis='user', reference=reference, reason='Synthetic source mapping, not semantic approval')
    before = deepcopy(raw)
    decoded = protocol.decode_review_response(raw, protocol.review_targets(story), evidence_sources=sources)
    unit = next(u for u in decoded['units'] if f"{u['shot_id']}:{u['beat_index']}" == key)
    result = next(d for d in unit['result_admission']['decisions'] if d['source'] == path)
    assert result['evidence'] == sources['fields'][reference]
    assert result['evidence'] in story['production_plan']['director_intent']['source_request']
    assert result['reason'] == decision['reason']
    assert result['minimal'] is True and raw == before


@pytest.mark.parametrize('source_text', ['', ' \n ', 'Please finish. 保持连续！结尾完成？\n引号"和/路径~0不解释。', '长' * 3201],
                         ids=['empty', 'whitespace', 'multilingual', 'long'])
def test_source_spans_are_exact_bounded_text_not_generated_requirements(source_text):
    story = record()['storyboard.json']['value']
    story['production_plan']['director_intent']['source_request'] = source_text
    sources = review_evidence_sources(story)
    user = sources['allowed']['3:0']['user']
    excerpts = [sources['fields'][path] for path in user]
    assert all(text in source_text and 0 < len(text) <= 1600 for text in excerpts)
    assert ''.join(excerpts).replace('\n', '').replace(' ', '') == source_text.replace('\n', '').replace(' ', '')
    schema = protocol.compile_review_response_schema(PlanVerdict.model_json_schema(),
                                                      protocol.review_targets(story), evidence_sources=sources)
    variant = next(value['oneOf'] for name, value in schema['$defs'].items()
                   if name.startswith('BoundResultDecision_'))
    assert any(v['properties']['basis']['const'] == 'user' for v in variant) == bool(source_text.strip())


@pytest.mark.parametrize('actions', [
    ('装配者递出零件', '接收者接住零件', '把零件装入设备'),
    ('舞者牵引舞伴旋转', '舞伴顺势转身', '双方携手向前移动'),
    ('球员把球传出', '队友接到球', '持球队员向球门跑动'),
])
def test_sources_follow_data_boundaries_across_scenes(actions):
    story = {'production_plan': {'director_intent': {'source_request': '动作连贯，展示完成结果。'}},
             'shots': [{'shot_id': 17, 'action_beats': [
                 {'action': actions[0], 'visible_result': actions[1]},
                 {'action': actions[2], 'visible_result': '完成'}]}],
             'review_scope': {'next_shot': {'action_beats': [{'action': '展示完成状态'}]}}}
    sources = review_evidence_sources(story)
    assert sources['allowed']['17:0']['response'] == [
        '/shots/0/action_beats/0/action', '/shots/0/action_beats/0/visible_result']
    assert '/shots/0/action_beats/1/action' in sources['allowed']['17:0']['continuity']
    assert '/shots/0/action_beats/0/action' not in sources['allowed']['17:1']['continuity']
    assert sources['allowed']['17:1']['continuity'] == ['/review_scope/next_shot/action_beats/0/action']


def test_new_protocol_success_is_cached_and_changed_request_is_not_rejudged(tmp_path, monkeypatch):
    from planning_protocol_cases import sourced_wire, annotated_outcome
    story = record()['storyboard.json']['value']
    wire = sourced_wire(annotated_outcome(synthetic_wire(story), story), story)
    sdk, calls = sdk_reviewer(monkeypatch, wire)
    with sdk:
        accepted = ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
        before = (tmp_path / 'planning_semantic_review.json').read_bytes()
        assert ensure_planning_semantics(story, tmp_path, save_to(tmp_path)) == accepted
        changed = deepcopy(story)
        changed['production_plan']['director_intent']['source_request'] += ' 新的限制。'
        with pytest.raises(ValueError, match='不一致'):
            ensure_planning_semantics(changed, tmp_path, save_to(tmp_path))
    assert len(calls) == 1
    prompt = calls[0]['messages'][0]['content']
    assert 'evidence必须' not in prompt and 'minimal表示' not in prompt
    assert 'composition=single_category' in prompt
    schema_text = json.dumps(calls[0]['response_format']['json_schema']['schema'])
    assert '"minimal"' not in schema_text and '"BoundResultDecision"' not in schema_text
    assert (tmp_path / 'planning_semantic_review.json').read_bytes() == before
