"""Review applicability and source binding, including the unmodified real failure."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline.planning_review import validate_planning_verdict
from pipeline.action_results import result_claims
from pipeline.result_admission import apply_result_admission
from test_result_admission import candidate, annotated
from pipeline import planning_review_protocol as protocol
from pipeline.planning_review import PlanVerdict, ensure_planning_semantics


def synthetic_wire(story):
    """Explicit synthetic labels, independent of the decoder/legacy conversion.

    These labels exercise protocol plumbing, never approve the recorded film.
    """
    units = {}
    for shot in story['shots']:
        for index, beat in enumerate(shot['action_beats']):
            fields = ['action']
            unit = {'cause_and_participants_valid': True,
                    'classification_matches_description': True,
                    'evidence_fields': fields, 'reason': 'Synthetic offline judgment'}
            if beat.get('outcomes'):
                fields.append('outcomes')
                unit['outcome_contract_valid'] = True
                if beat.get('scene_changes'):
                    fields.append('scene_changes')
            claims = result_claims(beat)
            if claims:
                unit['result_admission'] = {
                    'decisions': {path: {'basis': 'response', 'minimal': True,
                                        'evidence': 'Synthetic minimal response label'} for path in claims},
                    'requirements_preserved': True, 'continuous_performance': True,
                    'reason': 'Synthetic labels; not a semantic re-review'}
            units[f"{shot['shot_id']}:{index}"] = unit
    return {'units': units, 'continuity_valid': True, 'requested_outcome_valid': True,
            'reason': 'Synthetic whole-plan label'}


def real_story():
    return recorded()['storyboard.json']['value']


def first_claim_unit(wire):
    return next(unit for unit in wire['units'].values() if unit.get('result_admission'))


def save_to(directory):
    from pipeline.run_state import RunWorkspace
    return lambda name, data: RunWorkspace._atomic_write_json(directory / name, data)


def sdk_reviewer(monkeypatch, answer):
    import httpx
    from openai import OpenAI
    from structured_transport_cases import completion_response
    calls = []
    def receive(request):
        body = json.loads(request.content)
        calls.append(body)
        raw = answer(body) if callable(answer) else answer
        return completion_response(raw, body)
    sdk = OpenAI(api_key='offline-key', base_url='https://offline.invalid/v1', max_retries=0,
                 http_client=httpx.Client(transport=httpx.MockTransport(receive)))
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: sdk)
    return sdk, calls


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    def deny(*args, **kwargs):
        pytest.fail('offline review protocol attempted network access')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)


def recorded():
    return json.loads((Path(__file__).parent / 'fixtures/real_planning_review_binding_failure.json').read_text())['sources']


@pytest.mark.parametrize('absent', ['omitted', 'null'])
def test_no_candidate_claims_do_not_require_a_fictitious_result_review(absent):
    story = candidate()
    verdict = annotated(story)
    original = deepcopy(story)
    for unit in verdict['units']:
        if not result_claims(story['shots'][unit['shot_id'] - 1]['action_beats'][unit['beat_index']]):
            if absent == 'null':
                unit['result_admission'] = None
            else:
                unit.pop('result_admission')
    assert validate_planning_verdict(verdict, story) == []
    executed = apply_result_admission(story, verdict['units'])
    assert all(s['result_admission'] == 'reviewed-results-v1' for s in executed['shots'])
    assert story == original


def test_actual_response_is_not_auto_repaired_or_promoted():
    sources = recorded()
    story = sources['storyboard.json']['value']
    verdict = sources['planning_semantic_review.json']['value']['verdict']
    original = deepcopy(verdict)
    assert sum(u.get('result_admission') is None for u in verdict['units']) == 3
    # Missing nonapplicable reviews are not the whole defect: prose was used as
    # source IDs, and the final interaction has an extra classification.
    assert all(not d['source'].startswith(('outcomes/', 'scene_changes/'))
               for u in verdict['units'] if u.get('result_admission') for d in u['result_admission']['decisions'])
    with pytest.raises(ValueError, match='来源|逐条'):
        validate_planning_verdict(verdict, story)
    assert verdict == original


def test_missing_applicable_review_is_not_unknown_or_approval():
    story = candidate()
    verdict = annotated(story)
    unit = next(u for u in verdict['units'] if u['result_admission']['decisions'])
    unit['result_admission'] = None
    with pytest.raises(ValueError, match='缺少.*结果'):
        validate_planning_verdict(verdict, story)


def test_fixed_keys_preserve_every_claim_and_source_without_mutation():
    story = real_story()
    wire = synthetic_wire(story)
    before = deepcopy((story, wire))
    targets = protocol.review_targets(story)
    decoded = protocol.decode_review_response(wire, targets)
    assert validate_planning_verdict(decoded, story) == []
    assert len(targets) == 11
    assert sum(not t['candidate_results'] for t in targets.values()) == 3
    for unit in decoded['units']:
        key = f"{unit['shot_id']}:{unit['beat_index']}"
        assert [d['source'] for d in (unit.get('result_admission') or {}).get('decisions', [])] == list(targets[key]['candidate_results'])
    assert (story, wire) == before


def test_equal_claim_text_is_not_deduplicated():
    story = real_story()
    beat = next(b for s in story['shots'] for b in s['action_beats'] if b.get('outcomes'))
    beat['outcomes'].append(deepcopy(beat['outcomes'][0]))
    wire = synthetic_wire(story)
    decoded = protocol.decode_review_response(wire, protocol.review_targets(story))
    assert validate_planning_verdict(decoded, story) == []
    first_claim_unit(wire)['result_admission']['decisions'].pop(f"outcomes/{len(beat['outcomes']) - 1}")
    with pytest.raises(ValueError):
        protocol.decode_review_response(wire, protocol.review_targets(story))


@pytest.mark.parametrize('fault', ['missing_unit', 'extra_unit', 'list_units', 'missing_claim',
    'extra_claim', 'prose_source', 'embedded_id', 'embedded_source', 'missing_admission',
    'invented_admission', 'missing_outcome_check', 'string_bool', 'unknown_field'])
def test_malformed_wire_cannot_be_promoted_even_if_provider_ignores_schema(fault):
    story = real_story()
    wire = synthetic_wire(story)
    unit = first_claim_unit(wire)
    decisions = unit['result_admission']['decisions']
    path = next(iter(decisions))
    if fault == 'missing_unit':
        wire['units'].pop(next(iter(wire['units'])))
    elif fault == 'extra_unit':
        wire['units']['999:0'] = deepcopy(unit)
    elif fault == 'list_units':
        wire['units'] = list(wire['units'].values())
    elif fault == 'missing_claim':
        decisions.pop(path)
    elif fault == 'extra_claim':
        decisions['outcomes/999'] = deepcopy(decisions[path])
    elif fault == 'prose_source':
        decisions['participant: prose copied from source'] = decisions.pop(path)
    elif fault == 'embedded_id':
        unit['shot_id'] = 1
    elif fault == 'embedded_source':
        decisions[path]['source'] = path
    elif fault == 'missing_admission':
        unit['result_admission'] = None
    elif fault == 'invented_admission':
        next(u for u in wire['units'].values() if 'result_admission' not in u)['result_admission'] = deepcopy(unit['result_admission'])
    elif fault == 'missing_outcome_check':
        unit.pop('outcome_contract_valid')
    elif fault == 'string_bool':
        unit['cause_and_participants_valid'] = 'true'
    else:
        unit['unrecognized'] = True
    before = deepcopy(wire)
    with pytest.raises(ValueError):
        validate_planning_verdict(protocol.decode_review_response(wire, protocol.review_targets(story)), story)
    assert wire == before


@pytest.mark.parametrize('flag', ['continuity_valid', 'requested_outcome_valid',
    'requirements_preserved', 'continuous_performance', 'classification_matches_description'])
def test_bound_protocol_preserves_negative_semantic_judgments(flag):
    story = real_story()
    wire = synthetic_wire(story)
    if flag in wire:
        wire[flag] = False
    elif flag == 'classification_matches_description':
        next(u for u in wire['units'].values() if 'result_admission' not in u)[flag] = False
    else:
        first_claim_unit(wire)['result_admission'][flag] = False
    assert validate_planning_verdict(protocol.decode_review_response(wire, protocol.review_targets(story)), story)


def test_local_and_provider_schemas_describe_same_bound_keys():
    jsonschema = pytest.importorskip('jsonschema')
    from pipeline.llm_options import ark_output_schema
    story = real_story()
    wire = synthetic_wire(story)
    schema = protocol.compile_review_response_schema(PlanVerdict.model_json_schema(), protocol.review_targets(story))
    for shape in (schema, ark_output_schema(schema)):
        jsonschema.Draft202012Validator.check_schema(shape)
        jsonschema.validate(wire, shape)
        invalid = deepcopy(wire)
        first_claim_unit(invalid)['result_admission']['decisions']['extra'] = {}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, shape)
        for unit in shape['properties']['units']['properties'].values():
            assert 'shot_id' not in unit['properties'] and 'beat_index' not in unit['properties']
        assert 'source' not in shape['$defs']['BoundResultDecision']['properties']


@pytest.mark.parametrize('fault', [None, 'missing_claim', 'semantic_rejection'])
def test_actual_sdk_boundary_caches_raw_success_and_failure_without_retry(tmp_path, monkeypatch, fault):
    from planning_protocol_cases import annotated_outcome, sourced_wire
    story = real_story()
    wire = sourced_wire(annotated_outcome(synthetic_wire(story), story), story)
    if fault == 'missing_claim':
        first_claim_unit(wire)['result_admission']['decisions'].popitem()
    elif fault:
        wire['continuity_valid'] = False
    sdk, calls = sdk_reviewer(monkeypatch, wire)
    with sdk:
        for _ in range(2):
            if fault:
                with pytest.raises(ValueError):
                    ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
            else:
                result = ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
                assert all(s['result_admission'] == 'reviewed-results-v1' for s in result['shots'])
    assert len(calls) == 1
    payload = json.loads(calls[0]['messages'][-1]['content'])
    assert payload['review_targets'] == protocol.review_targets(story)
    fmt = calls[0]['response_format']['json_schema']
    assert fmt['strict'] is True and fmt['schema']['properties']['units']['type'] == 'object'
    saved = json.loads((tmp_path / 'planning_semantic_review.json').read_text())
    assert saved['verdict'] == wire and saved['protocol_version'] == protocol.VERSION
    assert story == real_story()


def test_real_legacy_failed_cache_is_not_rewritten_or_rereviewed(tmp_path, monkeypatch):
    import config
    import hashlib
    # Replay the recorded request's model, independent of the developer's .env.
    monkeypatch.setattr(config, 'LLM_MODEL', 'doubao-seed-2.1-lite')
    story = real_story()
    cache = recorded()['planning_semantic_review.json']['value']
    payload = {'version': 'planning-semantics-v5-result-necessity', 'model': config.LLM_MODEL, 'storyboard': story}
    assert cache['fingerprint'] == hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    save_to(tmp_path)('planning_semantic_review.json', cache)
    path = tmp_path / 'planning_semantic_review.json'
    original = path.read_bytes()
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: pytest.fail('historical failure was retried'))
    for _ in range(2):
        with pytest.raises(ValueError, match='来源|逐条'):
            ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
    assert path.read_bytes() == original


def test_legacy_nonapplicable_negative_judgment_is_not_discarded():
    story = candidate()
    raw = annotated(story)
    passive = next(u for u in raw['units'] if not u['result_admission']['decisions'])
    passive['result_admission']['continuous_performance'] = False
    assert validate_planning_verdict(raw, story)
    with pytest.raises(ValueError):
        apply_result_admission(story, raw['units'])


def test_unknown_cache_version_stops_without_migration(tmp_path, monkeypatch):
    save_to(tmp_path)('planning_semantic_review.json', {'protocol_version': 'unknown'})
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: pytest.fail('unknown cache retried'))
    with pytest.raises(ValueError, match='缓存协议'):
        ensure_planning_semantics(real_story(), tmp_path, save_to(tmp_path))


def test_bound_batch_interruption_resumes_without_repeating_saved_judgments(tmp_path, monkeypatch):
    from planning_protocol_cases import annotated_outcome, sourced_wire
    story = real_story()
    story['shots'] = [{**deepcopy(story['shots'][0]), 'shot_id': i} for i in range(1, 10)]
    fail = True
    def answer(body):
        nonlocal fail
        payload = json.loads(body['messages'][-1]['content'])
        batch = payload['storyboard']
        if batch['shots'][0]['shot_id'] > 1 and fail:
            fail = False
            raise RuntimeError('offline interruption')
        return sourced_wire(annotated_outcome(synthetic_wire(batch), batch), batch)
    sdk, calls = sdk_reviewer(monkeypatch, answer)
    with sdk:
        with pytest.raises(Exception, match='offline interruption|Connection|调用失败|LLM'):
            ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
        admitted = ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
        count = len(calls)
        assert ensure_planning_semantics(story, tmp_path, save_to(tmp_path)) == admitted
        assert len(calls) == count
    batches = [json.loads(c['messages'][-1]['content'])['storyboard'] for c in calls]
    assert sum(b['shots'][0]['shot_id'] == 1 for b in batches) == 1
    assert [s['shot_id'] for s in admitted['shots']] == list(range(1, 10))
    assert all(s['result_admission'] == 'reviewed-results-v1' for s in admitted['shots'])
