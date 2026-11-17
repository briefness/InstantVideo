"""合法未知不冒充协议故障，坏来源也不能借未知绕过校验。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from pipeline.result_observations import decode_observations, ObservationUncertaintyError
from pipeline.semantic_review import SemanticTakeReviewer, SemanticReviewUnavailableError
from pipeline.orchestrator import VideoPipeline
from pipeline.run_state import RunWorkspace
from test_contribution_witness import offline
from test_creation_schedule_flow import SCENARIOS
from test_full_production_flow import completion
from test_scheduled_production_flow import scheduled_services


def captured():
    root = Path(__file__).parents[1]/'output/20261117_104852_942983'
    story = json.loads((root/'result_admission.json').read_text())['execution_storyboard']
    evidence = json.loads(next((root/'semantic_reviews').glob('*.evidence.json')).read_text())
    return root, story['shots'][0], evidence


def unresolved(raw, target='goal'):
    value = deepcopy(raw)
    if target == 'terminal':
        value['terminal_observation']['status'] = 'unknown'
        return value, 'terminal_observation'
    key = next(iter(value['observations']))
    value['observations'][key].update(status='unknown',
        observation='The required event is occluded in the supplied samples.')
    return value, key


@pytest.mark.parametrize('fault', [None, 'frame_address', 'terminal', 'field_type', 'event_address', 'foreign_field'])
@pytest.mark.parametrize('target', ['goal', 'terminal'])
def test_unknown_classification_follows_complete_source_validation(fault, target):
    from pipeline.causality import compile_action_contract
    from pipeline.narrative import requires_narrative_review
    from pipeline.semantic_review import _review_contract, _review_response_format
    _, shot, evidence = captured()
    action_contract = compile_action_contract(shot)
    contract = _review_contract(shot)
    schema = _review_response_format(
        required_entity_count=len(contract.get('required_visible_entities') or []),
        identity_crop_count=len(shot['characters']), action_contract=action_contract,
        sample_count=len(evidence['sample_timestamps']), causal_review=bool(contract.get('action_events')),
        evidence_review=False, narrative_review=requires_narrative_review(shot),
        blocking_review=bool(shot.get('blocking')), composition_review=False,
        invariant_review=bool(contract.get('visual_invariants')), boundary_review=False,
        state_handoff_review=False)['json_schema']['schema']
    value, key = unresolved(evidence['response'], target)
    if fault == 'frame_address':
        value['observations'][next(iter(value['observations']))]['sample_indices'] = [999]
    elif fault == 'terminal':
        del value['terminal_observation']['action_phase']
    elif fault == 'field_type':
        value['action_geometry_valid'] = 'unknown'
    elif fault == 'event_address':
        value['action_event_evidence'][0]['event_id'] = 999
    elif fault == 'foreign_field':
        value['action_event_evidence'][0]['unexpected_fact'] = True
    catalog = action_contract.review_projection['completion_contract']
    with pytest.raises(ValueError) as caught:
        decode_observations(value, catalog, schema, len(evidence['sample_timestamps']))
    assert isinstance(caught.value, ObservationUncertaintyError) is (fault is None)


@pytest.mark.parametrize('target', ['goal', 'terminal'])
def test_original_pixels_with_annotated_unknown_pause_once_and_restore(tmp_path, target):
    root, shot, evidence = captured()
    source = root/'shots/shot_001.mp4'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    value, key = unresolved(evidence['response'], target)
    calls = []
    def create(**request):
        calls.append(request)
        return completion(value)
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-unknown',
        client=NS(chat=NS(completions=NS(create=create))))
    for _ in range(2):
        with pytest.raises(SemanticReviewUnavailableError, match='语义不确定'):
            reviewer.review(str(source), shot, identity_crop_entities=shot['characters'])
    saved = json.loads(next(reviewer.cache_dir.glob('*.evidence.json')).read_text())
    assert saved['failure_kind'] == 'semantic_uncertainty'
    assert saved['unresolved_criteria'] == [key] and saved['response'] == value
    assert len(calls) == 1 and not (reviewer.cache_dir/'support').exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', SCENARIOS)
async def test_formal_flow_restores_semantic_unknown_without_review_or_new_take(scheduled_services, topic):
    brief, services = scheduled_services(topic)
    original = services.create
    calls = []
    def create(**request):
        calls.append(request)
        reply = original(**request)
        if isinstance(request['messages'][-1]['content'], list):
            value, _ = unresolved(json.loads(reply.choices[0].message.content))
            from jsonschema import Draft202012Validator
            schema = request['response_format']['json_schema']['schema']
            # 共享替身保留旧版效果标签；本用例按当前请求构造合法响应，再测试unknown分类。
            fields = schema['properties']['action_event_evidence']['items']['properties']
            value['action_event_evidence'] = [{k:v for k,v in row.items() if k in fields}
                                              for row in value['action_event_evidence']]
            errors = list(Draft202012Validator(schema).iter_errors(value))
            assert not errors, [(list(e.path), e.message) for e in errors]
            return completion(value)
        return reply
    services.create = create
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'],
        paid_take_budget=3, take_policy='single_pass')
    with pytest.raises(SemanticReviewUnavailableError, match='语义不确定'):
        await pipeline.run(brief)
    counts = len(calls), len(services.submissions)
    with pytest.raises(SemanticReviewUnavailableError, match='语义不确定'):
        await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert (len(calls), len(services.submissions)) == counts
    assert len(services.submissions) == 1
    assert not (pipeline.workspace/'semantic_reviews/support').exists()
    assert RunWorkspace.resume(pipeline.workspace).manifest.shots['1'].canonical_take_id is None
    assert not (pipeline.workspace/'final.mp4').exists()
