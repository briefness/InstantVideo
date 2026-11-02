"""Source-bound review citations; real failed judgments are never repaired."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from pipeline import planning_review_protocol as protocol
from pipeline.planning_review import PlanVerdict, ensure_planning_semantics, validate_planning_verdict
from test_planning_review_protocol import synthetic_wire, save_to, sdk_reviewer


def recorded():
    return json.loads((Path(__file__).parent / 'fixtures/real_planning_citation_failure.json').read_text())['sources']


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    def deny(*args, **kwargs):
        pytest.fail('citation regression attempted network access')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket.socket, 'connect_ex', deny)


def test_original_real_cache_stays_failed_without_rereview_or_rewrite(tmp_path, monkeypatch):
    import config
    from pipeline.models import validate_storyboard
    monkeypatch.setattr(config, 'LLM_MODEL', 'doubao-seed-2.1-lite')
    raw = recorded()
    # The formal workspace validates persisted JSON before fingerprinting it
    # (e.g. JSON integer durations become typed floats).
    story = validate_storyboard(raw['storyboard.json']['value'])
    cache = raw['planning_semantic_review.json']['value']
    assert cache['verdict']['requested_outcome_valid'] is True
    save_to(tmp_path)('planning_semantic_review.json', cache)
    path = tmp_path / 'planning_semantic_review.json'
    before = path.read_bytes()
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: pytest.fail('failed review retried'))
    for _ in range(2):
        with pytest.raises(ValueError, match='衔接依据'):
            ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
    assert path.read_bytes() == before


def test_provider_schema_only_offers_readable_forward_references():
    from pipeline.planning_review_sources import continuity_sources
    story = recorded()['storyboard.json']['value']
    sources = continuity_sources(story)
    targets = protocol.review_targets(story)
    schema = protocol.compile_review_response_schema(PlanVerdict.model_json_schema(), targets, sources=sources)
    unit = schema['properties']['units']['properties']['2:3']
    ref = unit['properties']['result_admission']['properties']['decisions']['properties']['outcomes/1']['$ref']
    variants = schema['$defs'][ref.split('/')[-1]]['oneOf']
    choices = next(v for v in variants if v['properties']['basis'].get('const') == 'continuity')['properties']['evidence']['enum']
    assert '/shots/2/start_state/prop_state' in choices
    assert '/shots/3/start_state' not in choices
    assert not any(p.startswith('/shots/1/') for p in choices)
    bad = recorded()['planning_semantic_review.json']['value']['verdict']
    assert bad['units']['2:3']['result_admission']['decisions']['outcomes/1']['evidence'] not in choices


@pytest.mark.parametrize('reference', [
    '/shots/3/start_state', '/shots/2/start_state', '/shots/01/start_state/subject',
    '/shots/0/start_state/subject', '/shots/1/action_beats/3/action',
    '后续依赖：/shots/2/start_state/subject', '/etc/passwd', 'https://example.invalid',
])
def test_ignored_provider_schema_cannot_admit_invented_or_backward_reference(reference):
    from pipeline.planning_review_sources import continuity_sources
    story = recorded()['storyboard.json']['value']
    wire = synthetic_wire(story)
    wire['units']['2:3']['result_admission']['decisions']['outcomes/1'].update(
        basis='continuity', evidence=reference)
    with pytest.raises(ValueError, match='衔接引用'):
        protocol.decode_review_response(wire, protocol.review_targets(story), sources=continuity_sources(story))


def test_valid_within_shot_and_next_shot_references_preserve_source_identity():
    from pipeline.planning_review_sources import continuity_sources
    story = recorded()['storyboard.json']['value']
    wire = synthetic_wire(story)
    # Explicit offline labels exercise binding, not approval of the recorded plot.
    wire['units']['1:2']['result_admission']['decisions']['outcomes/0'].update(
        basis='continuity', evidence='/shots/0/action_beats/3/action')
    wire['units']['2:3']['result_admission']['decisions']['outcomes/1'].update(
        basis='continuity', evidence='/shots/2/start_state/prop_state')
    before = deepcopy((story, wire))
    decoded = protocol.decode_review_response(wire, protocol.review_targets(story), sources=continuity_sources(story))
    assert validate_planning_verdict(decoded, story) == []
    assert (story, wire) == before


def test_batch_sources_use_only_forward_external_context():
    from pipeline.planning_review_sources import continuity_sources
    story = recorded()['storyboard.json']['value']
    scoped = {**story, 'shots': [story['shots'][1]], 'review_scope': {
        'previous_shot': story['shots'][0], 'next_shot': story['shots'][2], 'is_final_batch': False}}
    sources = continuity_sources(scoped)
    assert '/review_scope/next_shot/start_state/prop_state' in sources['allowed']['2:3']
    assert not any('previous_shot' in path for path in sources['fields'])
    final = continuity_sources({**story, 'shots': [story['shots'][2]]})
    assert final['allowed']['3:1'] == []


def test_real_sdk_request_binds_citations_and_keeps_semantic_rejection(tmp_path, monkeypatch):
    from planning_protocol_cases import annotated_outcome, sourced_wire
    story = recorded()['storyboard.json']['value']
    wire = synthetic_wire(story)
    wire.update(requested_outcome_valid=False,
        reason='Offline negative annotation: incapacitation does not establish the requested death.')
    wire['units']['2:3']['result_admission']['decisions']['outcomes/1'].update(
        basis='continuity', evidence='/shots/2/start_state/prop_state')
    wire = sourced_wire(annotated_outcome(wire, story), story)
    sdk, calls = sdk_reviewer(monkeypatch, wire)
    with sdk:
        for _ in range(2):
            with pytest.raises(ValueError, match='incapacitation'):
                ensure_planning_semantics(story, tmp_path, save_to(tmp_path))
    assert len(calls) == 1
    payload = json.loads(calls[0]['messages'][-1]['content'])
    assert payload['version'] == protocol.VERSION
    assert payload['evidence_sources']['fields']['/shots/2/start_state/prop_state'] == story['shots'][2]['start_state']['prop_state']
    saved = json.loads((tmp_path / 'planning_semantic_review.json').read_text())
    assert saved['verdict'] == wire and not (tmp_path / 'result_admission.json').exists()


@pytest.mark.parametrize('action,following', [
    ('把零件交给同事', '同事接住零件并装入机架'),
    ('释放一个舞伴的手', '舞伴借旋转惯性继续滑步'),
    ('把杯子放在桌上', '客人拿起杯子喝水'),
])
def test_reference_binding_depends_on_source_order_not_genre_or_shot_number(action, following):
    from pipeline.planning_review_sources import continuity_sources
    story = {'shots': [
        {'shot_id': 42, 'action_beats': [{'action': action}, {'action': following}]},
        {'shot_id': 7, 'start_state': {'open_motion': following, 'lighting': '', 'prop_state': None},
         'action_beats': [{'action': following}]},
    ]}
    sources = continuity_sources(story)
    assert sources['fields']['/shots/1/start_state/open_motion'] == following
    assert '/shots/0/action_beats/1/action' in sources['allowed']['42:0']
    assert '/shots/0/action_beats/1/action' not in sources['allowed']['42:1']
    assert sources['allowed']['7:0'] == []
    assert all(isinstance(value, str) and value.strip() for value in sources['fields'].values())
    assert not any('/shots/42/' in path for path in sources['fields'])


@pytest.mark.asyncio
async def test_formal_entry_replays_original_failure_without_video_or_model_calls(tmp_path, monkeypatch):
    import config
    from pipeline.models import validate_storyboard
    from pipeline.orchestrator import VideoPipeline
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(config, 'LLM_MODEL', 'doubao-seed-2.1-lite')
    sources = recorded()
    story = validate_storyboard(sources['storyboard.json']['value'])
    def replay(**kwargs):
        kwargs['save_diagnostic']('planning_semantic_review.json', sources['planning_semantic_review.json']['value'])
        return deepcopy(story)
    monkeypatch.setattr('pipeline.orchestrator.generate_storyboard', replay)
    monkeypatch.setattr('pipeline.planning_review.OpenAI', lambda **kw: pytest.fail('review retried'))
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda **kw: pytest.fail('video stage reached'))
    pipeline = VideoPipeline(plan_only=True, paid_take_budget=0, take_policy='single_pass')
    with pytest.raises(ValueError, match='衔接依据'):
        await pipeline.run(story['production_plan']['director_intent']['source_request'])
    assert pipeline.run_workspace.manifest.status.value == 'failed'
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations
    assert not (pipeline.workspace / 'result_admission.json').exists()
