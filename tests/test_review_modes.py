"""模式改变交付决策，不改写观察，也不授权新增付费重试。"""
from copy import deepcopy
from dataclasses import replace

import pytest

from pipeline.models import RunOptions
from pipeline.review_policy import decide_review, timing_is_quality_only
from pipeline.semantic_review import SemanticReview
from test_scheduled_production_flow import scheduled_services


def reviewed(**changes):
    return replace(SemanticReview(accepted=True, required_entities_visible={'A': True},
        action_geometry_valid=True, primary_action_completed=True,
        observed_end_state={'location': 'room', 'subject': 'A', 'action_phase': 'completed'}), **changes)


@pytest.mark.parametrize('field', ['blocking_valid', 'composition_change_valid', 'screen_direction_valid', 'creative_brief_valid'])
def test_normal_allows_classified_quality_warning_without_changing_verdict(field):
    review = reviewed(accepted=False, **{field: False})
    before = deepcopy(review)
    assert not decide_review(review, 'strict').allowed
    assert decide_review(review, 'normal').allowed
    assert decide_review(review, 'normal').warnings
    assert review == before and review.accepted is False


@pytest.mark.parametrize('field', ['identity_continuity_valid', 'primary_action_completed',
    'requested_outcome_valid', 'action_handoff_valid', 'prop_continuity_valid',
    'reaction_causality_valid', 'action_geometry_valid', 'visual_invariants_valid'])
@pytest.mark.parametrize('claimed_pass', [False, True])
def test_serious_failure_cannot_be_hidden_by_normal_or_aggregate_pass(field, claimed_pass):
    review = reviewed(accepted=claimed_pass, blocking_valid=False, **{field: False})
    assert not decide_review(review, 'normal').allowed


def test_normal_does_not_treat_unknown_or_missing_handoff_as_minor():
    assert not decide_review(reviewed(accepted=False), 'normal').allowed
    assert not decide_review(reviewed(observed_end_state={}), 'normal').allowed
    assert not decide_review(reviewed(required_entities_visible={'A': False}), 'normal').allowed


@pytest.mark.parametrize('order,allowed', [('uncertain', True), ('passed', True), ('failed', False)])
def test_normal_uses_structured_temporal_classification(order, allowed):
    assessment = {'status': 'unavailable', 'failure_kind': 'semantic_uncertainty',
        'content_issues': [], 'timing': {'status': 'uncertain', 'causal_order': {'status': order}}}
    review = reviewed(accepted=False, action_timing_valid=False, event_assessment=assessment)
    assert decide_review(review, 'normal').allowed is allowed
    assessment['content_issues'] = ['observable missing cause']
    assert not timing_is_quality_only(assessment)


def test_mode_default_serialization_and_validation():
    assert RunOptions(request='historical').delivery_mode == 'reviewed'
    assert 'delivery_mode' not in RunOptions(request='historical').model_dump()
    assert RunOptions(request='new', delivery_mode='normal').model_dump()['delivery_mode'] == 'normal'
    for invalid in (None, True, {}, 'normal-ish'):
        with pytest.raises(ValueError):
            RunOptions(request='new', delivery_mode=invalid)


@pytest.mark.asyncio
@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
async def test_normal_finishes_reviewed_film_with_warnings_and_resumes_without_calls(scheduled_services, monkeypatch, topic):
    import json
    from pathlib import Path
    from pipeline.orchestrator import VideoPipeline
    from pipeline.run_state import RunWorkspace
    from web_server import _run_summary
    brief, services = scheduled_services(topic)
    create = services.create
    def warned(**request):
        response = create(**request)
        user = request['messages'][-1]['content']
        if isinstance(user, list) and 'Shot contract:' in user[0].get('text', ''):
            raw = json.loads(response.choices[0].message.content)
            raw['blocking_valid'] = False
            response.choices[0].message.content = json.dumps(raw)
        return response
    monkeypatch.setattr(services, 'create', warned)
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], delivery_mode='normal',
                             paid_take_budget=3, take_policy='single_pass')
    output = await pipeline.run(brief)
    assert Path(output).name == 'final.mp4' and Path(output).is_file()
    assert len(services.submissions) == len(services.reviews) == 3
    assert len(services.entries) == 2
    workspace = RunWorkspace.resume(pipeline.workspace)
    assert workspace.options.delivery_mode == 'normal'
    assert all(s.semantic_accepted is False and s.review_warnings
               and s.acceptance_policy == 'reviewed_with_warnings' for s in workspace.manifest.shots.values())
    report = json.loads((pipeline.workspace/'delivery_report.json').read_text())
    assert report['content_acceptance'] == 'with_warnings' and len(report['warnings']) >= 3
    assert _run_summary(pipeline.workspace)['review_warnings'] == report['warnings']
    counts = len(services.submissions), len(services.reviews), len(services.entries), len(services.author.calls)
    assert await VideoPipeline.from_workspace(pipeline.workspace).run() == output
    assert counts == (len(services.submissions), len(services.reviews), len(services.entries), len(services.author.calls))


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,field', [('reviewed', 'blocking_valid'), ('normal', 'identity_continuity_valid')])
async def test_strict_or_serious_failure_stops_without_another_paid_take(scheduled_services, monkeypatch, mode, field):
    import json
    from pipeline.orchestrator import VideoPipeline
    brief, services = scheduled_services('货物交接')
    create = services.create
    def rejected(**request):
        response = create(**request)
        user = request['messages'][-1]['content']
        if isinstance(user, list) and 'Shot contract:' in user[0].get('text', ''):
            raw = json.loads(response.choices[0].message.content)
            raw[field] = False
            response.choices[0].message.content = json.dumps(raw)
        return response
    monkeypatch.setattr(services, 'create', rejected)
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], delivery_mode=mode,
                             paid_take_budget=10, take_policy='completion_first')
    with pytest.raises(RuntimeError):
        await pipeline.run(brief)
    assert len(services.submissions) == 1
    assert not (pipeline.workspace/'final.mp4').exists()


def test_cli_and_web_select_normal_and_do_not_change_a_saved_mode(monkeypatch, tmp_path):
    import main
    from pipeline.orchestrator import VideoPipeline
    from pipeline.run_state import RunWorkspace
    from web_server import _parse_create_payload
    calls = []
    async def run(self, request):
        calls.append(self.delivery_mode)
        return 'offline.mp4'
    monkeypatch.setattr(VideoPipeline, 'run', run)
    for mode, expected in [('strict', 'reviewed'), ('normal', 'normal')]:
        monkeypatch.setattr('sys.argv', ['main.py', 'offline', '--delivery-mode', mode])
        main.main()
        assert calls[-1] == expected
        assert _parse_create_payload({'request': 'offline', 'delivery_mode': mode}).delivery_mode == expected
    workspace = RunWorkspace.create(tmp_path, RunOptions(request='offline', delivery_mode='normal'))
    assert VideoPipeline.from_workspace(workspace.path).delivery_mode == 'normal'
    original = workspace.manifest_path.read_bytes()
    monkeypatch.setattr('sys.argv', ['main.py', '--resume', str(workspace.path), '--delivery-mode', 'strict'])
    with pytest.raises(SystemExit):
        main.main()
    assert workspace.manifest_path.read_bytes() == original


@pytest.mark.asyncio
async def test_web_run_creation_persists_normal_mode_and_rejects_invalid_mode(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    import asyncio
    from test_web_server import FakePipeline
    from web_server import create_app, ACTIVE_RUNS_KEY
    captured = []
    def factory(**kwargs):
        captured.append(kwargs['delivery_mode'])
        return FakePipeline(**kwargs)
    async with TestClient(TestServer(create_app(output_root=tmp_path, pipeline_factory=factory))) as client:
        response = await client.post('/api/runs', json={'request': 'offline', 'delivery_mode': 'normal'})
        assert response.status == 202
        assert (await response.json())['options']['delivery_mode'] == 'normal'
        assert captured == ['normal']
        bad = await client.post('/api/runs', json={'request': 'offline', 'delivery_mode': 'bypass_all'})
        assert bad.status == 400 and captured == ['normal']
        await asyncio.gather(*list(client.server.app[ACTIVE_RUNS_KEY].values()))


def test_normal_sampling_uncertainty_returns_an_honest_warning_and_reuses_review_cache(tmp_path):
    import json
    from types import SimpleNamespace as NS
    from test_intent_bounded_execution import captured, bounded, RUN, FIXTURE
    from test_full_production_flow import completion
    from pipeline.semantic_review import SemanticTakeReviewer
    if not (RUN/'shots/shot_001.mp4').exists():
        pytest.skip('本机原片回放；纯策略和跨场景流程仍执行')
    story, _, observation = captured()
    support = deepcopy(FIXTURE['support'])
    # 合成标注隔离粗采样路径，不把原片重新批准。
    for row in support['decisions'].values():
        row.update(requirement='fulfilled', issue='none', finding='Offline functional-result annotation.')
    calls = []
    def create(**request):
        calls.append(request)
        return completion(observation if len(calls) == 1 else support)
    reviewer = SemanticTakeReviewer(tmp_path, model='offline', review_mode='normal',
        client=NS(chat=NS(completions=NS(create=create))))
    for _ in range(2):
        result = reviewer.review(str(RUN/'shots/shot_001.mp4'), bounded(story['shots'][0]),
                                 identity_crop_entities=story['shots'][0]['characters'])
        assert result.accepted is False
        assert decide_review(result, 'normal').allowed
        assert not decide_review(result, 'strict').allowed
    assert len(calls) == 2
    saved = json.loads(next(reviewer.cache_dir.glob('*.json')).read_text())
    assert saved['review']['accepted'] is False


@pytest.mark.parametrize('problem', ['sampling', 'reversed', 'missing_result'])
def test_film_gate_reuses_policy_and_never_waives_core_or_reversed_causality(tmp_path, problem):
    import hashlib
    import json
    from dataclasses import asdict
    from pipeline.causality import compile_action_contract
    from pipeline.event_assessment import assess_event_evidence
    from pipeline.film_rhythm import ensure_observed_film_rhythm
    from pipeline.generator import ShotResult
    from pipeline.run_state import RunWorkspace
    from pipeline.semantic_review import SemanticReviewUnavailableError
    from test_performance_outcomes import outcome_shot, outcome_response
    shot = outcome_shot()
    shot['execution_semantics'] = 'intent-bounded-v1'
    shot['production_slot'].update(min_interaction_seconds=0, result_hold_seconds=0)
    contract = compile_action_contract(shot)
    events = outcome_response()['action_event_evidence']
    events[0]['observed_reaction_seconds'] = events[0]['observed_effect_seconds']
    if problem == 'reversed':
        events[0]['observed_reaction_seconds'] = .1
    if problem == 'missing_result':
        events[0]['result_visible'] = False
    times = [i/4 for i in range(21)]
    assessment = assess_event_evidence(contract, events, timestamps=times)
    review = reviewed(accepted=False, action_timing_valid=False,
        action_sequence_valid=not assessment['issues'], event_assessment=assessment)
    video = tmp_path/'offline-evidence.mp4'
    video.write_bytes(b'Synthetic offline ledger; not a model-generated clip.')
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    cache = tmp_path/'semantic_reviews'/f'{digest}.json'
    cache.parent.mkdir()
    RunWorkspace._atomic_write_json(cache, {'video_hash': digest, 'action_contract': asdict(contract),
        'review': asdict(review), 'action_event_evidence': events, 'media_duration': 5, 'sample_timestamps': times})
    result = ShotResult(shot_id=shot['shot_id'], status='success', local_path=str(video),
        semantic_accepted=False, acceptance_policy='reviewed_with_warnings', review_warnings=['sample uncertainty'],
        accepted_contract_fingerprint=contract.fingerprint())
    story = {'shots': [shot], 'production_plan': {'min_interaction_seconds': 0}}
    with pytest.raises(SemanticReviewUnavailableError):
        ensure_observed_film_rhythm(story, [result], tmp_path)
    if problem == 'sampling':
        assert ensure_observed_film_rhythm(story, [result], tmp_path, review_mode='normal')
        report = json.loads((tmp_path/'film_rhythm_review.json').read_text())
        assert report['status'] == 'uncertain' and report['delivery_allowed'] is True
    else:
        with pytest.raises((SemanticReviewUnavailableError, RuntimeError)):
            ensure_observed_film_rhythm(story, [result], tmp_path, review_mode='normal')


@pytest.mark.asyncio
async def test_normal_partial_resume_reuses_warning_take_and_actual_continuity(scheduled_services, monkeypatch):
    import json
    from pipeline.orchestrator import VideoPipeline
    from pipeline.generator import VideoGenerator
    from pipeline.semantic_review import SemanticReviewUnavailableError
    brief, services = scheduled_services('双人舞')
    create = services.create
    def warned(**request):
        response = create(**request)
        user = request['messages'][-1]['content']
        if isinstance(user, list) and 'Shot contract:' in user[0].get('text', ''):
            raw = json.loads(response.choices[0].message.content)
            raw['blocking_valid'] = False
            response.choices[0].message.content = json.dumps(raw)
        return response
    monkeypatch.setattr(services, 'create', warned)
    original = VideoGenerator._generate_single_shot
    async def interrupted(self, shot, *args, **kwargs):
        if shot['shot_id'] == 2:
            raise SemanticReviewUnavailableError('Offline interruption before the second submission')
        return await original(self, shot, *args, **kwargs)
    monkeypatch.setattr(VideoGenerator, '_generate_single_shot', interrupted)
    pipeline = VideoPipeline(resolution='480p', platforms=['youtube'], delivery_mode='normal',
                             paid_take_budget=3, take_policy='single_pass')
    with pytest.raises(SemanticReviewUnavailableError):
        await pipeline.run(brief)
    assert len(services.submissions) == len(services.reviews) == 1
    monkeypatch.setattr(VideoGenerator, '_generate_single_shot', original)
    await VideoPipeline.from_workspace(pipeline.workspace).run()
    assert len(services.submissions) == len(services.reviews) == 3
    assert len(services.entries) == 2
