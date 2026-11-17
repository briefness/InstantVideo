"""Captured inputs stay immutable; offline annotations are not fresh model judgments."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.action_results import required_outcomes, functional_outcomes
from pipeline.causality import compile_action_contract
from pipeline.result_admission import apply_result_admission
from pipeline.semantic_review import _review_response_format, SemanticTakeReviewer, SemanticReviewUnavailableError


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def captured(*, fresh=True):
    r = json.loads((Path(__file__).parent / 'fixtures/real_pose_requirement_failure.json').read_text())
    sources = {}
    for name, source in r['sources'].items():
        assert hashlib.sha256(source['text'].encode()).hexdigest() == source['sha256']
        sources[name] = json.loads(source['text'])
    story = sources['storyboard.json']
    if fresh:
        from pipeline.action_authoring import compile_authored_storyboard
        from pipeline.storyboard import _compile_storyboard_contract
        plan = story['production_plan']
        story = _compile_storyboard_contract(compile_authored_storyboard(sources['planning_authoring.json'], plan),
            story['aspect_ratio'], story['resolution'], story['style'], production_plan=plan)
    return r, sources, apply_result_admission(story, sources['result_admission.json']['units'])


def format_for(shot):
    return _review_response_format(required_entity_count=3, identity_crop_count=0,
        action_contract=compile_action_contract(shot), sample_count=25,
        causal_review=True, evidence_review=False, narrative_review=True, blocking_review=True,
        composition_review=False, invariant_review=True, boundary_review=False, state_handoff_review=False)


def test_original_response_is_self_inconsistent_and_cannot_be_relabelled():
    _, sources, old = captured(fresh=False)
    raw = next(v for k, v in sources.items() if '/responses/' in k)
    response = json.loads(raw['attempts'][0]['choices'][0]['content'])
    assert '未完全从杯边收回' in response['failure_reason']
    assert '手从杯边收回' in response['observed_end_state']['pose_and_gaze']
    assert response['action_event_evidence'][0]['result_visible'] is False
    assert len(required_outcomes(old['shots'][0]['action_beats'][2])) == 5


def test_response_realization_is_not_a_second_fixed_state_obligation():
    _, sources, story = captured()
    shot = story['shots'][0]
    beat = shot['action_beats'][2]
    original = sources['result_admission.json']['execution_storyboard']['shots'][0]['action_beats'][2]
    assert all(c['entity'] != 'Lin_Ayi' for c in required_outcomes(beat))
    assert original['outcomes'][0] in functional_outcomes(beat)
    assert original['outcomes'][1] in required_outcomes(beat)
    assert beat['action'] == original['action'] and beat['outcomes'] == original['outcomes']


def test_wire_has_one_observation_owner_for_results_and_handoff():
    _, _, story = captured()
    props = format_for(story['shots'][0])['json_schema']['schema']['properties']
    assert 'accepted' not in props and 'failure_reason' not in props
    assert 'primary_action_completed' not in props and 'observed_end_state' not in props
    assert 'observations' in props and 'end_state_sources' in props


def test_original_contradictory_wire_is_not_accepted_as_new_evidence(tmp_path):
    record, sources, story = captured()
    video = Path(__file__).resolve().parents[1] / record['video']['path']
    if not video.is_file():
        pytest.skip('original local video unavailable; raw JSON replay remains covered')
    calls = []
    raw = next(v for k, v in sources.items() if '/responses/' in k)['attempts'][0]['choices'][0]['content']
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=raw, refusal=None))])
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-captured',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    for _ in range(2):
        with pytest.raises(SemanticReviewUnavailableError):
            reviewer.review(str(video), story['shots'][0])
    assert len(calls) == 1  # Invalid observations do not turn into another model attempt or a paid retake.
    assert hashlib.sha256(video.read_bytes()).hexdigest() == record['video']['sha256']


@pytest.mark.parametrize('topic', ['family', 'ensemble', 'product'])
def test_non_interaction_scenes_also_have_completion_evidence(topic):
    from test_ordinary_scene_contract import scene_case
    from test_staging_direction import compile_story
    from test_action_result_flow import annotated_verdict
    from pipeline.production_plan import build_production_plan
    from pipeline.models import DirectorIntentSpec
    brief, reply, identity, _, shots = scene_case(topic)
    # 检验无目标目录的历史 functional-v1 合同，不能被新版共享服务夹具升级。
    reply.pop('completion_goals', None)
    intent = DirectorIntentSpec.model_validate({**reply, 'source_request': brief,
        'authoring_protocol': 'scene-units-v1'}).model_dump()
    plan = build_production_plan(intent['content_focus'], 24, intent=intent)
    for shot in shots:
        shot['result_admission'] = 'review-required-v1'
    story = compile_story({**identity, 'shots': shots}, plan)
    story = apply_result_admission(story, annotated_verdict(story)['units'])
    for shot in story['shots']:
        contract = compile_action_contract(shot)
        assert not contract.review_projection.get('action_events')
        criteria = contract.review_projection['completion_contract']['criteria']
        assert len(criteria) == len(shot['action_beats'])
        wire = annotated_wire(format_for(shot)['json_schema']['schema'])
        wire['observations'][next(iter(criteria))]['status'] = 'violated'
        assert decode(shot, wire)['primary_action_completed'] is False


def test_explicitly_requested_pose_keeps_its_authority():
    _, sources, story = captured()
    units = deepcopy(sources['result_admission.json']['units'])
    pose = 'The performer must finish with both arms raised.'
    story['production_plan']['director_intent']['source_request'] += pose
    for shot in story['shots']:
        shot['result_admission'] = 'review-required-v1'
    story['shots'][0]['action_beats'][2]['outcomes'][0]['state'] = pose
    decision = next(u for u in units if u['shot_id'] == 1 and u['beat_index'] == 2)['result_admission']['decisions'][0]
    decision.update(basis='user', evidence=pose)
    shot = apply_result_admission(story, units)['shots'][0]
    assert any(c['state'] == pose for c in required_outcomes(shot['action_beats'][2]))
    wire = annotated_wire(format_for(shot)['json_schema']['schema'])
    wire['observations']['beat:3/outcomes/0'].update(status='violated', observation='Both arms remain lowered.')
    assert decode(shot, wire)['accepted'] is False


def annotated_wire(schema):
    """Synthetic observations for protocol tests, not labels for the captured film."""
    def example(node):
        kind = node.get('type')
        if 'enum' in node:
            return node['enum'][0]
        if kind == 'object':
            return {k: example(v) for k, v in node['properties'].items()}
        if kind == 'array':
            return [example(node['items']) for _ in range(node.get('minItems', 0))]
        if kind in ('boolean', ['boolean', 'null']):
            return True
        if kind in ('number', 'integer'):
            return node.get('minimum', 0)
        return 'offline synthetic observation'
    value = example(schema)
    if value['observations']:
        count = next(iter(schema['properties']['observations']['properties'].values()))['properties']['sample_indices']['items']['maximum']
        for witness in value['observations'].values():
            witness.update(status='satisfied', sample_indices=[1, count])
        last = next(reversed(value['observations']))
        value['end_state_sources'] = {k: [last] for k in value['end_state_sources']}
    event_schema = schema['properties'].get('action_event_evidence')
    if event_schema:
        for item, event_id in zip(value['action_event_evidence'], event_schema['items']['properties']['event_id']['enum']):
            item['event_id'] = event_id
    return value


def decode(shot, value):
    from pipeline.result_observations import decode_observations
    return decode_observations(value, compile_action_contract(shot).review_projection['completion_contract'],
                               format_for(shot)['json_schema']['schema'], 25)


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_every_function_is_required_without_locking_its_pose(topic):
    from test_creation_schedule_flow import scenario_story
    from test_staging_direction import compile_story
    from test_action_result_flow import annotated_verdict
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    _, _, raw, plan = scenario_story(topic)
    for source in raw['shots']:
        source['result_admission'] = 'review-required-v1'
    story = compile_story(raw, plan)
    story = apply_result_admission(story, annotated_verdict(story)['units'])
    for shot in story['shots']:
        schema = format_for(shot)['json_schema']['schema']
        wire = annotated_wire(schema)
        assert decode(shot, wire)['primary_action_completed'] is True
        for index, beat in enumerate(shot['action_beats'], 1):
            missing = deepcopy(wire)
            missing['observations'][f'beat:{index}/completion'].update(
                status='violated', observation='The action is attempted but does not complete.')
            rejected = decode(shot, missing)
            assert not rejected['accepted'] and not rejected['primary_action_completed']
            assert f'beat:{index}/completion' in rejected['failure_reason']
            if beat['activity'] == 'interaction':
                assert next(e for e in rejected['action_event_evidence'] if e['event_id'] == index)['result_visible'] is False
                for claim in functional_outcomes(beat):
                    assert claim['state'] in compile_normal_provider_prompt(shot, story, has_observed_start=False)
                    assert claim['state'] in compile_policy_safe_prompt(shot, has_state_reference=False,
                        image_role=None, reference_count=0)


def test_user_state_is_still_required_and_only_bound_witness_supplies_handoff():
    _, _, story = captured()
    shot = story['shots'][0]
    wire = annotated_wire(format_for(shot)['json_schema']['schema'])
    key = 'beat:3/outcomes/1'  # Original explicit user requirement, not the actor's incidental pose.
    wire['observations'][key].update(status='violated', observation='Object remains in the hand; transfer has not completed.')
    wire['end_state_sources']['prop_state'] = [key]
    verdict = decode(shot, wire)
    assert not verdict['accepted'] and not verdict['action_event_evidence'][0]['result_visible']
    assert verdict['observed_end_state']['prop_state'] == wire['observations'][key]['observation']
    assert wire['observations'][key]['observation'] in verdict['failure_reason']


@pytest.mark.parametrize('mutation', ['missing', 'unknown', 'duplicate', 'out_of_range', 'bool_index',
    'early_handoff', 'foreign_handoff', 'aggregate', 'event_result', 'new_end_state'])
def test_unbound_or_duplicate_observations_are_unavailable(mutation):
    _, _, story = captured()
    shot = story['shots'][0]
    wire = annotated_wire(format_for(shot)['json_schema']['schema'])
    key = next(iter(wire['observations']))
    if mutation == 'missing':
        wire['observations'].pop(key)
    elif mutation == 'unknown':
        wire['observations'][key]['status'] = 'unknown'
    elif mutation in ('duplicate', 'out_of_range', 'bool_index'):
        wire['observations'][key]['sample_indices'] = {'duplicate': [1, 1], 'out_of_range': [26], 'bool_index': [True]}[mutation]
    elif mutation == 'early_handoff':
        wire['observations'][key]['sample_indices'] = [1]
        wire['end_state_sources']['action_phase'] = [key]
    elif mutation == 'foreign_handoff':
        wire['end_state_sources']['action_phase'] = ['invented-key']
    elif mutation == 'aggregate':
        wire['accepted'] = True
    elif mutation == 'event_result':
        wire['action_event_evidence'][0]['result_visible'] = True
    else:
        wire['observed_end_state'] = {'pose_and_gaze': 'a contradictory second account'}
    with pytest.raises(ValueError):
        decode(shot, wire)


@pytest.mark.parametrize('status', ['satisfied', 'violated', 'unknown'])
def test_real_media_sampling_with_synthetic_witnesses_and_durable_cache(tmp_path, status):
    record, _, story = captured()
    video = Path(__file__).resolve().parents[1] / record['video']['path']
    if not video.is_file():
        pytest.skip('captured original media is local only')
    calls = []
    def create(**request):
        import jsonschema
        calls.append(request)
        schema = request['response_format']['json_schema']['schema']
        if 'decisions' in schema['properties']:
            from functional_observation_cases import annotated_support_response
            value = annotated_support_response(request)
        else:
            value = annotated_wire(schema)
            value['observations']['beat:3/completion'].update(status=status,
                observation='Offline test witness, not an assessment of this footage.')
        jsonschema.validate(value, schema)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content=json.dumps(value), refusal=None))])
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-synthetic',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    for _ in range(2):
        if status == 'unknown':
            with pytest.raises(SemanticReviewUnavailableError):
                reviewer.review(str(video), story['shots'][0])
        else:
            result = reviewer.review(str(video), story['shots'][0])
            assert result.accepted == (status == 'satisfied')
    assert len(calls) == (2 if status == 'satisfied' else 1)
    assert hashlib.sha256(video.read_bytes()).hexdigest() == record['video']['sha256']
