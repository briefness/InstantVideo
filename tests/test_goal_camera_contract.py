"""真实失败证据不可改写；新协议离线重放只改变事实的归属，不创造观察。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pipeline.causality import compile_action_contract
from pipeline.event_assessment import assess_event_evidence
from pipeline.generator import VideoGenerator
from pipeline.result_observations import decode_observations
from pipeline.semantic_review import SemanticReview


def capture():
    return json.loads((Path(__file__).parent / 'fixtures/real_goal_camera_conflict.json').read_text())


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))
    monkeypatch.setattr(socket.socket, 'connect_ex', lambda *_: pytest.fail('offline only'))


def test_original_failure_is_preserved_and_local_source_hashes_match():
    record = capture()
    assert record['review']['accepted'] is False
    assert record['review']['primary_action_completed'] is True
    assert record['review']['composition_change_valid'] is False
    assert record['action_event_evidence'][0]['target_reaction_visible'] is False
    assert record['generation']['image_role'] == 'first_frame'
    from pipeline.causality import CompiledActionContract
    old = CompiledActionContract(**record['original_action_contract'])
    assert assess_event_evidence(old, record['action_event_evidence'])['issues'] == [
        '事件1未满足 target_reaction_visible']
    assert old.fingerprint() != compile_action_contract(record['shots'][1]).fingerprint()
    root = Path(__file__).parents[1] / record['source_run']
    if root.exists():
        for name, digest in record['source_hashes'].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_function_owns_completion_not_each_preparation_but_camera_still_rejects():
    record = capture()
    contract = compile_action_contract(record['shots'][1])
    fields = contract.event_evidence_schema()['items']['properties']
    assert 'target_reaction_visible' not in fields
    assert 'effect_visible' not in fields
    assert {'causal_order_valid', 'scope_valid', 'result_visible'} <= fields.keys()
    # Unmodified old response is decoded using its original schema, not rewritten to pass.
    original_catalog = record['original_action_contract']['review_projection']['completion_contract']
    decoded = decode_observations(record['observation_response'],
        original_catalog, record['observation_schema'], 25)
    assert decoded['primary_action_completed'] is True
    assert assess_event_evidence(contract, decoded['action_event_evidence'])['status'] == 'passed'
    assert SemanticReview.from_dict(decoded, require_composition=True).accepted is False
    assert record['review']['accepted'] is False


@pytest.mark.parametrize('failure', ['goal', 'causal_order_valid', 'scope_valid', 'unknown'])
def test_missing_work_wrong_cause_and_unknown_never_become_success(failure):
    record = capture()
    contract = compile_action_contract(record['shots'][1])
    raw = deepcopy(record['observation_response'])
    catalog = record['original_action_contract']['review_projection']['completion_contract']
    key = next(k for k in raw['observations'] if k.startswith('/intent/'))
    if failure in ('goal', 'unknown'):
        raw['observations'][key]['status'] = 'unknown' if failure == 'unknown' else 'violated'
    else:
        raw['action_event_evidence'][0][failure] = False
    if failure == 'unknown':
        with pytest.raises(ValueError, match='观察不足'):
            decode_observations(raw, catalog, record['observation_schema'], 25)
        return
    decoded = decode_observations(raw, catalog, record['observation_schema'], 25)
    assert assess_event_evidence(contract, decoded['action_event_evidence'])['status'] == 'rejected'


@pytest.mark.parametrize('transition,role', [('intentional_cut', 'reference_image'), ('seamless', 'first_frame')])
def test_reference_role_tracks_editing_boundary_not_story_topic(tmp_path, transition, role):
    record = capture()
    previous, shot = record['shots']
    shot['continuity_from_previous'] = transition
    gen = VideoGenerator(output_dir=str(tmp_path), review_content=False)
    refs, actual = gen._build_image_refs(shot, '/accepted/tail.jpg', previous)
    assert actual == role
    assert refs[0] == '/accepted/tail.jpg'
    assert gen._build_state_only_refs(shot, '/accepted/tail.jpg', previous) == (refs, role)


def test_cut_does_not_overwrite_new_camera_with_previous_observation():
    previous, shot = capture()['shots']
    planned = deepcopy(shot['start_state'])
    previous['observed_end_state'] = {**previous['end_state'], 'camera': 'wide static old camera',
                                    'prop_state': 'observed actual objects'}
    VideoGenerator._reconcile_start_state(shot, previous)
    assert shot['start_state']['camera'] == planned['camera']
    assert shot['start_state']['prop_state'] == 'observed actual objects'


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
@pytest.mark.parametrize('status', ['satisfied', 'violated', 'unknown'])
def test_other_interactions_keep_required_work_and_explicit_methods(topic, status):
    from test_creation_schedule_flow import scenario_story
    from test_staging_direction import compile_story
    from test_goal_allocation import bind, evidence
    from test_goal_authority import annotated_snapshot
    from test_functional_results import format_for
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    _, _, raw, plan = scenario_story(topic)
    # 独立场景标注：用户明确指定这些动作，不能把必要方法也当成随意表演。
    goals = ['; '.join(u['action'] for u in s['action_plan'] if u['kind'] == 'interaction')
             for s in raw['shots']]
    plan['director_intent'].update(authoring_protocol='scene-units-v1',
        source_request=plan['director_intent']['source_request'] + '; '.join(goals),
        completion_goals=[{'goal': goal, 'basis': 'user', 'evidence': goal} for goal in goals])
    story = compile_story(raw, plan)
    for s in story['shots']:
        s['result_admission'] = 'review-required-v1'
    judgments = {f'/intent/completion_goals/{i}': {'parts': [evidence(goal,
        *(f"{s['shot_id']}:{j}" for j in range(len(s['action_beats']))))]}
        for i, (s, goal) in enumerate(zip(story['shots'], goals))}
    admitted = bind(story, judgments)
    shot = admitted['shots'][1]
    original = deepcopy(shot['action_beats'])
    contract = compile_action_contract(shot)
    catalog = contract.review_projection['completion_contract']
    schema = format_for(shot)['json_schema']['schema']
    value = annotated_snapshot(shot)
    for key in value['observations']:
        criterion = catalog['criteria'][key]
        value['observations'][key]['status'] = (status if criterion['kind'] == 'completion' else
            'satisfied' if shot['action_beats'][criterion['beat_id'] - 1]['activity'] == 'interaction' else 'violated')
    if status == 'unknown':
        with pytest.raises(ValueError, match='观察不足'):
            decode_observations(value, catalog, schema, 25)
    else:
        result = decode_observations(value, catalog, schema, 25)
        assert result['primary_action_completed'] is (status == 'satisfied')
        assert all(e['result_visible'] is (status == 'satisfied') for e in result['action_event_evidence'])
    for prompt in (compile_normal_provider_prompt(shot, admitted, has_observed_start=True),
                   compile_policy_safe_prompt(shot, storyboard=admitted, has_state_reference=True,
                                              image_role='reference_image', reference_count=1)):
        assert goals[1] in prompt
    assert shot['action_beats'] == original


def test_reference_boundaries_identity_capacity_and_privacy():
    from pipeline.reference_binding import resolve_image_references
    shot = {'scene_id': 'stage', 'continuity_from_previous': 'intentional_cut'}
    previous = {'scene_id': 'stage'}
    refs, role = resolve_image_references(shot, 'tail', previous, ['actor', 'actor'])
    assert (refs, role) == (['tail', 'actor'], 'reference_image')
    assert not VideoGenerator._has_removable_identity_refs(['tail'], role, has_state_reference=True)
    assert VideoGenerator._has_removable_identity_refs(refs, role, has_state_reference=True)
    with pytest.raises(ValueError, match='9 张'):
        resolve_image_references(shot, 'tail', previous, [f'actor{i}' for i in range(9)])
    assert resolve_image_references({**shot, 'continuity_from_previous': 'seamless'}, 'tail', previous,
        [f'actor{i}' for i in range(10)]) == (['tail'], 'first_frame')
    refs, role = resolve_image_references({**shot, 'scene_id': 'new_stage'}, 'tail', previous, ['actor'])
    assert (refs, role) == (['actor'], 'reference_image')


def test_reference_chain_uses_actual_image_provenance_not_api_role():
    gen = object.__new__(VideoGenerator)
    previous = {'scene_id': 'room', 'output_reference_depth': 2}
    for role in ('first_frame', 'reference_image'):
        assert gen._next_reference_depth({}, previous, '/tail.jpg', reference_role=role,
            reference_fingerprints=[gen._reference_fingerprint('/tail.jpg')]) == 3
    assert gen._next_reference_depth({}, previous, '/tail.jpg', reference_role='reference_image',
        reference_fingerprints=[gen._reference_fingerprint('/canonical.jpg')]) == 0


def test_continuation_preserves_current_camera_and_reuses_actual_state(tmp_path):
    from pipeline.continuation import prepare_continuation
    from test_continuation import scenario
    from test_director_intent import Responses
    previous, shot = capture()['shots']
    _, observed, tail = scenario(tmp_path)
    previous['observed_end_state'] = {**observed['observed_end_state'], 'camera': 'old wide framing'}
    original = deepcopy(shot)
    sdk = Responses({'compatible': True, 'entry_action': 'Continue from the actual state.',
                     'reason': 'independent offline compatibility annotation'})
    adapted = prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk)
    assert adapted['start_state']['camera'] == original['start_state']['camera']
    assert adapted['start_state']['prop_state'] == previous['observed_end_state']['prop_state']
    assert adapted['camera'] == original['camera'] and adapted['action_beats'] == original['action_beats']
    assert prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk, allow_create=False) == adapted
    assert len(sdk.calls) == 1 and shot == original


def test_original_video_replayed_through_review_still_fails_camera_without_retries(tmp_path):
    from types import SimpleNamespace
    import jsonschema
    from pipeline.semantic_review import SemanticTakeReviewer
    from test_full_production_flow import completion
    record = capture()
    root = Path(__file__).parents[1] / record['source_run']
    video = root / 'shots/shot_002_rejected_1.mp4'
    tail = root / 'shots/shot_001_lastframe.jpg'
    if not video.is_file() or not tail.is_file():
        pytest.skip('original local media unavailable; JSON replay remains covered')
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        schema = kwargs['response_format']['json_schema']['schema']
        # 声明为离线协议投影：保留原观察事实，不构造新的模型肯定或更改视频。
        if 'decisions' in schema['properties']:
            from functional_observation_cases import annotated_support_response
            return completion(annotated_support_response(kwargs))
        value = deepcopy(record['observation_response'])
        value['observations'] = {k: v for k, v in value['observations'].items()
                                 if k in schema['properties']['observations']['properties']}
        fields = schema['properties']['action_event_evidence']['items']['properties']
        value['action_event_evidence'] = [{k: v for k, v in e.items() if k in fields}
                                         for e in value['action_event_evidence']]
        value['identity_crop_boxes'] = []
        assert [(e.json_path, e.message) for e in jsonschema.Draft202012Validator(schema).iter_errors(value)] == []
        assert 'independent completion obligations' in kwargs['messages'][0]['content']
        return completion(value)
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-record-projection',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    for _ in range(2):
        result = reviewer.review(str(video), record['shots'][1], previous_frame_path=str(tail),
            boundary_context={'same_scene': True, 'state_reference_role': 'reference_image'})
        assert result.primary_action_completed is True and result.action_sequence_valid is True
        assert result.composition_change_valid is False and result.accepted is False
    assert len(calls) == 2
    assert hashlib.sha256(video.read_bytes()).hexdigest() == record['video_sha256']
