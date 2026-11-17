"""真实请求/原片和跨场景数据流；替身判定不是视觉准确率证据。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_goal_allocation import offline

ROOT = Path(__file__).parents[1] / 'output/20261109_202658_928234'


def captured():
    story = json.loads((ROOT / 'result_admission.json').read_text())['execution_storyboard']
    continuation = json.loads((ROOT / 'continuations/shot_002.json').read_text())
    return story, continuation


def test_original_scope_and_state_loss_reproduces_but_new_execution_does_not():
    from pipeline.provider_prompt import compile_normal_provider_prompt
    story, record = captured()
    first = json.loads((ROOT / 'submission_requests/shot_001_attempt_001.json').read_text())['prompt']
    second = json.loads((ROOT / 'submission_requests/shot_002_attempt_001.json').read_text())['prompt']
    criteria = record['input']['completion_contract']['criteria']
    joint = next(c['action'] for c in criteria.values() if c.get('execution_scope'))
    state = record['input']['observed_state']
    assert joint in first and state['prop_state'] not in second
    new = deepcopy(story['shots'][0])
    new['generation_execution'] = 'scene-instructions-v2'
    prompt = compile_normal_provider_prompt(new, story, has_observed_start=False)
    assert joint not in prompt
    assert all(beat['action'] in prompt for beat in new['action_beats'])


def test_continuation_preserves_actual_origin_and_does_not_reinject_wrong_advice(tmp_path):
    from pipeline.continuation import prepare_continuation
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    from test_director_intent import Responses
    story, record = captured()
    shot = deepcopy(story['shots'][1]); shot['generation_execution'] = 'scene-instructions-v2'
    before = deepcopy(shot)
    previous = {**story['shots'][0], 'observed_end_state': record['input']['observed_state']}
    tail = ROOT / 'shots/shot_001_lastframe.jpg'
    client = Responses(deepcopy(record['plan']))
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=client)
    assert shot == before and result['action_beats'] == before['action_beats']
    assert result['camera'] == before['camera']
    assert not result.get('observed_entry_path')
    assert result['observed_origin']['state'] == previous['observed_end_state']
    for prompt in (compile_normal_provider_prompt(result, story, has_observed_start=True),
                   compile_policy_safe_prompt(result, storyboard=story, has_state_reference=True,
                                              image_role='reference_image', reference_count=3)):
        assert previous['observed_end_state']['prop_state'] in prompt
        assert 'medium shot' in prompt
    assert prepare_continuation(shot, previous, str(tail), tmp_path, client=client, allow_create=False) == result
    assert len(client.calls) == 1


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_formal_compiler_and_runtime_preserve_observed_state_across_scenes(tmp_path, topic):
    from test_creation_schedule_flow import scenario_story
    from test_staging_direction import compile_story
    from test_view_execution import view_case
    from test_director_intent import Responses
    from pipeline.continuation import prepare_continuation
    from pipeline.provider_prompt import compile_normal_provider_prompt
    _, _, raw, plan = scenario_story(topic)
    story = compile_story(raw, plan)
    assert all(s['generation_execution'] == 'scene-instructions-v2' for s in story['shots'])
    shot, previous, tail, _, _ = view_case(tmp_path)
    shot['generation_execution'] = 'scene-instructions-v2'
    previous['observed_end_state']['prop_state'] = f'{topic}: actual endpoint recorded only by observation'
    client = Responses({'compatible': True, 'reason': 'Offline fixture', 'entry_action': None, 'view_window': None})
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=client)
    prompt = compile_normal_provider_prompt(result, story, has_observed_start=True)
    assert previous['observed_end_state']['prop_state'] in prompt
    from pipeline.result_observations import completion_contract
    criteria = completion_contract(result)['criteria']
    assert criteria['boundary/physical_progress']['previous_observed_state'] == previous['observed_end_state']
    assert criteria['boundary/camera_change']['camera'] == shot['camera']


def test_actual_camera_counterevidence_survives_unresolved_action_review(tmp_path):
    from pipeline.semantic_review import SemanticTakeReviewer, SemanticReviewUnavailableError
    from test_full_production_flow import completion
    story, record = captured()
    digest = '6fcb69d86b14d9ad33b00d061f7bf40dbc3f35dfc837c2ce44c5a9033b0c3f70'
    source = json.loads((ROOT / f'semantic_reviews/{digest}.evidence.json').read_text())
    raw = source['response']
    support = json.loads((ROOT / 'semantic_reviews/support/01bb84a7fda14a3f2d8b0a20f6ecc72c22f4902aa8c84341a91e596117bbda0e.json').read_text())
    assert hashlib.sha256((ROOT / 'shots/shot_002.mp4').read_bytes()).hexdigest() == digest
    calls = []
    def create(**request):
        calls.append(request)
        return completion(deepcopy(raw) if isinstance(request['messages'][-1]['content'], list)
                          else json.loads(support['response']['choices'][0]['content']))
    reviewer = SemanticTakeReviewer(tmp_path, model='offline',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    for _ in range(2):
        with pytest.raises(SemanticReviewUnavailableError, match='语义不确定'):
            reviewer.review(str(ROOT / 'shots/shot_002.mp4'), record['execution_shot'],
                previous_frame_path=str(ROOT / 'shots/shot_001_lastframe.jpg'),
                boundary_context={'same_scene': True, 'state_reference_role': 'reference_image',
                                  'previous_observed_end_state': record['input']['observed_state']})
    saved = json.loads((reviewer.cache_dir / f'{digest}.evidence.json').read_text())
    assert saved['composition_measurement']['status'] == 'unchanged'
    assert saved['independent_findings']['composition_change_valid'] is False
    assert raw['composition_change_valid'] is True and saved['response'] == raw
    assert len(calls) == 2


@pytest.mark.parametrize('domain', ['continuity', 'camera'])
def test_boundary_observation_overrides_positive_aggregate_without_erasing_completed_actions(tmp_path, domain):
    from pipeline.continuation import prepare_continuation
    from pipeline.result_observations import completion_contract, observation_properties, decode_observations
    from test_director_intent import Responses
    story, stored = captured()
    shot = deepcopy(story['shots'][1]); shot['generation_execution'] = 'scene-instructions-v2'
    previous = {**story['shots'][0], 'observed_end_state': stored['input']['observed_state']}
    result = prepare_continuation(shot, previous, str(ROOT / 'shots/shot_001_lastframe.jpg'), tmp_path,
                                 client=Responses(stored['plan']))
    catalog = completion_contract(result)
    facts = {
        'continuity': 'The prior endpoint has the package on the floor. The current opening resets it to the table without a visible move.',
        'camera': 'Both sides show the same wide frame; only the performer changes posture.',
    }
    value = {'observations': {k: {'status': 'violated' if c.get('domain') == domain else 'satisfied',
                                'observation': facts.get(c.get('domain'), 'Local contribution performed.'),
                                'sample_indices': [1, 2]} for k, c in catalog['criteria'].items()
                              if c['kind'] != 'realization'},
             'end_state_context': {k: v for k, v in stored['input']['observed_state'].items()
                                   if k in ('location', 'subject', 'camera', 'screen_direction', 'lighting')},
             'terminal_observation': {'sample_index': 2, 'status': 'observed', 'action_phase': 'aftermath',
                 'pose_and_gaze': 'People remain.', 'prop_state': 'Package is on the floor.', 'open_motion': ''},
             'boundary_continuity_valid': True, 'composition_change_valid': True}
    schema = {'properties': {k: {} for k in value}}
    schema['properties']['end_state_context'] = {'properties': {k: {} for k in value['end_state_context']}}
    checked = decode_observations(value, catalog, schema, 2)
    assert checked['accepted'] is False
    assert checked['primary_action_completed'] is True
    assert checked['boundary_continuity_valid' if domain == 'continuity' else 'composition_change_valid'] is False
    assert value['boundary_continuity_valid'] is True and value['composition_change_valid'] is True


@pytest.mark.parametrize('change', ['state', 'source', 'version'])
def test_changed_runtime_origin_is_rejected_before_provider_submission(tmp_path, change):
    from pipeline.continuation import prepare_continuation
    from pipeline.provider_prompt import compile_normal_provider_prompt
    from test_director_intent import Responses
    story, stored = captured()
    shot = deepcopy(story['shots'][1]); shot['generation_execution'] = 'scene-instructions-v2'
    previous = {**story['shots'][0], 'observed_end_state': stored['input']['observed_state']}
    result = prepare_continuation(shot, previous, str(ROOT / 'shots/shot_001_lastframe.jpg'), tmp_path,
                                 client=Responses(stored['plan']))
    if change == 'state': result['start_state']['prop_state'] = 'Invented initial state'
    if change == 'source': result['observed_origin']['source_sha256'] = 'not a source hash'
    if change == 'version': result['observed_origin']['version'] = 'invented'
    with pytest.raises(ValueError, match='起点'):
        compile_normal_provider_prompt(result, story, has_observed_start=True)


@pytest.mark.parametrize('first,last', [
    ('Courier reaches the receiver.', 'Receiver signs for the parcel.'),
    ('Rescuer secures the rope.', 'Swimmer reaches the river bank.'),
    ('Dancers enter the stage.', 'Dancers finish a synchronized turn.'),
])
def test_execution_scope_never_serializes_other_shots_work(first, last):
    from pipeline.generation_instructions import local_goal_instructions
    catalog = {'criteria': {'goal': {'kind': 'completion', 'action': first + ' ' + last,
        'beat_ids': [1], 'execution_scope': {'shot_id': 7, 'units': [
            {'shot_id': 7, 'beat_index': 0, 'actor': 'A', 'target': 'B', 'action': first},
            {'shot_id': 2, 'beat_index': 0, 'actor': 'A', 'target': 'B', 'action': last}]}}}}
    original = deepcopy(catalog)
    assert local_goal_instructions(catalog) == [first]
    assert catalog == original


def test_boundary_support_receives_prior_endpoint_as_evidence_and_rejects_mixed_sources(tmp_path):
    from pipeline.continuation import prepare_continuation
    from pipeline.result_observations import completion_contract
    from pipeline.observation_support import support_input
    from test_director_intent import Responses
    story, stored = captured()
    shot = deepcopy(story['shots'][1]); shot['generation_execution'] = 'scene-instructions-v2'
    previous = {**story['shots'][0], 'observed_end_state': stored['input']['observed_state']}
    result = prepare_continuation(shot, previous, str(ROOT / 'shots/shot_001_lastframe.jpg'), tmp_path,
                                 client=Responses(stored['plan']))
    catalog = completion_contract(result)
    value = {'observations': {k: {'status': 'satisfied', 'observation': 'Explicit offline observation.',
                                 'sample_indices': [1]} for k, c in catalog['criteria'].items()
                             if c['kind'] != 'realization'}}
    payload = support_input(catalog, value)
    assert payload['observed_context']['previous_endpoint']['facts'] == previous['observed_end_state']
    assert payload['observed_context']['previous_endpoint']['source'] == result['observed_origin']['source_sha256']
    catalog['criteria']['boundary/camera_change']['source_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='不同来源'):
        support_input(catalog, value)


def test_boundary_positive_observation_cannot_erase_unknown_aggregate():
    from pipeline.result_observations import decode_observations
    catalog = {'criteria': {'camera': {'kind': 'state', 'domain': 'camera', 'beat_id': 0}},
               'intent_policy': 'goal-allocation-v1'}
    value = {'observations': {'camera': {'status': 'satisfied', 'observation': 'Explicit fixture.', 'sample_indices': [1]}},
        'composition_change_valid': None,
        'end_state_context': {'location': 'stage', 'subject': 'dancer'},
        'terminal_observation': {'sample_index': 1, 'status': 'observed', 'action_phase': 'aftermath',
                                'pose_and_gaze': '', 'prop_state': '', 'open_motion': ''}}
    schema = {'properties': {k: {} for k in value}}
    schema['properties']['end_state_context'] = {'properties': {k: {} for k in value['end_state_context']}}
    assert decode_observations(value, catalog, schema, 1)['composition_change_valid'] is None


def test_review_resume_preserves_boundary_rejection_when_geometry_is_inconclusive(tmp_path, monkeypatch):
    from pipeline.continuation import prepare_continuation
    from pipeline.semantic_review import SemanticTakeReviewer
    from test_director_intent import Responses
    from test_full_production_flow import completion
    from tools.camera_measurement import VERSION
    story, stored = captured()
    shot = deepcopy(story['shots'][1]); shot['generation_execution'] = 'scene-instructions-v2'
    previous = {**story['shots'][0], 'observed_end_state': stored['input']['observed_state']}
    tail = ROOT / 'shots/shot_001_lastframe.jpg'
    shot = prepare_continuation(shot, previous, str(tail), tmp_path,
                                client=Responses(stored['plan']))
    digest = '6fcb69d86b14d9ad33b00d061f7bf40dbc3f35dfc837c2ce44c5a9033b0c3f70'
    original = json.loads((ROOT / f'semantic_reviews/{digest}.evidence.json').read_text())['response']
    candidate = deepcopy(original)
    # 新问题使用明确的离线负例；原始聚合标签仍为 true。
    candidate['observations'].update({
        'boundary/physical_progress': {'status': 'satisfied', 'observation': 'Offline fixture: physical state continues.',
                                       'sample_indices': [1]},
        'boundary/camera_change': {'status': 'violated', 'observation': 'Offline fixture: both sides have identical framing.',
                                   'sample_indices': [1]},
    })
    calls = []
    def create(**request):
        calls.append(request)
        return completion(deepcopy(candidate))
    monkeypatch.setattr('tools.camera_measurement.measure_video_composition',
                        lambda *_: {'version': VERSION, 'status': 'inconclusive'})
    reviewer = SemanticTakeReviewer(tmp_path, model='offline',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    first = reviewer.review(str(ROOT / 'shots/shot_002.mp4'), shot, previous_frame_path=str(tail),
        boundary_context={'same_scene': True, 'state_reference_role': 'reference_image',
                          'previous_observed_end_state': previous['observed_end_state']})
    resumed = reviewer.review(str(ROOT / 'shots/shot_002.mp4'), shot, previous_frame_path=str(tail),
        boundary_context={'same_scene': True, 'state_reference_role': 'reference_image',
                          'previous_observed_end_state': previous['observed_end_state']})
    assert first.accepted is resumed.accepted is False
    assert first.composition_change_valid is resumed.composition_change_valid is False
    assert first.primary_action_completed is resumed.primary_action_completed is True
    assert original['composition_change_valid'] is True and len(calls) == 1
