"""真实记录不改判；取景服务标注只检验程序执行，不冒充模型视觉能力。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from test_goal_allocation import offline


def capture():
    data = json.loads((Path(__file__).parent / 'fixtures/real_view_execution_failure.json').read_text())
    for record in data['files'].values():
        assert hashlib.sha256(record['text'].encode()).hexdigest() == record['sha256']
    values = {k: json.loads(v['text']) for k, v in data['files'].items()}
    return data, values


def test_real_failure_remains_camera_failure_and_optional_questions_are_not_requested():
    from pipeline.causality import compile_action_contract
    from test_functional_results import format_for
    _, values = capture()
    shot = values['continuations/shot_002.json']['execution_shot']
    old = next(v for k, v in values.items() if k.startswith('semantic_reviews/'))
    assert old['review']['primary_action_completed'] is True
    assert old['observation_response']['composition_change_valid'] is False
    assert old['observation_response']['observations']['beat:2/completion']['status'] == 'unknown'
    contract = compile_action_contract(shot).review_projection['completion_contract']
    observed = format_for(shot)['json_schema']['schema']['properties']['observations']['properties']
    assert set(observed) == {k for k, c in contract['criteria'].items() if c['kind'] != 'realization'}
    assert any(c['kind'] == 'completion' for c in contract['criteria'].values())


def view_case(tmp_path):
    """合成图和明确服务标注，只测试数据流，不声称识别人像或画面合格。"""
    import cv2
    import numpy as np
    from pipeline.reference_view import camera_view_context
    _, values = capture()
    original = values['continuations/shot_002.json']
    shot = deepcopy(original['execution_shot'])
    previous = {'shot_id': 1, 'scene_id': shot['scene_id'],
                'observed_end_state': original['input']['observed_state'], 'output_reference_depth': 3}
    tail = tmp_path / 'accepted-tail.jpg'
    image = np.indices((496, 864)).sum(axis=0).astype('uint8')
    assert cv2.imwrite(str(tail), image)
    context = camera_view_context(shot, previous, tail)
    assert context is not None
    window = {'left': .1, 'top': .1, 'scale': .75,
              'regions': {e: [.2, .2, .7, .7] for e in context['region_entities']}}
    return shot, previous, tail, context, window


@pytest.mark.parametrize('select_window', [True, False])
def test_formal_continuation_uses_actual_image_once_and_preserves_actions_and_resume(tmp_path, select_window):
    import base64
    from pipeline.continuation import prepare_continuation, OPTIONAL_VIEW_VERSION
    from test_director_intent import Responses
    shot, previous, tail, context, window = view_case(tmp_path)
    original = deepcopy(shot)
    sdk = Responses({'compatible': True, 'entry_action': 'Keep the existing action from the observed state.',
        'reason': 'Offline service annotation only.', 'view_window': window if select_window else None})
    changed = prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk)
    assert shot == original and changed['action_beats'] == original['action_beats']
    assert changed['camera'] == original['camera']
    assert bool(changed.get('reference_view')) is select_window
    request = sdk.calls[0]
    url = request['messages'][1]['content'][0]['image_url']['url']
    assert base64.b64decode(url.split(',', 1)[1]) == tail.read_bytes()
    planned = json.loads(request['messages'][-1]['content'])
    assert planned['version'] == OPTIONAL_VIEW_VERSION and planned['camera_view'] == context
    assert prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk, allow_create=False) == changed
    assert len(sdk.calls) == 1


def test_view_pixels_provenance_and_privacy_route_keep_same_state_lineage(tmp_path):
    import cv2
    import numpy as np
    from pipeline.reference_view import bind_view, materialize_view, matches_view
    from pipeline.generator import VideoGenerator
    shot, previous, tail, context, window = view_case(tmp_path)
    original_hash = hashlib.sha256(tail.read_bytes()).hexdigest()
    shot['reference_view'] = bind_view(window, context, original_hash)
    gen = VideoGenerator(output_dir=str(tmp_path), review_content=False)
    refs, role = gen._build_image_refs(shot, str(tail), previous)
    assert role == 'first_frame' and refs != [str(tail)]
    assert gen._build_state_only_refs(shot, str(tail), previous) == (refs, role)
    assert gen._has_state_reference(shot, previous, str(tail), refs, role)
    assert not gen._has_removable_identity_refs(refs, role, has_state_reference=True)
    expected = cv2.imread(str(tail))[50:422, 86:734]
    actual = cv2.imread(refs[0])
    assert actual.shape == expected.shape
    assert np.abs(actual.astype(float) - expected).mean() < 3
    provenance = gen._generation_provenance(shot, refs, role)
    assert provenance['reference_source_fingerprints'] == [original_hash]
    assert provenance['reference_fingerprints'] != [original_hash]
    assert gen._next_reference_depth(shot, previous, str(tail), reference_role=role,
        reference_fingerprints=provenance['reference_fingerprints'],
        reference_source_fingerprints=provenance['reference_source_fingerprints']) == 4
    assert materialize_view(shot, str(tail), tmp_path) == refs[0]
    assert matches_view(shot, str(tail), refs[0])
    assert hashlib.sha256(tail.read_bytes()).hexdigest() == original_hash


@pytest.mark.parametrize('bad', ['missing_entity', 'clipped_entity', 'outside', 'zero', 'tiny', 'nan', 'extra', 'boolean'])
def test_invalid_or_incomplete_view_stops_before_generation(tmp_path, bad):
    from pipeline.continuation import prepare_continuation, ContinuationUnavailableError
    from test_director_intent import Responses
    shot, previous, tail, context, window = view_case(tmp_path)
    entity = context['region_entities'][0]
    if bad == 'missing_entity':
        del window['regions'][entity]
    elif bad == 'clipped_entity':
        window['regions'][entity] = [0, 0, 1, 1]
    elif bad == 'outside':
        window['left'] = .9
    elif bad == 'extra':
        window['replace_story'] = True
    else:
        window['scale'] = {'zero': 0, 'tiny': .01, 'nan': float('nan'), 'boolean': True}[bad]
    sdk = Responses({'compatible': True, 'entry_action': 'Keep the action.', 'reason': 'offline negative',
                     'view_window': window})
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=sdk)
    assert 'reference_view' not in result
    assert result['reference_view_decision']['reason'] == 'invalid_proposal'
    assert result['camera'] == shot['camera']
    assert len(sdk.calls) == 1


@pytest.mark.parametrize('change', ['source', 'output', 'receipt', 'camera'])
def test_changed_view_source_output_and_camera_fail_closed(tmp_path, change):
    from pipeline.reference_view import bind_view, materialize_view
    shot, _, tail, context, window = view_case(tmp_path)
    shot['reference_view'] = bind_view(window, context, hashlib.sha256(tail.read_bytes()).hexdigest())
    path = Path(materialize_view(shot, str(tail), tmp_path))
    if change == 'source':
        tail.write_bytes(tail.read_bytes() + b'changed')
    elif change == 'output':
        path.write_bytes(path.read_bytes() + b'changed')
    elif change == 'receipt':
        path.with_suffix('.json').unlink()
    else:
        shot['camera']['primary_movement'] = 'pan left'
    with pytest.raises(ValueError):
        materialize_view(shot, str(tail), tmp_path)


@pytest.mark.parametrize('change', ['wide', 'reestablish', 'seamless', 'new_angle', 'motion'])
def test_crop_cannot_claim_unseen_scene_or_change_motion_reference(tmp_path, change):
    from pipeline.reference_view import camera_view_context
    shot, previous, tail, _, _ = view_case(tmp_path)
    if change == 'wide':
        shot['camera']['start_framing'] = 'wide shot'
    elif change == 'reestablish':
        shot['camera']['axis_change'] = 'reestablish'
    elif change == 'new_angle':
        shot['camera']['composition'] = 'over the shoulder'
    elif change == 'seamless':
        shot['continuity_from_previous'] = 'seamless'
    else:
        shot['motion_reference'] = {'uri': 'asset://offline'}
    assert camera_view_context(shot, previous, tail) is None


def test_latest_original_video_replay_keeps_camera_rejection_and_no_optional_guessing(tmp_path):
    from types import SimpleNamespace
    import jsonschema
    from pipeline.semantic_review import SemanticTakeReviewer
    from test_full_production_flow import completion
    fixture, values = capture()
    root = Path(__file__).parents[1] / fixture['source']
    video, tail = root / 'shots/shot_002_rejected_1.mp4', root / 'shots/shot_001_lastframe.jpg'
    if not video.is_file() or not tail.is_file():
        pytest.skip('original local media unavailable')
    raw = next(v for k, v in values.items() if k.startswith('semantic_reviews/'))
    shot = values['continuations/shot_002.json']['execution_shot']
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        schema = kwargs['response_format']['json_schema']['schema']
        value = deepcopy(raw['observation_response'])
        if 'decisions' in schema['properties']:
            from functional_observation_cases import annotated_support_response
            return completion(annotated_support_response(kwargs))
        # 只投影字段；必要事件、实际终态和镜头反例的原始观察不改写。
        value['observations'] = {k: v for k, v in value['observations'].items()
                                if k in schema['properties']['observations']['properties']}
        value['identity_crop_boxes'] = []
        jsonschema.validate(value, schema)
        return completion(value)
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-original-response-projection',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    for _ in range(2):
        result = reviewer.review(str(video), shot, previous_frame_path=str(tail),
            boundary_context={'same_scene': True, 'state_reference_role': 'reference_image'})
        assert result.primary_action_completed is True
        assert result.composition_change_valid is False and result.accepted is False
    assert len(calls) == 2 and hashlib.sha256(video.read_bytes()).hexdigest() == digest


def test_optional_only_shot_still_requires_actual_handoff_without_invented_events():
    from pipeline.causality import compile_action_contract
    from pipeline.result_observations import decode_observations, observation_criteria, ObservationUncertaintyError
    from test_functional_results import format_for
    from test_goal_authority import annotated_snapshot
    _, values = capture()
    shot = deepcopy(values['continuations/shot_002.json']['execution_shot'])
    for beat in shot['action_beats']:
        beat['completion_intent'] = {'kind': 'allocated', 'requirements': []}
    catalog = compile_action_contract(shot).review_projection['completion_contract']
    schema = format_for(shot)['json_schema']['schema']
    assert schema['properties']['observations']['properties'] == {}
    wire = annotated_snapshot(shot)
    assert not wire['observations'] and not wire.get('action_event_evidence')
    assert decode_observations(wire, catalog, schema, 25)['observed_end_state']['action_phase']
    wire['terminal_observation']['status'] = 'unknown'
    before = deepcopy(wire)
    with pytest.raises(ObservationUncertaintyError) as caught:
        decode_observations(wire, catalog, schema, 25)
    assert caught.value.keys == ['terminal_observation'] and wire == before
    with pytest.raises(ValueError, match='未知'):
        observation_criteria({**catalog, 'observation_policy': 'untrusted-version'})


@pytest.mark.parametrize('entities', [['dancer', 'partner'], ['package', 'conveyor'], ['rescuer', 'rope']])
def test_view_geometry_is_driven_by_visible_entities_not_story_words(tmp_path, entities):
    from pipeline.reference_view import camera_view_context, bind_view, materialize_view
    shot, previous, tail, _, _ = view_case(tmp_path)
    shot.update(characters=[], required_visible_entities=entities)
    context = camera_view_context(shot, previous, tail)
    assert context['region_entities'] == entities
    window = {'left': .1, 'top': .1, 'scale': .75,
              'regions': {e: [.2, .2, .7, .7] for e in entities}}
    shot['reference_view'] = bind_view(window, context, hashlib.sha256(tail.read_bytes()).hexdigest())
    assert Path(materialize_view(shot, str(tail), tmp_path)).is_file()
