"""场景存在、镜头呈现和验收证据分离；真实记录保持不变，禁止外部调用。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from test_goal_allocation import offline


def captured():
    fixture = json.loads((Path(__file__).parent / 'fixtures/real_scene_frame_failure.json').read_text())
    for record in fixture['files'].values():
        assert hashlib.sha256(record['text'].encode()).hexdigest() == record['sha256']
    return fixture, {k: json.loads(v['text']) for k, v in fixture['files'].items()}


def test_real_generation_has_separate_world_and_frame_authority_without_losing_actions():
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    from pipeline.visual_invariants import compile_visual_invariants
    _, records = captured()
    original = records['continuations/shot_002.json']['execution_shot']
    shot = deepcopy(original)
    shot['generation_execution'] = 'local-events-v2'
    story = records['storyboard.json']
    normal = compile_normal_provider_prompt(shot, story, has_observed_start=True)
    safe = compile_policy_safe_prompt(shot, storyboard=story, has_state_reference=True,
                                    image_role='reference_image', reference_count=1)
    for prompt in (normal, safe):
        assert 'visible props:' not in prompt
        assert 'scene inventory (not a demand to include every prop in frame)' in prompt
        assert 'world persistence' in prompt
        assert all(prop in prompt for prop in shot['continuity_props'])
    assert normal.index('Required observable functions') < normal.index('Authored staging guidance')
    assert normal.index('camera framing:') < normal.index('Authored staging guidance')
    for beat in shot['action_beats']:
        assert beat['action'] in normal
    assert shot['camera'] == original['camera'] and shot['action_beats'] == original['action_beats']
    scope = compile_visual_invariants(shot).review_projection['visibility_scope']
    assert scope['required_in_frame'] == shot['required_visible_entities']
    assert scope['persistent_in_world'] == shot['continuity_props']
    assert 'visibility_scope' not in compile_visual_invariants(original).review_projection


def images(tmp_path):
    # 独立、可重复的纹理场景，不使用场景关键词推断镜头。
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    original = tmp_path / 'source.jpg'
    cv2.imwrite(str(original), frame)
    return frame, original


@pytest.mark.parametrize('change', ['subject', 'light', 'crop', 'blank'])
def test_geometry_distinguishes_subject_or_light_changes_from_reframing(tmp_path, change):
    from tools.camera_measurement import measure_composition
    frame, source = images(tmp_path)
    changed = frame.copy()
    if change == 'subject':
        cv2.rectangle(changed, (210, 170), (350, 320), 0, -1)
    elif change == 'light':
        changed = cv2.convertScaleAbs(changed, alpha=.7, beta=25)
    elif change == 'crop':
        changed = cv2.resize(changed[60:420, 80:560], (640, 480))
    else:
        changed[:] = 0
    target = tmp_path / 'candidate.jpg'
    cv2.imwrite(str(target), changed)
    result = measure_composition(source, target)
    assert result['status'] == ('changed' if change == 'crop' else 'inconclusive' if change == 'blank' else 'unchanged')


def test_real_camera_true_has_independent_unchanged_geometry(tmp_path):
    from tools.camera_measurement import measure_composition
    fixture, records = captured()
    root = Path(__file__).parents[1] / fixture['source']
    video = root / 'shots/shot_002_rejected_1.mp4'
    if not video.exists():
        pytest.skip('local original video unavailable')
    from tools.frame_extractor import extract_frame
    first = tmp_path / 'first.jpg'
    extract_frame(str(video), str(first), timestamp=0)
    result = measure_composition(root / 'shots/shot_001_lastframe.jpg', first)
    assert result['status'] == 'unchanged'
    assert result['coverage'] > .6 and result['inliers'] >= 50
    raw = records['semantic_reviews/' + hashlib.sha256(video.read_bytes()).hexdigest() + '.json']
    assert raw['observation_response']['composition_change_valid'] is True
    assert raw['review']['primary_action_completed'] is False


def test_real_review_replay_preserves_event_failure_and_refutes_camera_true_without_new_call(tmp_path):
    from types import SimpleNamespace
    from pipeline.semantic_review import SemanticTakeReviewer, SemanticReviewUnavailableError
    from test_full_production_flow import completion
    fixture, records = captured()
    root = Path(__file__).parents[1] / fixture['source']
    video, tail = root / 'shots/shot_002_rejected_1.mp4', root / 'shots/shot_001_lastframe.jpg'
    if not video.exists():
        pytest.skip('local original video unavailable')
    key = hashlib.sha256(video.read_bytes()).hexdigest()
    response = records[f'semantic_reviews/{key}.json']['observation_response']
    original = deepcopy(response)
    shot = deepcopy(records['continuations/shot_002.json']['execution_shot'])
    shot['generation_execution'] = 'local-events-v2'
    calls = []
    def create(**kwargs):
        import jsonschema
        jsonschema.validate(response, kwargs['response_format']['json_schema']['schema'])
        calls.append(kwargs)
        return completion(deepcopy(response))
    reviewer = SemanticTakeReviewer(tmp_path, model='offline-original-record',
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    for _ in range(2):
        result = reviewer.review(str(video), shot, previous_frame_path=str(tail),
            boundary_context={'same_scene': True, 'state_reference_role': 'reference_image'})
        assert result.accepted is False and result.primary_action_completed is False
        assert result.composition_change_valid is False
    assert len(calls) == 1 and response == original
    saved_path = reviewer.cache_dir / f'{key}.json'
    saved = json.loads(saved_path.read_text())
    assert saved['observation_response']['composition_change_valid'] is True
    saved['composition_measurement']['status'] = 'changed'
    saved_path.write_text(json.dumps(saved))
    with pytest.raises(SemanticReviewUnavailableError):
        reviewer.review(str(video), shot, previous_frame_path=str(tail),
            boundary_context={'same_scene': True, 'state_reference_role': 'reference_image'})
    assert len(calls) == 1


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_other_scenes_keep_world_props_without_forcing_them_into_detail_views(topic):
    from test_creation_schedule_flow import scenario_story
    from test_staging_direction import compile_story
    from pipeline.provider_prompt import compile_normal_provider_prompt, compile_policy_safe_prompt
    from pipeline.visual_invariants import compile_visual_invariants
    _, _, raw, plan = scenario_story(topic)
    story = compile_story(raw, plan)
    shot = story['shots'][1]
    assert shot['generation_execution'] == 'scene-instructions-v2'
    shot['key_props'] = ['off screen continuity object']
    shot['continuity_props'] = ['off screen continuity object']
    shot['camera']['start_framing'] = 'close-up detail shot'
    normal = compile_normal_provider_prompt(shot, story, has_observed_start=True)
    safe = compile_policy_safe_prompt(shot, storyboard=story, has_state_reference=True,
                                    image_role='reference_image', reference_count=1)
    for prompt in (normal, safe):
        assert 'world persistence' in prompt and 'visible props:' not in prompt
        assert 'close detail shot' in prompt
    assert all(beat['action'] in normal for beat in shot['action_beats'])
    scope = compile_visual_invariants(shot).review_projection['visibility_scope']
    assert 'off screen continuity object' not in scope['required_in_frame']
    assert scope['persistent_in_world'] == ['off screen continuity object']


@pytest.mark.parametrize('status', ['changed', 'inconclusive'])
def test_geometry_cannot_override_a_semantic_rejection(status):
    from tools.camera_measurement import composition_verdict, VERSION
    assert composition_verdict(False, {'version': VERSION, 'status': status}) is False


def test_missing_source_is_not_treated_as_inconclusive(tmp_path):
    from tools.camera_measurement import measure_composition
    _, source = images(tmp_path)
    with pytest.raises(ValueError):
        measure_composition(source, tmp_path / 'missing.jpg')
