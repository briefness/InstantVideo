"""重演反证来自连续原片，不来自模型的通过标签。"""
from pathlib import Path
import json
import numpy as np
import pytest
from test_provider_materials import original

ROOT = Path(__file__).parents[1] / 'output/20261117_133857_156942'


def test_portable_failure_record_keeps_operation_and_false_positive_separate():
    import hashlib
    fixture = json.loads((Path(__file__).parent/'fixtures/real_sequence_reuse_failure.json').read_text())
    third = fixture['shots'][2]
    assert third['operation']['task'] == 'extend_after'
    assert third['operation']['camera_boundary'] == 'cut_to_current_shot'
    assert third['review_flags']['accepted'] is True
    assert third['review_flags']['action_handoff_valid'] is True
    if ROOT.exists():
        for name, digest in fixture['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest


def test_captured_repeat_has_no_new_work_despite_positive_review():
    from tools.sequence_reuse import measure_reuse
    if not ROOT.exists():
        pytest.skip('本地原片不随仓库分发')
    result = measure_reuse(ROOT/'shots/shot_002.mp4', ROOT/'shots/shot_003.mp4', 6, 6)
    assert result['status'] == 'replayed_prefix'
    assert result['matched_seconds'] >= 2
    assert result['previous_start_seconds'] < 1
    assert result['current_start_seconds'] == 0
    review = json.loads((ROOT/'semantic_reviews'/f"{result['current_sha256']}.json").read_text())
    assert review['review']['action_handoff_valid'] is True  # 真实漏检，不能改fixture凑通过。


@pytest.mark.parametrize('seed', [5, 6, 7])
def test_dynamic_prefix_copy_is_not_static_or_shared_background(seed):
    from tools.sequence_reuse import compare_sequences
    rng = np.random.default_rng(seed)
    frames = rng.uniform(0, 255, (24, 36, 64)).astype(np.float32)
    assert compare_sequences(frames, frames + 1)['status'] == 'replayed_prefix'
    assert compare_sequences(frames, frames[::-1])['status'] == 'inconclusive'
    assert compare_sequences(frames, rng.uniform(0, 255, frames.shape))['status'] == 'inconclusive'
    static = np.repeat(frames[:1], 24, axis=0)
    assert compare_sequences(static, static)['status'] == 'inconclusive'
    assert compare_sequences(frames, frames[-3:])['status'] == 'inconclusive'


def test_same_background_and_moving_subjects_with_different_paths_are_not_replay():
    from tools.sequence_reuse import compare_sequences
    left = np.full((24,36,64), 70, dtype=np.float32)
    right = left.copy()
    for i in range(24):
        left[i, 8:28, i:i+12] = 230
        right[i, 8:28, 48-i:60-i] = 230
    assert compare_sequences(left, right)['status'] == 'inconclusive'


def test_new_successor_intent_is_not_inferred_from_source_duration():
    from pipeline.provider_operation import execution_intent, compile_operation
    state = dict(usage='selected_state', logical_index=1, media_label='视频1', source_task_id='offline',
                 source_sha256='a'*64, timestamp=5.95, source_duration=6, source_fps='24',
                 selected_end_seconds=6, media_type='video_url')
    shot = {'continuity_from_previous':'intentional_cut', 'observed_origin': {'version':'observed-origin-v1'}}
    op = compile_operation([state], execution_intent(shot))
    assert op['task'] == 'generate_successor'
    assert op['transport_control'] == 'semantic_reference'
    assert compile_operation([state], {'boundary':'seamless'})['task'] == 'extend_after'
    assert compile_operation([state], {'boundary':'intentional_cut'})['task'] == 'extend_after'  # 旧契约不改写。


def test_review_has_full_history_while_motion_estimation_keeps_recent_tail():
    from pipeline.temporal_evidence import sample_clip, combine
    if not ROOT.exists():
        pytest.skip('本地原片不随仓库分发')
    video = str(ROOT/'shots/shot_002.mp4')
    shot = {'duration': 6, 'media_selection':'planned-prefix-v1'}
    history = sample_clip(video, shot, scope='previous_clip')
    tail = sample_clip(video, shot, scope='previous_tail')
    assert history['manifest']['sources'][0]['start_seconds'] == 0
    assert tail['manifest']['sources'][0]['start_seconds'] > 3
    assert combine(history)['manifest'] == history['manifest']


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1, 31, True])
def test_invalid_media_bounds_cannot_become_a_clean_result(invalid, tmp_path):
    from tools.sequence_reuse import measure_reuse
    p = tmp_path/'bad.mp4'
    p.write_bytes(b'not media')
    with pytest.raises(ValueError):
        measure_reuse(p, p, invalid, 6)


def test_new_operation_keeps_all_official_material_operands(original):
    from pipeline import provider_materials as media
    from pipeline.provider_operation import execution_intent
    state, video = original(kind='selected_state', character=None, timestamp=5.95, box=None)
    media.bind_derivation(state, str(video), kind='selected_state', timestamp=5.95, selected_end_seconds=6.08)
    identity, _ = original(2, character='Lead')
    refs = [state, identity]
    old = media.material_plan(refs, 'reference_image', execution={'boundary':'intentional_cut'})
    new = media.material_plan(refs, 'reference_image', execution=execution_intent(
        {'continuity_from_previous':'intentional_cut', 'observed_origin': {'version':'observed-origin-v1'}}))
    assert old['content'] == new['content'] and old['bindings'] == new['bindings']
    assert new['operation']['task'] == 'generate_successor'


@pytest.mark.parametrize('mode', ['strict', 'normal'])
def test_formal_reviewer_rejects_replay_before_model_and_does_not_rewrite_source(tmp_path, mode):
    from pipeline.semantic_review import SemanticTakeReviewer
    from pipeline.review_policy import decide_review
    from types import SimpleNamespace as NS
    if not ROOT.exists():
        pytest.skip('本地原片不随仓库分发')
    cached = json.loads(next(p for p in (ROOT/'semantic_reviews').glob('*.json')
                        if p.stem.startswith('0e4ba') and not p.name.endswith('.evidence.json')).read_text())
    story = json.loads((ROOT/'result_admission.json').read_text())['execution_storyboard']
    def forbidden(**kwargs):
        pytest.fail('重复原片不能发起新的视觉调用')
    reviewer = SemanticTakeReviewer(tmp_path, review_mode=mode, client=NS(chat=NS(completions=NS(create=forbidden))))
    shot = json.loads((ROOT/'continuations/shot_003.json').read_text())['execution_shot']
    for _ in range(2):
        result = reviewer.review(str(ROOT/'shots/shot_003.mp4'), shot,
            previous_video_path=str(ROOT/'shots/shot_002.mp4'), previous_shot=story['shots'][1],
            previous_frame_path=str(ROOT/'shots/shot_002_lastframe.jpg'))
        assert not result.accepted and not result.action_handoff_valid
        assert not decide_review(result, mode).allowed
    assert cached['review']['accepted'] is True
