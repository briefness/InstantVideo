"""本地呈现使用真实编解码；区域是合成标注，不证明视觉识别正确。"""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess

import pytest


def test_region_solver_preserves_all_protected_regions():
    from pipeline.presentation_media import solve_window
    context = {'source_size': [864, 496], 'region_entities': ['person', 'object']}
    regions = {'person': [.15, .1, .29, .94], 'object': [.65, .22, .83, .98]}
    original = deepcopy(regions)
    result = solve_window({'regions': regions}, context)
    assert regions == original
    assert result['scale'] == pytest.approx(.88)
    for x1, y1, x2, y2 in regions.values():
        assert result['left'] <= x1 < x2 <= result['left'] + result['scale'] + 1e-9
        assert result['top'] <= y1 < y2 <= result['top'] + result['scale'] + 1e-9
    # 全画面保护意味着无裁切解，不能偷偷丢弃人物或改变目标。
    assert solve_window({'regions': {'person': [0., 0., 1., 1.], 'object': [.3, .3, .4, .4]}}, context) is None


@pytest.mark.parametrize('regions', [None, {}, {'person': [0., 0., float('nan'), 1.]},
    {'person': [.5, .1, .2, .9]}, {'person': [-.1, .1, .8, .9]}])
def test_invalid_regions_are_not_repaired(regions):
    from pipeline.presentation_media import solve_window
    with pytest.raises((ValueError, TypeError)):
        solve_window({'regions': regions}, {'source_size': [864, 496], 'region_entities': ['person']})


def make_video(root, duration=2.):
    import cv2
    import numpy as np
    from tools.frame_extractor import extract_frame
    source = root / 'original.mp4'
    pattern = root / 'pattern.png'
    cv2.imwrite(str(pattern), np.random.default_rng(17).integers(0, 256, (496, 864, 3), dtype=np.uint8))
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-loop', '1', '-framerate', '24', '-i',
        str(pattern), '-f', 'lavfi', '-i', f'sine=duration={duration}', '-t', str(duration),
        '-vf', 'eq=brightness=0.12*sin(2*PI*t):eval=frame',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
        '-shortest', str(source)], check=True)
    tail = root / 'prior.jpg'
    extract_frame(str(source), str(tail), timestamp=0.)
    return source, tail


def shot_spec(tail):
    from pipeline.reference_view import bind_view, camera_view_context
    shot = {'shot_id': 2, 'completion_policy': 'goal-allocation-v1',
        'presentation_policy': 'local-reframe-v1', 'continuity_from_previous': 'intentional_cut',
        'characters': ['actor'], 'camera': {'primary_movement': 'fixed', 'start_framing': 'medium shot',
        'end_framing': 'medium shot', 'composition': 'centered', 'axis_change': 'hold'}}
    context = camera_view_context(shot, {}, tail)
    shot['reference_view'] = bind_view({'left': .2, 'top': .2, 'scale': .6,
        'regions': {'actor': [.3, .3, .6, .6]}}, context, hashlib.sha256(tail.read_bytes()).hexdigest())
    return shot


def test_render_resolve_integrity_and_no_source_mutation(tmp_path):
    from pipeline.presentation_media import ensure_presentation, resolve_presentation, source_box
    from tools.ffmpeg_ops import get_video_info
    source, tail = make_video(tmp_path)
    shot = shot_spec(tail)
    before = source.read_bytes()
    result = ensure_presentation(str(source), shot, str(tail), tmp_path)
    assert result != str(source) and source.read_bytes() == before
    assert resolve_presentation(str(source), shot, tmp_path) == result
    assert ensure_presentation(str(source), shot, str(tail), tmp_path) == result
    info = get_video_info(result)
    assert any(s['codec_type'] == 'audio' for s in info['streams'])
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    assert (video['width'], video['height']) == (864, 496)
    assert float(video['duration']) == pytest.approx(2., abs=1/24)
    box = source_box(str(source), shot, tmp_path, [.25, .25, .75, .75])
    assert box == pytest.approx([.35, .35, .65, .65], abs=.005)
    Path(result).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='呈现'):
        resolve_presentation(str(source), shot, tmp_path)


def test_unmarked_history_is_native_and_missing_new_receipt_fails_closed(tmp_path):
    from pipeline.presentation_media import resolve_presentation
    source, tail = make_video(tmp_path)
    assert resolve_presentation(str(source), {}, tmp_path) == str(source)
    with pytest.raises(ValueError, match='呈现'):
        resolve_presentation(str(source), shot_spec(tail), tmp_path)


def test_resume_rechecks_observed_source_even_when_presentation_is_cached(tmp_path):
    from pipeline.presentation_media import ensure_presentation
    source, tail = make_video(tmp_path)
    shot = shot_spec(tail)
    ensure_presentation(str(source), shot, str(tail), tmp_path)
    tail.write_bytes(b'changed observation')
    with pytest.raises(ValueError, match='来源'):
        ensure_presentation(str(source), shot, str(tail), tmp_path)


@pytest.mark.parametrize('movement', ['orbit', 'tracking', 'dolly in'])
def test_crop_never_impersonates_a_new_viewpoint(tmp_path, movement):
    from pipeline.presentation_media import ensure_presentation
    source, tail = make_video(tmp_path)
    shot = shot_spec(tail)
    shot['camera']['primary_movement'] = movement
    assert ensure_presentation(str(source), shot, str(tail), tmp_path) == str(source)


def test_current_real_failure_preserves_source_and_does_not_claim_semantic_acceptance(tmp_path):
    import json
    from pipeline.presentation_media import solve_window, ensure_presentation
    from pipeline.reference_view import bind_view
    from tools.camera_measurement import measure_video_composition, composition_verdict
    root = Path(__file__).parents[1] / 'output/20261112_164734_207960'
    source = root / 'shots/shot_002_rejected_1.mp4'
    tail = root / 'shots/shot_001_lastframe.jpg'
    if not source.exists():
        pytest.skip('真实原片仅在本地可用')
    record = json.loads((root / 'continuations/shot_002.json').read_text())
    context = record['input']['camera_view']
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    assert before == '43ec8e48cc1212514f386916bfe0fad1f4e6defa876f0789284fd296f42429d4'
    # 只重放真实区域观察以验证算法。没有修改旧提案、旧分镜、旧结论或恢复旧运行。
    window = solve_window({'regions': record['plan']['view_window']['regions']}, context)
    assert window['scale'] == pytest.approx(.88)
    shot = deepcopy(record['execution_shot'])
    shot['presentation_policy'] = 'local-reframe-v1'
    shot['reference_view'] = bind_view(window, context, hashlib.sha256(tail.read_bytes()).hexdigest())
    presented = ensure_presentation(str(source), shot, str(tail), tmp_path)
    assert presented != str(source)
    measurement = measure_video_composition(tail, presented, 0.)
    assert measurement['status'] == 'changed'
    assert composition_verdict(False, measurement) is False
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
