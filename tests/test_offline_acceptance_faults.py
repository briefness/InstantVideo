"""Fault injection through real project interfaces; never call remote services."""
import copy
import hashlib
import json
import shutil
import subprocess

import httpx
import pytest

from test_action_sequence import sequence_story
from test_continuation import Planner, scenario
from test_motion_reference import package
from pipeline.continuation import prepare_continuation
from pipeline.motion_reference import prepare_motion_reference
from pipeline.provider_prompt import compile_policy_safe_prompt
from tools.prepare_motion_run import prepare_motion_run
from tools.publish_motion_reference import prepare_upload


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("self acceptance attempted external HTTP")
    monkeypatch.setattr(httpx.Client, "send", reject)
    monkeypatch.setattr(httpx.AsyncClient, "send", reject)


def test_continuation_cannot_rewrite_a_trigger_that_is_a_physical_event(tmp_path):
    _, previous, tail = scenario(tmp_path)
    shot = sequence_story()['shots'][0]
    shot.update(shot_id=2, scene_id='roof', continuity_from_previous='seamless')
    shot['action_beats'][0]['phase'] = 'trigger'
    original = copy.deepcopy(shot['action_beats'])
    planner = Planner()
    result = prepare_continuation(shot, previous, str(tail), tmp_path, client=planner)
    assert result['action_beats'] == original
    sent = json.loads(planner.calls[0]['messages'][1]['content'])
    assert sent['entry_editable'] is False


def test_safe_profile_retains_transition_and_result_hold():
    shot = sequence_story()['shots'][0]
    first = shot['action_beats'][0]
    first['start_seconds'] = .5
    shot['action_beats'].insert(0, {
        'phase': 'trigger', 'activity': 'transition', 'start_seconds': 0., 'end_seconds': .5,
        'actor': 'attacker', 'target': 'defender', 'action': 'moves into range',
        'visible_result': 'both palms are within reach'})
    prompt = compile_policy_safe_prompt(shot, has_state_reference=False, image_role=None, reference_count=0)
    assert 'both palms are within reach' in prompt
    assert 'attacker remains back' in prompt
    assert prompt.index('both palms are within reach') < prompt.index('event 2')


@pytest.mark.parametrize('replacement', ['blank', 'frozen', 'reversed'])
@pytest.mark.parametrize('entry', ['upload', 'workspace'])
def test_self_consistent_hashes_cannot_certify_wrong_reference_pixels(package, tmp_path, replacement, entry):
    board, _, original = package
    target = tmp_path / 'mutated'
    shutil.copytree(original, target)
    source, changed = target / 'reference.mp4', target / 'changed.mp4'
    filters = {
        'blank': 'drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill',
        'frozen': "select='eq(n,0)',setpts=N/FRAME_RATE/TB,tpad=stop_mode=clone:stop_duration=4",
        'reversed': 'reverse',
    }
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source), '-vf', filters[replacement],
                    '-frames:v', '96', '-r', '24', '-an', '-c:v', 'libx264', '-threads', '1',
                    '-pix_fmt', 'yuv420p', str(changed)], check=True, capture_output=True, timeout=20)
    changed.replace(source)
    manifest_path = target / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['reference.mp4'] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='画面|像素|运动'):
        if entry == 'upload':
            prepare_upload(target)
        else:
            prepare_motion_run(board, target, tmp_path / 'runs',
                               model='doubao-seedance-2-0-mini-260615')
    assert not (tmp_path / 'runs').exists()


def test_bound_reference_is_rechecked_from_pixels_before_submission(package, tmp_path):
    board, _, source = package
    workspace = prepare_motion_run(board, source, tmp_path,
                                   model='doubao-seedance-2-0-mini-260615', asset_uri='asset://test')
    shot = workspace.load_storyboard()['shots'][0]
    video = workspace.path / shot['motion_reference']['path']
    replacement = video.with_name('replacement.mp4')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(video), '-vf',
                    'drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill', '-an', '-c:v', 'libx264',
                    '-threads', '1', str(replacement)], capture_output=True, check=True, timeout=20)
    replacement.replace(video)
    shot['motion_reference']['sha256'] = hashlib.sha256(video.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='画面|像素|运动'):
        prepare_motion_reference(shot, '16:9', workspace.path, 'doubao-seedance-2-0-mini-260615')


def test_renderer_cannot_certify_its_own_wrong_joint_projection(package, tmp_path, monkeypatch):
    from PIL import Image
    from tools.motion_renderer import draw_motion_frame
    from tools.motion_previs import build_motion_previs
    board, plan, _ = package
    def shifted(*args, **kwargs):
        original = draw_motion_frame(*args, **kwargs)
        result = Image.new('RGB', original.size, '#101820')
        result.paste(original, (40, 0))
        return result
    # 即使渲染器与像素模板共享同一个错误，独立关节投影仍必须发现。
    monkeypatch.setattr('tools.motion_renderer.draw_motion_frame', shifted)
    monkeypatch.setattr('tools.motion_media_validation.draw_motion_frame', shifted)
    output = tmp_path / 'wrong_projection'
    with pytest.raises(ValueError, match='关节|画面'):
        build_motion_previs(board, plan, output)
    assert not output.exists()


@pytest.mark.parametrize('failure', ['timeout', 'decode_error', 'short_frames'])
def test_pixel_verification_failure_never_becomes_success(package, monkeypatch, failure):
    from types import SimpleNamespace
    from pipeline.motion_plan import compile_motion_previs
    from tools.motion_media_validation import verify_rendered_motion
    board, plan, path = package
    payload = (path / 'reference.mp4').read_bytes()
    compiled = compile_motion_previs(board['shots'][0], '16:9', plan)
    def broken(*args, **kwargs):
        assert kwargs['timeout'] == 30
        if failure == 'timeout':
            raise subprocess.TimeoutExpired('ffmpeg', 30)
        if failure == 'decode_error':
            raise subprocess.CalledProcessError(1, 'ffmpeg')
        return SimpleNamespace(stdout=b'incomplete')
    monkeypatch.setattr('tools.motion_media_validation.subprocess.run', broken)
    with pytest.raises(ValueError, match='画面'):
        verify_rendered_motion(payload, plan, compiled)


def test_missing_numeric_plan_blocks_reference_without_regeneration(package, tmp_path):
    board, _, source = package
    workspace = prepare_motion_run(board, source, tmp_path,
                                   model='doubao-seedance-2-0-mini-260615', asset_uri='asset://test')
    shot = workspace.load_storyboard()['shots'][0]
    video = workspace.path / shot['motion_reference']['path']
    video.with_name('motion_plan.json').unlink()
    with pytest.raises(ValueError, match='缺少本地数值运动计划'):
        prepare_motion_reference(shot, '16:9', workspace.path, 'doubao-seedance-2-0-mini-260615')
