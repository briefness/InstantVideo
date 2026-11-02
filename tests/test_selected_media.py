"""评审和续拍只消费进入成片的同一前缀；原片保持完整。"""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from pipeline.generator import VideoGenerator
from pipeline.semantic_review import SemanticTakeReviewer
from test_performance_outcomes import outcome_shot, outcome_response
from test_sequence_review import LocalCompletions
from tools.ffmpeg_ops import normalize_video, get_video_info


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *_: pytest.fail('offline only'))


@pytest.fixture
def long_video(tmp_path):
    path = tmp_path / 'provider.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
        'testsrc2=s=320x180:r=24:d=6', '-f', 'lavfi', '-i', 'sine=duration=7',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', str(path)], check=True)
    return path


def selected_shot():
    shot = outcome_shot()
    shot['media_selection'] = 'planned-prefix-v1'
    return shot


def test_review_tail_and_export_share_selected_prefix_and_cache(long_video, tmp_path, monkeypatch):
    from tools.frame_extractor import extract_frame
    original = hashlib.sha256(long_video.read_bytes()).hexdigest()
    sdk = LocalCompletions(outcome_response())
    reviewer = SemanticTakeReviewer(tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=sdk)), model='offline-vision')
    shot = selected_shot()
    assert reviewer.review(str(long_video), shot).accepted
    cache = json.loads(next((tmp_path / 'semantic_reviews').glob('*.json')).read_text())
    assert cache['media_duration'] == 5
    assert max(cache['sample_timestamps']) < 5
    calls = []
    def capture(video, output, timestamp=None):
        calls.append(timestamp)
        return extract_frame(video, output, timestamp)
    monkeypatch.setattr('pipeline.generator.extract_frame', capture)
    generator = object.__new__(VideoGenerator)
    generator.output_dir, generator.character_refs = tmp_path, {}
    generator.character_ref_hashes = {}
    (tmp_path / 'shots').mkdir()
    (tmp_path / 'character_refs').mkdir()
    generator._extract_local_tail_frame(1, str(long_video), shot)
    assert calls[-1] == cache['sample_timestamps'][-1]
    generator._register_identity_crops(shot_id=1, video_path=str(long_video),
        crop_boxes={'attacker': (0., 0., .5, 1.)}, shot=shot)
    assert calls[-1] == 2.5
    output = tmp_path / 'selected.mp4'
    normalize_video(str(long_video), str(output), resolution='320:180', duration=shot['duration'])
    stream = next(s for s in get_video_info(str(output))['streams'] if s['codec_type'] == 'video')
    assert float(stream['duration']) == 5 and int(stream['nb_frames']) == 120
    assert hashlib.sha256(long_video.read_bytes()).hexdigest() == original
    assert reviewer.review(str(long_video), shot).accepted
    assert len(sdk.calls) == 1


@pytest.mark.parametrize('source_duration', [4., 5., 5.088, 6.])
def test_selection_uses_existing_plan_without_padding_or_new_quota(source_duration):
    from pipeline.media_selection import selected_media_window
    window = selected_media_window(selected_shot(), source_duration)
    assert window['duration'] == min(5, source_duration)
    assert window['end_seconds'] == window['duration'] and window['planned_end_seconds'] == 5
    assert 0 <= window['last_sample_seconds'] < window['duration']
    assert window['midpoint_seconds'] == window['duration'] / 2


@pytest.mark.parametrize('invalid', [0, -1, True, float('nan'), float('inf')])
def test_invalid_media_extent_is_not_an_observation(invalid):
    from pipeline.media_selection import selected_media_window
    with pytest.raises(ValueError):
        selected_media_window(selected_shot(), invalid)


def test_result_only_after_cut_is_not_accepted_or_retried(long_video, tmp_path):
    from pipeline.semantic_review import SemanticReviewUnavailableError
    response = outcome_response()
    response['action_event_evidence'][-1].update(observed_effect_seconds=5.2,
        observed_reaction_seconds=5.4, observed_end_seconds=5.8)
    sdk = LocalCompletions(response)
    reviewer = SemanticTakeReviewer(tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=sdk)), model='offline-vision')
    with pytest.raises(SemanticReviewUnavailableError):
        reviewer.review(str(long_video), selected_shot())
    assert len(sdk.calls) == 1


def test_media_scope_changes_cache_identity_without_migrating_legacy(long_video, tmp_path):
    g = object.__new__(VideoGenerator)
    g.output_dir, g.character_refs = tmp_path, {}
    old = outcome_shot()
    original = g._acceptance_context(old, [], None)
    assert 'media_selection' not in original
    new = selected_shot()
    assert g._acceptance_context(new, [], None)['media_selection']['end_seconds'] == 5
    assert g._acceptance_context(old, [], None) == original
    sdk = LocalCompletions(outcome_response())
    reviewer = SemanticTakeReviewer(tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=sdk)), model='offline-vision')
    assert reviewer.review(str(long_video), old).accepted
    assert reviewer.review(str(long_video), new).accepted
    assert len(sdk.calls) == 2
    assert reviewer.review(str(long_video), new).accepted and len(sdk.calls) == 2
