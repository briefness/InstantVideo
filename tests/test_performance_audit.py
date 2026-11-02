import hashlib
import json
from pathlib import Path
import socket

import cv2
import numpy as np
import pytest

from tools.performance_audit import build_audit, controls, read_frames


def write_fixture(path, count=24):
    frame = np.random.default_rng(7).integers(0, 256, (96, 160, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 24, (160, 96))
    assert writer.isOpened()
    try:
        for index in range(count):
            moved = cv2.warpAffine(frame, np.float32([[1, 0, index], [0, 1, 0]]), (160, 96))
            writer.write(moved)
    finally:
        writer.release()


def test_control_sources_disclose_exact_edit():
    result = controls(120)
    assert result['baseline']['frames'] == list(range(120))
    assert result['inserted_hold']['frames'][60:84] == [60] * 24
    assert result['removed_interval']['frames'][59:61] == [59, 84]
    assert result['camera_only']['frames'] == [60] * 120


@pytest.mark.parametrize('count', [True, 11, 901, '24'])
def test_control_limits(count):
    with pytest.raises(ValueError):
        controls(count)


def test_bundle_keeps_source_and_runs_without_network(tmp_path, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError('Offline audit attempted network access')
    monkeypatch.setattr(socket.socket, 'connect', forbid)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbid)
    video = tmp_path / 'source.mp4'
    write_fixture(video)
    original = video.read_bytes()
    output = tmp_path / 'audit'
    report = build_audit(video, output)
    assert video.read_bytes() == original == (output / 'original.mp4').read_bytes()
    assert report['source_sha256'] == hashlib.sha256(original).hexdigest()
    assert report['remote_calls'] == report['paid_submissions'] == 0
    assert report['reviewer_calibration'] == 'not_established'
    assert json.loads((output / 'audit.json').read_text()) == report
    assert (output / 'frames_0000.jpg').is_file()
    assert (output / 'index.html').is_file()
    for name, entry in report['controls'].items():
        frames, times, _ = read_frames(output / f'{name}.mp4')
        assert len(frames) == len(times) == len(entry['source_frames'])
        assert entry['diagnostic']['action_fluency'] == 'not_evaluated'
        assert entry['technical_check']['measurement_scope'] == 'technical_image_checks'
        assert entry['technical_check']['frame_change_coverage'] == entry['technical_check']['motion_coverage']
    camera = report['controls']['camera_only']
    assert len(set(camera['source_frames'])) == 1
    # A fixed image can have full pixel-change coverage. That is the counterexample,
    # not a reason to infer independent actor motion from compensated residuals.
    assert camera['technical_check']['frame_change_coverage'] > .8
    with pytest.raises(ValueError, match='不覆盖'):
        build_audit(video, output)
    assert video.read_bytes() == original


def test_failed_decode_does_not_publish(tmp_path):
    video = tmp_path / 'bad.mp4'
    video.write_bytes(b'not a video')
    with pytest.raises(ValueError):
        build_audit(video, tmp_path / 'audit')
    assert not (tmp_path / 'audit').exists()


def test_publication_failure_leaves_no_partial_bundle(tmp_path, monkeypatch):
    video = tmp_path / 'source.mp4'
    write_fixture(video)
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr('tools.performance_audit.os.link', fail)
    with pytest.raises(OSError, match='disk full'):
        build_audit(video, tmp_path / 'audit')
    assert not (tmp_path / 'audit').exists()
    assert not list(tmp_path.glob('.performance-audit-*'))
