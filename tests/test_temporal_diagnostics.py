import cv2
import numpy as np
import pytest

from pipeline.temporal_diagnostics import frame_change, measure_frames


@pytest.fixture
def texture():
    noise = np.random.default_rng(912).integers(0, 256, (160, 240), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (3, 3), .7)


def test_global_translation_is_not_independent_action(texture):
    moved = cv2.warpAffine(texture, np.float32([[1, 0, 3], [0, 1, 0]]), (240, 160))
    result = frame_change(texture, moved, .04)
    assert result['registration'] == 'estimated'
    assert result['raw_change'] > 10
    assert result['aligned_change'] < result['raw_change'] * .1
    assert result['dominant_speed_heights_per_second'] == pytest.approx(3 / 160 / .04, rel=.05)
    slower = frame_change(texture, moved, .08)
    assert slower['dominant_speed_heights_per_second'] == pytest.approx(
        result['dominant_speed_heights_per_second'] / 2, rel=.05)


def test_local_change_survives_dominant_translation(texture):
    moved = cv2.warpAffine(texture, np.float32([[1, 0, 3], [0, 1, 0]]), (240, 160))
    camera_only = frame_change(texture, moved, .04)
    moved[50:90, 90:140] = 240
    local = frame_change(texture, moved, .04)
    assert local['registration'] == 'estimated'
    assert local['aligned_change'] > camera_only['aligned_change'] + 2


def test_flat_image_is_missing_registration_not_zero_action():
    flat = np.full((64, 64), 127, dtype=np.uint8)
    result = measure_frames([flat, flat], [0., .04])
    assert result['action_fluency'] == 'not_evaluated'
    assert result['summary']['estimated_pairs'] == 0
    assert result['summary']['aligned_change_median'] is None
    assert result['summary']['exact_duplicate_pairs'] == 1


def test_exact_hold_and_nonuniform_times(texture):
    result = measure_frames([texture] * 3, [0., .04, .11])
    assert result['summary']['exact_duplicate_pairs'] == 2
    assert result['pairs'][1]['end_seconds'] == .11
    assert result['summary']['raw_change_median'] == 0


@pytest.mark.parametrize('times', [[0, 0], [0, -1], [0, float('nan')], [0, float('inf')],
                                   [False, 1], [0], ['0', 1]])
def test_invalid_time_evidence_is_rejected(texture, times):
    with pytest.raises(ValueError):
        measure_frames([texture, texture], times)


@pytest.mark.parametrize('delta', [0, -1, True, float('nan'), float('inf'), '1'])
def test_invalid_delta(texture, delta):
    with pytest.raises(ValueError):
        frame_change(texture, texture, delta)


@pytest.mark.parametrize('bad', [None, np.zeros((10, 10), np.uint8), np.zeros((160, 240)),
                                np.zeros((160, 240, 3), np.uint8), np.zeros((700, 700), np.uint8)])
def test_invalid_images(texture, bad):
    with pytest.raises(ValueError):
        frame_change(bad, texture, .04)
