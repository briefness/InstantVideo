"""转场推导逻辑测试 (P2.5)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ffmpeg_ops import infer_transitions, _extract_direction


def test_empty_and_single():
    assert infer_transitions([]) == []
    assert infer_transitions([{"camera": {}}]) == []


def test_fast_motion_uses_cut():
    shots = [
        {"camera": {"primary_movement": "tracking", "speed": "fast"}},
        {"camera": {"primary_movement": "pan", "speed": "slow"}},
    ]
    result = infer_transitions(shots)
    assert result[0][0] == "cut"
    assert result[0][1] == 0.0


def test_crane_up_uses_fade_black():
    shots = [
        {"camera": {"primary_movement": "crane-up", "speed": "slow"}},
        {"camera": {"primary_movement": "fixed", "speed": "slow"}},
    ]
    result = infer_transitions(shots)
    assert result[0][0] == "fade_to_black"
    assert result[0][1] == 0.8


def test_same_direction_uses_dissolve():
    shots = [
        {"camera": {"primary_movement": "slow push-in", "speed": "slow"}},
        {"camera": {"primary_movement": "fast push-in", "speed": "slow"}},
    ]
    result = infer_transitions(shots)
    assert result[0][0] == "dissolve"


def test_direction_change_uses_crossfade():
    shots = [
        {"camera": {"primary_movement": "push-in", "speed": "slow"}},
        {"camera": {"primary_movement": "orbit", "speed": "slow"}},
    ]
    result = infer_transitions(shots)
    assert result[0][0] == "crossfade"


def test_explicit_transition_respected():
    shots = [
        {"camera": {"primary_movement": "push-in"}, "transition_to_next": "wipe_left"},
        {"camera": {"primary_movement": "pan"}},
    ]
    result = infer_transitions(shots)
    assert result[0][0] == "wipe_left"


def test_extract_direction():
    assert _extract_direction("slow push-in then rise") == "push-in"
    assert _extract_direction("gentle orbit around subject") == "orbit"
    assert _extract_direction("") == "unknown"
