"""转场推导逻辑测试 (P2.5)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ffmpeg_ops import concat_with_transitions, infer_transitions, _extract_direction


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


def test_hard_cut_alias_is_normalized_at_postprocessing_boundary():
    shots = [
        {"camera": {"speed": "slow"}, "transition_to_next": "hard cut"},
        {"camera": {"speed": "slow"}},
    ]

    assert infer_transitions(shots) == [("cut", 0.0)]


def test_mixed_transition_chain_preserves_real_hard_cut(monkeypatch):
    captured = {}
    monkeypatch.setattr("tools.ffmpeg_ops.get_video_duration", lambda _path: 5.0)
    monkeypatch.setattr(
        "tools.ffmpeg_ops.subprocess.run",
        lambda command, **_kwargs: captured.setdefault("command", command),
    )

    concat_with_transitions(
        ["one.mp4", "two.mp4", "three.mp4"],
        [("cut", 0.0), ("crossfade", 0.5)],
        "output.mp4",
    )

    command = captured["command"]
    filters = command[command.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=0" in filters
    assert "xfade=transition=fade:duration=0.5" in filters
    assert "duration=0.1" not in filters


def test_extract_direction():
    assert _extract_direction("slow push-in then rise") == "push-in"
    assert _extract_direction("gentle orbit around subject") == "orbit"
    assert _extract_direction("") == "unknown"
