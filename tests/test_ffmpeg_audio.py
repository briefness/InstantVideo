"""FFmpeg 音频处理测试 (P0.1: 拼接丢音频修复)

依赖 ffmpeg 系统命令, 会生成和清理临时文件。
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import ffmpeg_ops


@pytest.fixture
def work_dir(tmp_path):
    """临时工作目录"""
    return tmp_path


def _make_clip(path, with_audio, color="red", dur=2):
    """快速生成测试片段"""
    if with_audio:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s=320x180:d={dur}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s=320x180:d={dur}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ]
    subprocess.run(cmd, check=True, capture_output=True)


def test_normalize_adds_silent_track(work_dir):
    """无音频片段 normalize 后应有静音轨"""
    raw = work_dir / "raw.mp4"
    norm = work_dir / "norm.mp4"
    _make_clip(raw, with_audio=False)

    assert not ffmpeg_ops.has_audio_track(str(raw))
    ffmpeg_ops.normalize_video(str(raw), str(norm), resolution="320:180")
    assert ffmpeg_ops.has_audio_track(str(norm))


def test_normalize_preserves_existing_audio(work_dir):
    """有音频片段 normalize 后音频仍在"""
    raw = work_dir / "raw.mp4"
    norm = work_dir / "norm.mp4"
    _make_clip(raw, with_audio=True)

    assert ffmpeg_ops.has_audio_track(str(raw))
    ffmpeg_ops.normalize_video(str(raw), str(norm), resolution="320:180")
    assert ffmpeg_ops.has_audio_track(str(norm))


def test_concat_preserves_audio(work_dir):
    """拼接后成片应保留音频流 (核心回归测试)"""
    clips = []
    for i, (audio, color) in enumerate([(True, "red"), (False, "green"), (True, "blue")]):
        raw = work_dir / f"raw_{i}.mp4"
        norm = work_dir / f"norm_{i}.mp4"
        _make_clip(raw, audio, color, dur=2)
        ffmpeg_ops.normalize_video(str(raw), str(norm), resolution="320:180")
        clips.append(str(norm))

    out = work_dir / "concat.mp4"
    ffmpeg_ops.concat_with_transitions(clips, ["crossfade", "crossfade"], str(out))

    assert ffmpeg_ops.has_audio_track(str(out)), "拼接后音频丢失!"
    dur = ffmpeg_ops.get_video_duration(str(out))
    # 3×2s - 2×0.5s 转场重叠 ≈ 5s
    assert 4.5 < dur < 6.0, f"时长异常: {dur}"
