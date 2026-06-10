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
    ffmpeg_ops.concat_with_transitions(
        clips, [("crossfade", 0.5), ("crossfade", 0.5)], str(out)
    )

    assert ffmpeg_ops.has_audio_track(str(out)), "拼接后音频丢失!"
    dur = ffmpeg_ops.get_video_duration(str(out))
    # 3×2s - 2×0.5s 转场重叠 ≈ 5s (允许轻微编码偏差)
    assert 4.5 < dur < 6.5, f"时长异常: {dur}"


def test_title_card_does_not_drop_final_audio(work_dir):
    """片头 + 正片拼接后仍应保留音频, 否则导出版无法直接发布。"""
    title = work_dir / "title.mp4"
    body = work_dir / "body.mp4"
    final = work_dir / "final.mp4"

    ffmpeg_ops.generate_title_card(
        "测试片头",
        output_path=str(title),
        duration=0.5,
        resolution="320:180",
    )
    _make_clip(body, with_audio=True, color="blue", dur=1)

    assert ffmpeg_ops.has_audio_track(str(title))
    ffmpeg_ops.concat_simple([str(title), str(body)], str(final))
    assert ffmpeg_ops.has_audio_track(str(final))


def test_platform_export_preserves_audio_and_square_pixels(work_dir):
    """平台导出应保留音频并写入 setsar=1。"""
    source = work_dir / "source.mp4"
    exported = work_dir / "tiktok.mp4"
    _make_clip(source, with_audio=True, color="purple", dur=1)

    ffmpeg_ops.export_for_platform(str(source), "tiktok", str(exported))

    assert ffmpeg_ops.has_audio_track(str(exported))
    info = ffmpeg_ops.get_video_info(str(exported))
    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert video_stream["width"] == 1080
    assert video_stream["height"] == 1920
    assert video_stream["sample_aspect_ratio"] == "1:1"

    report = ffmpeg_ops.validate_publish_ready(str(exported), platform="tiktok")
    assert report["has_audio"]
    assert report["width"] == 1080
    assert report["height"] == 1920


def test_platform_export_adds_silent_audio_when_missing(work_dir):
    """源文件没有音轨时, 平台导出应补静音轨。"""
    source = work_dir / "source_no_audio.mp4"
    exported = work_dir / "youtube.mp4"
    _make_clip(source, with_audio=False, color="green", dur=1)

    ffmpeg_ops.export_for_platform(str(source), "youtube", str(exported))

    assert ffmpeg_ops.has_audio_track(str(exported))


def test_publish_validation_rejects_missing_audio(work_dir):
    """发布校验应拦截无音频输出。"""
    source = work_dir / "source_no_audio.mp4"
    _make_clip(source, with_audio=False, color="yellow", dur=1)

    with pytest.raises(RuntimeError, match="缺少音频流"):
        ffmpeg_ops.validate_publish_ready(str(source))


def test_publish_validation_rejects_duration_mismatch(work_dir):
    """发布校验应拦截明显的时长漂移。"""
    source = work_dir / "source.mp4"
    _make_clip(source, with_audio=True, color="orange", dur=1)

    with pytest.raises(RuntimeError, match="时长偏差过大"):
        ffmpeg_ops.validate_publish_ready(
            str(source),
            expected_duration=5.0,
            duration_tolerance=0.5,
        )
