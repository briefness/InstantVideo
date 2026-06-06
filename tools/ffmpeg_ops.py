"""FFmpeg 操作合集 — 拼接/转场/调色/字幕/混音/导出

Cherry-picked patterns from OpenMontage video_stitch.py, simplified.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def get_video_duration(filepath: str) -> float:
    """获取视频时长 (秒)"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", filepath],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def get_video_info(filepath: str) -> dict:
    """获取视频完整信息"""
    import json
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", filepath],
        capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def has_audio_track(filepath: str) -> bool:
    """检查视频是否包含音频轨"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", filepath],
        capture_output=True, text=True,
    )
    return "audio" in result.stdout


# ─── 规格统一 ───

def normalize_video(
    input_path: str,
    output_path: str,
    fps: int = 24,
    resolution: str = "1920:1080",
    pixel_format: str = "yuv420p",
):
    """统一视频规格 (帧率/分辨率/编码)，确保后续拼接不报错"""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", (
            f"fps={fps},"
            f"scale={resolution}:force_original_aspect_ratio=decrease,"
            f"pad={resolution}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1:1"
        ),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", pixel_format,
        "-c:a", "aac", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ─── 视频拼接 & 转场 ───

TRANSITION_MAP = {
    "crossfade": "fade",
    "fade_to_black": "fadeblack",
    "fade_to_white": "fadewhite",
    "cut": None,
    "dissolve": "dissolve",
    "wipe_left": "wipeleft",
    "wipe_right": "wiperight",
    "slide_left": "slideleft",
}


def concat_with_transitions(
    video_files: list[str],
    transitions: list[str],
    output_path: str,
    transition_duration: float = 0.5,
):
    """使用 FFmpeg xfade 拼接视频片段并添加转场效果"""

    if len(video_files) == 0:
        raise ValueError("没有视频文件")

    if len(video_files) == 1:
        subprocess.run(["cp", video_files[0], output_path], check=True)
        return

    # 获取每个视频的时长
    durations = [get_video_duration(f) for f in video_files]

    # 检查是否全是硬切
    all_cuts = all(
        TRANSITION_MAP.get(transitions[i] if i < len(transitions) else "crossfade") is None
        for i in range(len(video_files) - 1)
    )
    if all_cuts:
        concat_simple(video_files, output_path)
        return

    # 构建 xfade filter chain
    # 4 个输入需要 3 次 xfade: [0][1]→[v0], [v0][2]→[v1], [v1][3]→[vout]
    filter_parts = []
    cumulative_offset = 0.0

    for i in range(len(video_files) - 1):
        transition_type = transitions[i] if i < len(transitions) else "crossfade"
        xfade_name = TRANSITION_MAP.get(transition_type, "fade") or "fade"

        # offset = 前面所有视频总时长 - 已使用的转场时长
        if i == 0:
            cumulative_offset = durations[0] - transition_duration
            prev_label = "[0:v]"
        else:
            cumulative_offset += durations[i] - transition_duration
            prev_label = f"[v{i-1}]"

        next_label = f"[{i+1}:v]"
        # 最后一次输出用 [vout]
        out_label = "[vout]" if i == len(video_files) - 2 else f"[v{i}]"

        filter_parts.append(
            f"{prev_label}{next_label}xfade=transition={xfade_name}"
            f":duration={transition_duration}:offset={cumulative_offset:.3f}{out_label}"
        )

    filter_complex = ";".join(filter_parts)

    # 构建命令
    cmd = ["ffmpeg", "-y"]
    for f in video_files:
        cmd += ["-i", f]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def concat_simple(video_files: list[str], output_path: str):
    """简单拼接 (无转场, 使用 concat demuxer)"""
    list_file = str(Path(output_path).parent / "concat_list.txt")
    with open(list_file, "w") as f:
        for vf in video_files:
            f.write(f"file '{os.path.abspath(vf)}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    os.remove(list_file)


# ─── 调色 ───

def apply_lut(
    input_path: str,
    lut_path: str,
    output_path: str,
    intensity: float = 0.80,
):
    """应用 3D LUT 调色 (支持强度控制)"""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter_complex",
        (
            f"[0:v]split[a][b];"
            f"[a]lut3d=file={lut_path}[graded];"
            f"[b][graded]blend=all_mode=normal:all_opacity={intensity}[vout]"
        ),
        "-map", "[vout]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ─── 音频混合 ───

def add_background_music(
    video_path: str,
    music_path: str,
    output_path: str,
    music_volume: float = 0.25,
    fade_in: float = 2.0,
    fade_out: float = 3.0,
):
    """为视频添加背景音乐 (音量调节 + 淡入淡出)"""
    video_duration = get_video_duration(video_path)
    has_audio = has_audio_track(video_path)

    if has_audio:
        # 视频已有音频 (Seedance 原生) → 混音
        filter_complex = (
            f"[1:a]atrim=0:{video_duration},"
            f"afade=t=in:st=0:d={fade_in},"
            f"afade=t=out:st={video_duration - fade_out}:d={fade_out},"
            f"volume={music_volume}[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=3[aout]"
        )
    else:
        # 视频无音频 → 直接加音乐
        filter_complex = (
            f"[1:a]atrim=0:{video_duration},"
            f"afade=t=in:st=0:d={fade_in},"
            f"afade=t=out:st={video_duration - fade_out}:d={fade_out},"
            f"volume={music_volume}[aout]"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ─── 字幕 ───

def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    font_name: str = "PingFang SC",
    font_size: int = 24,
):
    """硬字幕烧录 — 使用 -i 输入 SRT 避免路径转义问题"""
    force_style = (
        f"FontName={font_name},"
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"Outline=2,Shadow=1,"
        f"Alignment=2,MarginV=30"
    )

    # 用 FFmpeg 的 -i 读取 SRT 作为第二输入流, 再用 subtitles 滤镜按流索引引用
    # 这样完全避免路径转义问题
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", srt_path,
        "-map", "0:v", "-map", "0:a?",
        "-vf", f"ass={srt_path}" if srt_path.endswith('.ass') else
               f"subtitles=filename={srt_path}:force_style='{force_style}'",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # 备选方案: 用 drawtext 逐条渲染 (最兼容)
        print(f"  ⚠ subtitles 滤镜失败, 尝试 drawtext 方案...")
        _burn_with_drawtext(video_path, srt_path, output_path, font_name, font_size)


def _burn_with_drawtext(video_path: str, srt_path: str, output_path: str,
                         font_name: str, font_size: int):
    """备选字幕方案: 解析 SRT 后用 drawtext 滤镜逐条渲染 (不依赖 libass)"""
    import pysrt

    try:
        subs = pysrt.open(srt_path, encoding='utf-8')
    except Exception:
        print(f"  ⚠ SRT 解析失败, 跳过字幕")
        subprocess.run(["cp", video_path, output_path], check=True)
        return

    if not subs:
        subprocess.run(["cp", video_path, output_path], check=True)
        return

    # 构建 drawtext filter chain
    drawtext_filters = []
    for sub in subs:
        start = sub.start.ordinal / 1000.0
        end = sub.end.ordinal / 1000.0
        text = sub.text.replace("'", "'\\''").replace(":", "\\:")

        drawtext_filters.append(
            f"drawtext=text='{text}'"
            f":fontsize={font_size}:fontcolor=white:borderw=2:bordercolor=black"
            f":x=(w-text_w)/2:y=h-th-40"
            f":enable='between(t,{start:.2f},{end:.2f})'"
        )

    if not drawtext_filters:
        subprocess.run(["cp", video_path, output_path], check=True)
        return

    vf = ",".join(drawtext_filters)

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠ drawtext 也失败, 跳过字幕")
        subprocess.run(["cp", video_path, output_path], check=True)


# ─── 片头片尾 ───

def generate_title_card(
    title: str,
    subtitle: str = "",
    output_path: str = "output/title.mp4",
    duration: float = 3.0,
    resolution: str = "1920:1080",
):
    """生成片头黑底白字定版 (支持中文)"""
    # macOS 字体路径
    font_file = "/System/Library/Fonts/PingFang.ttc"

    # FFmpeg filtergraph 中冒号是参数分隔符, color 源的 s= 子选项
    # 必须用 "1920x1080" 格式, 不能用 "1920:1080" (会被误解析为多个选项)
    size = resolution.replace(":", "x")

    # 转义 FFmpeg drawtext 特殊字符
    safe_title = title.replace(":", "\\:").replace("'", "'\\''")
    safe_subtitle = subtitle.replace(":", "\\:").replace("'", "'\\''") if subtitle else ""

    filter_parts = [
        f"color=c=black:s={size}:d={duration}[bg]",
        (
            f"[bg]drawtext=fontfile='{font_file}'"
            f":text='{safe_title}'"
            f":fontsize=64:fontcolor=white"
            f":x=(w-text_w)/2:y=(h-text_h)/2-30"
            f":alpha='if(lt(t,1),t,1)'[t1]"
        ),
    ]

    if safe_subtitle:
        filter_parts.append(
            f"[t1]drawtext=fontfile='{font_file}'"
            f":text='{safe_subtitle}'"
            f":fontsize=32:fontcolor=white@0.7"
            f":x=(w-text_w)/2:y=(h/2)+40"
            f":alpha='if(lt(t,1.5),t/1.5,1)'[vout]"
        )
    else:
        filter_parts.append("[t1]copy[vout]")

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # 片头失败不致命, 生成纯黑片头
        print(f"  ⚠ 片头生成失败, 使用纯黑替代: {result.stderr[-100:]}")
        cmd_fallback = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={size}:d={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path,
        ]
        subprocess.run(cmd_fallback, check=True, capture_output=True)


# ─── 多平台导出 ───

PLATFORM_SPECS = {
    "youtube": {"resolution": "1920:1080", "ratio": "16:9"},
    "tiktok": {"resolution": "1080:1920", "ratio": "9:16"},
    "bilibili": {"resolution": "1920:1080", "ratio": "16:9"},
    "xiaohongshu": {"resolution": "1080:1440", "ratio": "3:4"},
    "instagram_reels": {"resolution": "1080:1920", "ratio": "9:16"},
    "instagram_feed": {"resolution": "1080:1080", "ratio": "1:1"},
}


def export_for_platform(
    source: str,
    platform: str,
    output_path: str,
    crop: str = "center",
):
    """导出指定平台规格的视频"""
    spec = PLATFORM_SPECS[platform]
    res = spec["resolution"]

    if crop == "center":
        vf = f"scale={res}:force_original_aspect_ratio=increase,crop={res}"
    else:
        vf = (
            f"scale={res}:force_original_aspect_ratio=decrease,"
            f"pad={res}:(ow-iw)/2:(oh-ih)/2:color=black"
        )

    cmd = [
        "ffmpeg", "-y", "-i", source,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
