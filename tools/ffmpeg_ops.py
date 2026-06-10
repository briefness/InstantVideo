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
    """统一视频规格 (帧率/分辨率/编码)，确保后续拼接不报错

    关键: 始终保证输出带一条音轨。有原生音频则重编码；无音频则补静音轨,
    否则后续 acrossfade 拼接会因某些片段缺音频流而报错。
    """
    vf = (
        f"fps={fps},"
        f"scale={resolution}:force_original_aspect_ratio=decrease,"
        f"pad={resolution}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1:1"
    )

    if has_audio_track(input_path):
        # 有原生音频 → 正常重编码
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", pixel_format,
            "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        # 无音频 → 用 anullsrc 补一条与视频等长的静音轨
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf", vf,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", pixel_format,
            "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-shortest",  # 静音轨随视频长度截断
            "-movflags", "+faststart",
            output_path,
        ]
    subprocess.run(cmd, check=True, capture_output=True)


# ─── 视频拼接 & 转场 ───


def infer_transitions(shots: list[dict]) -> list[tuple[str, float]]:
    """根据相邻镜头运动方向智能推导转场类型和时长

    返回 [(transition_type, duration), ...]

    时长策略:
    - cut: 0 (硬切无转场)
    - 快速运动: 0.3s (短促利落)
    - 慢速/升华: 0.8s (柔和过渡)
    - 默认: 0.5s
    """
    if len(shots) < 2:
        return []

    transitions = []
    for i in range(len(shots) - 1):
        current = shots[i]
        next_shot = shots[i + 1]

        # 提取运动信息
        cam_cur = current.get("camera", {})
        cam_nxt = next_shot.get("camera", {})
        move_cur = cam_cur.get("primary_movement", "").lower()
        move_nxt = cam_nxt.get("primary_movement", "").lower()
        speed_cur = cam_cur.get("speed", "").lower()
        speed_nxt = cam_nxt.get("speed", "").lower()

        # 如果分镜明确指定了非默认转场, 尊重它
        explicit = current.get("transition_to_next", "")
        if explicit and explicit != "crossfade":
            dur = _speed_to_transition_duration(speed_cur, speed_nxt, explicit)
            transitions.append((explicit, dur))
            continue

        # 快速运动之间 → 硬切
        if speed_cur in ("fast", "quick", "rapid") or "fast" in move_cur:
            transitions.append(("cut", 0.0))
        # 收束/升华镜头 (pull-out, crane-up, pull-back) → 渐黑
        elif any(kw in move_cur for kw in ("pull-out", "pull-back", "crane-up", "drone-up")):
            transitions.append(("fade_to_black", 0.8))
        # 相同运动方向衔接 → dissolve
        elif move_cur and move_nxt and _extract_direction(move_cur) == _extract_direction(move_nxt):
            transitions.append(("dissolve", 0.5))
        # 方向突变 → 标准 fade
        elif move_cur and move_nxt and _extract_direction(move_cur) != _extract_direction(move_nxt):
            transitions.append(("crossfade", 0.5))
        # 默认
        else:
            transitions.append(("crossfade", 0.5))

    return transitions


def _speed_to_transition_duration(
    speed_cur: str, speed_nxt: str, transition_type: str
) -> float:
    """根据相邻镜头速度推导转场时长"""
    if transition_type == "cut":
        return 0.0
    # 任一侧是快速 → 短转场
    fast_keywords = {"fast", "quick", "rapid"}
    if speed_cur in fast_keywords or speed_nxt in fast_keywords:
        return 0.3
    # 任一侧是慢速 → 长转场
    slow_keywords = {"slow", "gentle", "smooth"}
    if speed_cur in slow_keywords or speed_nxt in slow_keywords:
        return 0.8
    return 0.5


def _extract_direction(movement: str) -> str:
    """从运镜描述中提取核心方向关键词"""
    for d in ("push-in", "pull-out", "pull-back", "pan-left", "pan-right",
              "pan", "orbit", "tracking", "crane-up", "crane-down",
              "drone", "fixed", "handheld", "tilt-up", "tilt-down"):
        if d in movement:
            return d
    return movement.split()[0] if movement else "unknown"


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
    transitions: list[tuple[str, float]],
    output_path: str,
):
    """使用 FFmpeg xfade 拼接视频片段并添加转场效果

    Args:
        transitions: [(type, duration), ...] — 由 infer_transitions 返回
    """

    if len(video_files) == 0:
        raise ValueError("没有视频文件")

    if len(video_files) == 1:
        subprocess.run(["cp", video_files[0], output_path], check=True)
        return

    # 获取每个视频的时长
    durations = [get_video_duration(f) for f in video_files]

    # 提取类型和时长
    t_types = [t[0] if i < len(transitions) else "crossfade"
               for i, t in enumerate(transitions)]
    t_durs = [t[1] if i < len(transitions) else 0.5
              for i, t in enumerate(transitions)]

    # 检查是否全是硬切
    all_cuts = all(
        TRANSITION_MAP.get(t_types[i]) is None
        for i in range(len(video_files) - 1)
    )
    if all_cuts:
        concat_simple(video_files, output_path)
        return

    # 构建 xfade filter chain (视频) + acrossfade chain (音频)
    v_parts = []
    a_parts = []
    cumulative_offset = 0.0

    for i in range(len(video_files) - 1):
        transition_type = t_types[i] if i < len(t_types) else "crossfade"
        transition_duration = t_durs[i] if i < len(t_durs) else 0.5
        # 确保 transition_duration > 0 (硬切前面已过滤, 这里兜底)
        transition_duration = max(transition_duration, 0.1)
        xfade_name = TRANSITION_MAP.get(transition_type, "fade") or "fade"

        # offset = 前面所有视频总时长 - 已使用的转场时长
        if i == 0:
            cumulative_offset = durations[0] - transition_duration
            v_prev = "[0:v]"
            a_prev = "[0:a]"
        else:
            cumulative_offset += durations[i] - transition_duration
            v_prev = f"[v{i-1}]"
            a_prev = f"[a{i-1}]"

        v_next = f"[{i+1}:v]"
        a_next = f"[{i+1}:a]"
        # 最后一次输出用 [vout]/[aout]
        is_last = i == len(video_files) - 2
        v_out = "[vout]" if is_last else f"[v{i}]"
        a_out = "[aout]" if is_last else f"[a{i}]"

        v_parts.append(
            f"{v_prev}{v_next}xfade=transition={xfade_name}"
            f":duration={transition_duration}:offset={cumulative_offset:.3f}{v_out}"
        )
        # acrossfade 把两段音频交叉淡化拼接, 时长与视频转场一致
        a_parts.append(
            f"{a_prev}{a_next}acrossfade=d={transition_duration}{a_out}"
        )

    filter_complex = ";".join(v_parts + a_parts)

    # 构建命令
    cmd = ["ffmpeg", "-y"]
    for f in video_files:
        cmd += ["-i", f]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def concat_simple(video_files: list[str], output_path: str):
    """简单拼接 (无转场, 使用 concat demuxer)

    优先尝试 -c copy 零损耗拼接 (当所有片段编码一致时);
    如果 copy 失败 (编码不一致), 回退到 re-encode。
    """
    list_file = str(Path(output_path).parent / "concat_list.txt")
    with open(list_file, "w") as f:
        for vf in video_files:
            f.write(f"file '{os.path.abspath(vf)}'\n")

    # 先尝试 -c copy (零损耗, 快速)
    cmd_copy = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd_copy, capture_output=True, text=True)
    if result.returncode == 0:
        os.remove(list_file)
        return

    # copy 失败 → re-encode (编码不一致时的兜底)
    cmd_reencode = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd_reencode, check=True, capture_output=True)
    os.remove(list_file)


# ─── 帧提取工具 ───

def extract_middle_frame(video_path: str, output_path: str):
    """从视频中间位置提取一帧 (用于片头背景等)"""
    dur = get_video_duration(video_path)
    ts = dur / 2
    cmd = [
        "ffmpeg", "-y", "-ss", f"{ts:.2f}",
        "-i", video_path, "-frames:v", "1",
        "-q:v", "2", output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ─── 调色 ───

def _escape_filter_path(path: str) -> str:
    r"""转义 FFmpeg filter 中的路径特殊字符

    FFmpeg filtergraph 中 : ; ' \ [ ] 等都是特殊字符,
    需要用反斜杠转义。
    """
    # FFmpeg filter 路径中需要转义的字符
    for ch in ("'", "\\", ":", ";", "[", "]"):
        path = path.replace(ch, f"\\{ch}")
    # 空格也需要转义
    path = path.replace(" ", "\\ ")
    return path


def apply_lut(
    input_path: str,
    lut_path: str,
    output_path: str,
    intensity: float = 0.80,
):
    """应用 3D LUT 调色 (支持强度控制)"""
    safe_lut = _escape_filter_path(lut_path)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter_complex",
        (
            f"[0:v]split[a][b];"
            f"[a]lut3d=file={safe_lut}[graded];"
            f"[b][graded]blend=all_mode=normal:all_opacity={intensity}[vout]"
        ),
        "-map", "[vout]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def apply_visual_filters(
    input_path: str,
    output_path: str,
    lut_path: str | None = None,
    srt_path: str | None = None,
    lut_intensity: float = 0.80,
    font_name: str = "PingFang SC",
    font_size: int = 24,
):
    """合并 LUT 调色 + 字幕烧录为单次编码, 减少画质损失

    单次编码 vs 分两步:
    - 分两步: concat → LUT(CRF18) → subtitle(CRF18) = 2 次编码
    - 合并后: concat → LUT+subtitle(CRF18) = 1 次编码
    """
    vf_parts = []

    # LUT 调色 (如果有)
    if lut_path and Path(lut_path).exists():
        safe_lut = _escape_filter_path(lut_path)
        # split → lut3d → blend 实现可控强度
        vf_parts.append(
            f"split[_a][_b];"
            f"[_a]lut3d=file={safe_lut}[_graded];"
            f"[_b][_graded]blend=all_mode=normal:all_opacity={lut_intensity}"
        )

    # 字幕烧录 (如果有)
    if srt_path and Path(srt_path).exists() and Path(srt_path).stat().st_size > 10:
        force_style = (
            f"FontName={font_name},"
            f"FontSize={font_size},"
            f"PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,"
            f"BackColour=&H80000000,"
            f"BorderStyle=4,"
            f"Outline=1,Shadow=0,"
            f"Alignment=2,MarginV=30"
        )
        vf_parts.append(f"subtitles=filename={srt_path}:force_style='{force_style}'")

    if not vf_parts:
        # 无滤镜需要应用, 直接复制
        subprocess.run(["cp", input_path, output_path], check=True)
        return

    # 构建 filter: 如果只有一个滤镜直接用 -vf, 多个用 -filter_complex
    if len(vf_parts) == 1 and "split" not in vf_parts[0]:
        # 简单滤镜 (仅字幕)
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", vf_parts[0],
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
    else:
        # 复杂滤镜 (LUT 或 LUT+字幕)
        if len(vf_parts) == 1:
            # 仅 LUT
            filter_complex = f"[0:v]{vf_parts[0]}[vout]"
        else:
            # LUT + 字幕: LUT 输出 → 字幕滤镜
            filter_complex = f"[0:v]{vf_parts[0]}[_lut];[_lut]{vf_parts[1]}[vout]"

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠ 合并视觉滤镜失败, 回退分步处理: {result.stderr[-200:]}")
        # 回退: 分步处理
        intermediate = input_path
        if lut_path and Path(lut_path).exists():
            lut_out = str(Path(output_path).parent / "_lut_temp.mp4")
            apply_lut(input_path, lut_path, lut_out, lut_intensity)
            intermediate = lut_out
        if srt_path and Path(srt_path).exists() and Path(srt_path).stat().st_size > 10:
            burn_subtitles(intermediate, srt_path, output_path, font_name, font_size)
        elif intermediate != output_path:
            subprocess.run(["cp", intermediate, output_path], check=True)
        # 清理临时文件
        temp_file = Path(output_path).parent / "_lut_temp.mp4"
        if temp_file.exists():
            temp_file.unlink()


# ─── 音频混合 ───

def add_background_music(
    video_path: str,
    music_path: str,
    output_path: str,
    music_volume: float = 0.35,
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


def mix_voiceover_with_ducking(
    video_path: str,
    voiceover_segments: list[dict],
    output_path: str,
    vo_volume: float = 1.0,
    ducking_ratio: float = 4.0,
):
    """将口播语音混入视频，带 ducking（人声出现时自动压低背景）

    voiceover_segments: [{"audio_path": str, "start_time": float, "duration": float}, ...]
    ducking_ratio: 侧链压缩比，越大 BGM 压得越狠 (推荐 3-6)
    """
    if not voiceover_segments:
        # 无口播，直接复制
        subprocess.run(["cp", video_path, output_path], check=True)
        return

    video_duration = get_video_duration(video_path)

    # 策略: 先把所有口播片段用 adelay+amix 合并成一条完整的口播轨,
    # 再用 sidechaincompress 让口播轨控制背景音量 (ducking)。
    #
    # 输入流:
    #   0 = 视频 (含 BGM + 原生音频)
    #   1..N = 各口播音频

    cmd = ["ffmpeg", "-y", "-i", video_path]
    for seg in voiceover_segments:
        cmd += ["-i", seg["audio_path"]]

    n = len(voiceover_segments)

    # 构建 filter_complex
    parts = []

    # 1. 每段口播做延迟定位 + 音量
    for i, seg in enumerate(voiceover_segments):
        delay_ms = int(seg["start_time"] * 1000)
        parts.append(
            f"[{i+1}:a]adelay={delay_ms}|{delay_ms},"
            f"apad=whole_dur={video_duration},"
            f"volume={vo_volume}[vo{i}]"
        )

    # 2. 所有口播混合成一条轨
    if n == 1:
        parts.append(f"[vo0]acopy[voall]")
    else:
        vo_inputs = "".join(f"[vo{i}]" for i in range(n))
        parts.append(
            f"{vo_inputs}amix=inputs={n}:duration=longest:normalize=0[voall]"
        )

    # 3. 用 sidechaincompress 做 ducking: 口播轨作为侧链输入,
    #    控制视频原始音频的压缩量。人声出现时背景被压低。
    parts.append(
        f"[0:a][voall]sidechaincompress="
        f"threshold=0.02:ratio={ducking_ratio}:attack=200:release=1000[bg_ducked]"
    )

    # 4. 把 ducked 背景 + 口播混在一起
    parts.append(
        f"[bg_ducked][voall]amix=inputs=2:duration=first:normalize=0[aout]"
    )

    filter_complex = ";".join(parts)

    cmd += [
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
        f"BackColour=&H80000000,"
        f"BorderStyle=4,"
        f"Outline=1,Shadow=0,"
        f"Alignment=2,MarginV=30"
    )

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
        print(f"  ⚠ subtitles 滤镜失败, 尝试 drawtext 方案...")
        _burn_with_drawtext(video_path, srt_path, output_path, font_name, font_size)


def _burn_with_drawtext(video_path: str, srt_path: str, output_path: str,
                         font_name: str, font_size: int):
    """备选字幕方案: 解析 SRT 后用 drawtext 滤镜逐条渲染 (不依赖 libass)"""
    import pysrt

    try:
        subs = pysrt.open(srt_path, encoding='utf-8')
    except Exception:
        print("  ⚠ SRT 解析失败, 跳过字幕")
        subprocess.run(["cp", video_path, output_path], check=True)
        return

    if not subs:
        subprocess.run(["cp", video_path, output_path], check=True)
        return

    drawtext_filters = []
    for sub in subs:
        start = sub.start.ordinal / 1000.0
        end = sub.end.ordinal / 1000.0
        text = sub.text.replace("'", "'\\''" ).replace(":", "\\:")

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
        print("  ⚠ drawtext 也失败, 跳过字幕")
        subprocess.run(["cp", video_path, output_path], check=True)


# ─── 片头片尾 ───

def generate_title_card(
    title: str,
    subtitle: str = "",
    output_path: str = "output/title.mp4",
    duration: float = 3.0,
    resolution: str = "1920:1080",
    **_kwargs,
):
    """生成片头黑底白字定版 (支持中文)

    片头也输出静音 AAC 轨。否则它与后续正片拼接时，concat demuxer
    会按第一段流结构输出，导致整条成片音频被丢弃。
    """
    font_file = "/System/Library/Fonts/PingFang.ttc"
    size = resolution.replace(":", "x")

    safe_title = title.replace(":", "\\:").replace("'", "'\\''" )
    safe_subtitle = subtitle.replace(":", "\\:").replace("'", "'\\''" ) if subtitle else ""

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

    filter_parts.append(
        f"anullsrc=channel_layout=stereo:sample_rate=48000,"
        f"atrim=0:{duration}[aout]"
    )

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-t", str(duration),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠ 片头生成失败, 使用纯黑替代: {result.stderr[-100:]}")
        cmd_fallback = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={size}:d={duration}",
            "-f", "lavfi", "-i", f"anullsrc=channel_layout=stereo:sample_rate=48000",
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart",
            output_path,
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


def validate_publish_ready(
    filepath: str,
    *,
    platform: str | None = None,
    expected_duration: float | None = None,
    duration_tolerance: float = 1.0,
    require_audio: bool = True,
    min_duration: float = 1.0,
) -> dict:
    """校验导出文件是否达到可发布的基础技术门槛。

    这不是审美/内容审核，而是 CI 级 guardrail: 文件可读、有视频流、
    有音频流、时长合理、平台尺寸正确、像素格式/像素比例适合主流平台。
    校验失败会抛 RuntimeError，避免坏文件继续流到人工发布环节。
    """
    path = Path(filepath)
    issues: list[str] = []

    if not path.exists():
        raise RuntimeError(f"发布校验失败: 文件不存在: {filepath}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"发布校验失败: 文件为空: {filepath}")

    try:
        info = get_video_info(filepath)
    except Exception as exc:
        raise RuntimeError(f"发布校验失败: ffprobe 无法读取 {filepath}: {exc}") from exc

    streams = info.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        issues.append("缺少视频流")
    if require_audio and not audio_streams:
        issues.append("缺少音频流")

    duration = float(info.get("format", {}).get("duration") or 0.0)
    if duration < min_duration:
        issues.append(f"时长过短: {duration:.2f}s < {min_duration:.2f}s")
    if expected_duration is not None:
        delta = abs(duration - expected_duration)
        if delta > duration_tolerance:
            issues.append(
                f"时长偏差过大: 实际 {duration:.2f}s, 预期 {expected_duration:.2f}s, "
                f"偏差 {delta:.2f}s"
            )

    report = {
        "path": filepath,
        "duration": duration,
        "has_audio": bool(audio_streams),
        "video_codec": None,
        "audio_codec": audio_streams[0].get("codec_name") if audio_streams else None,
        "width": None,
        "height": None,
        "pix_fmt": None,
        "sample_aspect_ratio": None,
    }

    if video_streams:
        video = video_streams[0]
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        pix_fmt = video.get("pix_fmt")
        sar = video.get("sample_aspect_ratio")

        report.update({
            "video_codec": video.get("codec_name"),
            "width": width,
            "height": height,
            "pix_fmt": pix_fmt,
            "sample_aspect_ratio": sar,
        })

        if video.get("codec_name") != "h264":
            issues.append(f"视频编码不是 h264: {video.get('codec_name')}")
        if pix_fmt != "yuv420p":
            issues.append(f"像素格式不是 yuv420p: {pix_fmt}")
        if sar not in (None, "1:1"):
            issues.append(f"像素比例不是 1:1: {sar}")

        if platform:
            spec = PLATFORM_SPECS.get(platform)
            if not spec:
                issues.append(f"未知平台: {platform}")
            else:
                expected_width, expected_height = _parse_resolution(spec["resolution"])
                if (width, height) != (expected_width, expected_height):
                    issues.append(
                        f"{platform} 尺寸不匹配: 实际 {width}x{height}, "
                        f"预期 {expected_width}x{expected_height}"
                    )

    if audio_streams and audio_streams[0].get("codec_name") != "aac":
        issues.append(f"音频编码不是 aac: {audio_streams[0].get('codec_name')}")

    if issues:
        joined = "; ".join(issues)
        raise RuntimeError(f"发布校验失败 ({filepath}): {joined}")

    return report


def _parse_resolution(resolution: str) -> tuple[int, int]:
    width, height = resolution.split(":")
    return int(width), int(height)


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
        vf = f"scale={res}:force_original_aspect_ratio=increase,crop={res},setsar=1"
    else:
        vf = (
            f"scale={res}:force_original_aspect_ratio=decrease,"
            f"pad={res}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
        )

    if has_audio_track(source):
        cmd = [
            "ffmpeg", "-y", "-i", source,
            "-vf", vf,
            "-map", "0:v:0", "-map", "0:a",
            "-c:v", "libx264", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", source,
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf", vf,
            "-map", "0:v:0", "-map", "1:a",
            "-c:v", "libx264", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
    subprocess.run(cmd, check=True, capture_output=True)
