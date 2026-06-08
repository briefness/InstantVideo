"""TTS 语音合成 — 可插拔引擎

默认使用 macOS say 命令跑通全链路；
火山语音技术密钥到位后，在 .env 中设 TTS_ENGINE=volcano 即可切换。

全链路:
  分镜 subtitle_text → 逐句合成语音 → 返回 [(音频文件, 真实时长)]
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Protocol

import config


@dataclass
class TTSSegment:
    """单条口播片段"""
    shot_id: int
    text: str
    audio_path: str
    duration: float   # 真实语音时长 (秒)


class TTSEngine(Protocol):
    """TTS 引擎接口 — 实现此协议即可接入"""
    async def synthesize(self, text: str, output_path: str) -> float:
        """合成语音，返回音频时长 (秒)"""
        ...


# ─── macOS say 引擎 ───

class MacOSSayEngine:
    """使用 macOS 内置 say 命令合成中文语音

    优点: 零配置、零成本、立即可用
    缺点: 音质机械，仅限 macOS
    """

    def __init__(self, voice: str = "Tingting"):
        """
        voice: macOS 中文语音名称。常见选项:
          - Tingting: 标准中文女声
          - Eddy, Flo, Reed: macOS 15+ 新增中文声音
        """
        self.voice = voice

    async def synthesize(self, text: str, output_path: str) -> float:
        """合成语音并返回真实时长"""
        # say 输出 AIFF，后续需要转 AAC
        aiff_path = output_path.replace(".mp3", ".aiff").replace(".m4a", ".aiff")
        if not aiff_path.endswith(".aiff"):
            aiff_path += ".aiff"

        # 调用 say 导出音频
        proc = await asyncio.create_subprocess_exec(
            "say", "-v", self.voice, "-o", aiff_path, text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"say 合成失败: {(await proc.stderr.read()).decode()}")

        # 转为 AAC (m4a)，更小且兼容 ffmpeg 混音
        m4a_path = output_path
        if not m4a_path.endswith(".m4a"):
            m4a_path = str(Path(output_path).with_suffix(".m4a"))

        proc2 = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", aiff_path,
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "1",
            m4a_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc2.wait()

        # 清理 AIFF 临时文件
        Path(aiff_path).unlink(missing_ok=True)

        # 读取真实时长
        duration = _get_audio_duration(m4a_path)
        return duration


# ─── 火山语音合成引擎 (占位，密钥到位后实现) ───

class VolcanoTTSEngine:
    """火山引擎语音合成 — 需要在 .env 配置 VOLCANO_TTS_APPID + VOLCANO_TTS_TOKEN

    TODO: 密钥到位后实现。接口保持 synthesize(text, output_path) -> duration
    """

    def __init__(self):
        self.appid = os.getenv("VOLCANO_TTS_APPID")
        self.token = os.getenv("VOLCANO_TTS_TOKEN")
        if not self.appid or not self.token:
            raise RuntimeError(
                "火山 TTS 需要 VOLCANO_TTS_APPID 和 VOLCANO_TTS_TOKEN，"
                "请在 .env 中配置，或设 TTS_ENGINE=macos 使用系统语音"
            )

    async def synthesize(self, text: str, output_path: str) -> float:
        raise NotImplementedError("火山 TTS 尚未实现，请先用 TTS_ENGINE=macos")


# ─── 引擎工厂 ───

def get_tts_engine() -> TTSEngine:
    """根据 .env 中的 TTS_ENGINE 配置返回对应引擎"""
    engine_name = os.getenv("TTS_ENGINE", "macos").lower()

    if engine_name == "macos":
        voice = os.getenv("TTS_VOICE", "Tingting")
        return MacOSSayEngine(voice=voice)
    elif engine_name == "volcano":
        return VolcanoTTSEngine()
    else:
        raise ValueError(f"未知 TTS 引擎: {engine_name}，可选: macos / volcano")


# ─── 批量合成 ───

async def synthesize_voiceover(
    storyboard: dict,
    output_dir: str,
) -> list[TTSSegment]:
    """为分镜中所有有字幕的镜头合成口播语音

    Returns:
        TTSSegment 列表 (只包含有 subtitle_text 的镜头)
    """
    engine = get_tts_engine()
    out_path = Path(output_dir) / "voiceover"
    out_path.mkdir(parents=True, exist_ok=True)

    segments: list[TTSSegment] = []

    for shot in storyboard["shots"]:
        text = shot.get("subtitle_text", "").strip()
        if not text:
            continue

        audio_file = str(out_path / f"vo_shot_{shot['shot_id']:03d}.m4a")
        print(f"  🎙️ TTS Shot {shot['shot_id']}: \"{text[:30]}...\"")

        duration = await engine.synthesize(text, audio_file)

        segments.append(TTSSegment(
            shot_id=shot["shot_id"],
            text=text,
            audio_path=audio_file,
            duration=duration,
        ))
        print(f"     ✓ {duration:.1f}s → {Path(audio_file).name}")

    return segments


# ─── 工具函数 ───

def _get_audio_duration(filepath: str) -> float:
    """获取音频文件时长"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", filepath],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())
