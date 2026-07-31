"""TTS 语音合成 — 可插拔引擎

默认使用 macOS say 命令跑通全链路；
火山语音技术密钥到位后，在 .env 中设 TTS_ENGINE=volcano 即可切换。

全链路:
  分镜 subtitle_text → 逐句合成语音 → 返回 [(音频文件, 真实时长)]
"""

from __future__ import annotations

import asyncio
import json
import struct
import subprocess
import tempfile
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol

import aiohttp

import config


_SESSION_FINISHED = 152
_TTS_RESPONSE = 352
_AUDIO_RESPONSE = 0xB
_ERROR_RESPONSE = 0xF


@dataclass(frozen=True)
class VoiceProfile:
    name: str
    speaker: str
    keywords: tuple[str, ...]


# Official catalog: https://docs.volcengine.com/docs/6561/1257544?lang=zh
_VOICE_PROFILES = (
    VoiceProfile(
        "少儿故事",
        "zh_female_shaoergushi_mars_bigtts",
        ("少儿", "儿童", "童话", "萌娃", "child", "children", "fairy tale"),
    ),
    VoiceProfile(
        "悬疑解说",
        "zh_male_changtianyi_mars_bigtts",
        (
            "悬疑", "惊悚", "恐怖", "末日", "灾难", "神秘", "压迫",
            "mysterious", "suspense", "horror", "thriller", "dark", "grim",
            "tense", "apocalypse",
        ),
    ),
    VoiceProfile(
        "阳光青年",
        "zh_male_yangguangqingnian_moon_bigtts",
        (
            "运动", "热血", "活力", "青春", "竞技", "快节奏",
            "energetic", "upbeat", "sports", "dynamic", "triumphant",
        ),
    ),
    VoiceProfile(
        "解说小明",
        "zh_male_jieshuoxiaoming_moon_bigtts",
        (
            "科普", "教程", "知识", "历史", "教育", "纪录片", "新闻",
            "explain", "tutorial", "educational", "documentary", "news",
            "informative",
        ),
    ),
    VoiceProfile(
        "知性女声",
        "zh_female_zhixingnvsheng_mars_bigtts",
        (
            "科技", "高端", "奢华", "优雅", "专业",
            "premium", "luxury", "futuristic", "technology", "elegant",
            "professional", "modern_tech",
        ),
    ),
    VoiceProfile(
        "广告解说",
        "zh_male_chunhui_mars_bigtts",
        ("广告", "品牌", "产品", "营销", "commercial", "brand", "product", "corporate"),
    ),
    VoiceProfile(
        "亲切女声",
        "zh_female_qinqienvsheng_moon_bigtts",
        (
            "温暖", "治愈", "家庭", "咖啡", "美食", "生活方式", "旅行", "浪漫",
            "warm", "healing", "family", "coffee", "food", "lifestyle", "travel",
            "romantic", "intimate", "contemplative", "serene",
        ),
    ),
)
_DEFAULT_VOICE = VoiceProfile("Vivi", "zh_female_vv_mars_bigtts", ())


@dataclass
class TTSSegment:
    """单条口播片段"""
    shot_id: int
    text: str
    audio_path: str
    duration: float   # 真实语音时长 (秒)


class TTSEngine(Protocol):
    """TTS 引擎接口 — 实现此协议即可接入"""
    voice_name: str

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
        self.voice_name = voice

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

        try:
            await _transcode_to_m4a(aiff_path, m4a_path)
        finally:
            Path(aiff_path).unlink(missing_ok=True)

        # 读取真实时长
        duration = _get_audio_duration(m4a_path)
        return duration


# ─── 火山语音合成引擎 ───

class VolcanoTTSEngine:
    """豆包语音合成模型 1.0，使用官方 WebSocket 单向流式 V3 接口。"""

    def __init__(self, voice: VoiceProfile):
        self.api_key = config.VOLCANO_TTS_API_KEY
        self.speaker = voice.speaker
        self.voice_name = voice.name
        if not self.api_key:
            raise RuntimeError("火山 TTS 需要 VOLCANO_TTS_API_KEY")

    async def synthesize(self, text: str, output_path: str) -> float:
        if not text.strip():
            raise ValueError("TTS 文本不能为空")
        if Path(output_path).suffix.lower() != ".m4a":
            raise ValueError("TTS 输出路径必须使用 .m4a 后缀")

        mp3_data = await self._request_audio(text)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f"{output.stem}_", suffix=".mp3", dir=output.parent, delete=False
            ) as temp_file:
                temp_file.write(mp3_data)
                temp_path = Path(temp_file.name)
            await _transcode_to_m4a(str(temp_path), str(output))
        finally:
            if temp_path:
                temp_path.unlink(missing_ok=True)
        return _get_audio_duration(str(output))

    async def _request_audio(self, text: str) -> bytes:
        request_id = str(uuid.uuid4())
        headers = {
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": config.VOLCANO_TTS_RESOURCE_ID,
            "X-Api-Request-Id": request_id,
        }
        request = _build_tts_request(text, self.speaker)
        timeout = aiohttp.ClientTimeout(total=120)
        audio = bytearray()
        finished = False

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(
                config.VOLCANO_TTS_ENDPOINT,
                headers=headers,
                heartbeat=30,
            ) as websocket:
                await websocket.send_bytes(request)
                async for message in websocket:
                    if message.type == aiohttp.WSMsgType.BINARY:
                        frame = _parse_tts_frame(message.data)
                        if frame.event == _TTS_RESPONSE:
                            audio.extend(frame.payload)
                        elif frame.event == _SESSION_FINISHED:
                            _validate_session_result(frame.payload)
                            finished = True
                            break
                    elif message.type == aiohttp.WSMsgType.TEXT:
                        raise RuntimeError(f"火山 TTS 返回了非预期文本响应: {message.data[:300]}")
                    elif message.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError(f"火山 TTS WebSocket 错误: {websocket.exception()}")

        if not finished:
            raise RuntimeError("火山 TTS 连接在 SessionFinished 前关闭")
        if not audio:
            raise RuntimeError("火山 TTS 未返回音频数据")
        return bytes(audio)


# ─── 引擎工厂 ───

def get_tts_engine(storyboard: dict | None = None) -> TTSEngine:
    """根据 .env 中的 TTS_ENGINE 配置返回对应引擎"""
    engine_name = config.TTS_ENGINE

    if engine_name == "macos":
        return MacOSSayEngine(voice=config.TTS_VOICE)
    elif engine_name == "volcano":
        return VolcanoTTSEngine(select_voice_for_storyboard(storyboard or {}))
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
    voiced_shots = [shot for shot in storyboard["shots"] if shot.get("subtitle_text", "").strip()]
    if not voiced_shots:
        return []

    engine = get_tts_engine(storyboard)
    print(f"  TTS 音色: {engine.voice_name}")
    out_path = Path(output_dir) / "voiceover"
    out_path.mkdir(parents=True, exist_ok=True)

    segments: list[TTSSegment] = []

    for shot in voiced_shots:
        text = shot.get("subtitle_text", "").strip()

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
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"读取 TTS 音频时长失败: {result.stderr.strip()}")
    return float(result.stdout.strip())


def select_voice_for_storyboard(storyboard: dict) -> VoiceProfile:
    """Choose one narrator for the whole video from the official 1.0 catalog."""
    context_parts = [
        storyboard.get("title", ""),
        storyboard.get("style", ""),
        storyboard.get("mood", ""),
        storyboard.get("music_style", ""),
    ]
    for shot in storyboard.get("shots", []):
        context_parts.extend((
            shot.get("mood", ""),
            shot.get("scene_description", ""),
            shot.get("subtitle_text", ""),
        ))
    context = " ".join(str(part) for part in context_parts).lower()

    best_voice = _DEFAULT_VOICE
    best_score = 0
    for voice in _VOICE_PROFILES:
        score = sum(context.count(keyword) for keyword in voice.keywords)
        if score > best_score:
            best_voice = voice
            best_score = score
    return best_voice


@dataclass(frozen=True)
class _TTSFrame:
    event: int | None
    payload: bytes


def _build_tts_request(text: str, speaker: str) -> bytes:
    payload = json.dumps(
        {
            "user": {"uid": str(uuid.uuid4())},
            "req_params": {
                "text": text,
                "speaker": speaker,
                "audio_params": {"format": "mp3", "sample_rate": 48000},
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return bytes((0x11, 0x10, 0x10, 0x00)) + struct.pack(">I", len(payload)) + payload


def _parse_tts_frame(data: bytes) -> _TTSFrame:
    if len(data) < 4:
        raise RuntimeError("火山 TTS 返回了不完整的协议头")

    header_size = (data[0] & 0x0F) * 4
    message_type = data[1] >> 4
    flags = data[1] & 0x0F
    if header_size < 4 or len(data) < header_size:
        raise RuntimeError("火山 TTS 返回了无效的协议头长度")

    offset = header_size
    if message_type == _ERROR_RESPONSE:
        error_code, offset = _read_uint32(data, offset)
        payload, _ = _read_sized_payload(data, offset)
        detail = payload.decode("utf-8", errors="replace")
        raise RuntimeError(f"火山 TTS 请求失败 ({error_code}): {detail}")

    if flags != 0x04:
        raise RuntimeError(f"火山 TTS 返回了不支持的协议帧标记: {flags}")

    event, offset = _read_uint32(data, offset)
    session_id, offset = _read_sized_payload(data, offset)
    if not session_id:
        raise RuntimeError("火山 TTS 响应缺少 session_id")
    payload, offset = _read_sized_payload(data, offset)
    if offset != len(data):
        raise RuntimeError("火山 TTS 响应包含未解析的尾部数据")
    if event == _TTS_RESPONSE and message_type != _AUDIO_RESPONSE:
        raise RuntimeError("火山 TTS 音频事件使用了错误的消息类型")
    return _TTSFrame(event=event, payload=payload)


def _read_uint32(data: bytes, offset: int) -> tuple[int, int]:
    if len(data) < offset + 4:
        raise RuntimeError("火山 TTS 返回了被截断的协议帧")
    return struct.unpack_from(">I", data, offset)[0], offset + 4


def _read_sized_payload(data: bytes, offset: int) -> tuple[bytes, int]:
    size, offset = _read_uint32(data, offset)
    end = offset + size
    if end > len(data):
        raise RuntimeError("火山 TTS 返回了被截断的 payload")
    return data[offset:end], end


def _validate_session_result(payload: bytes) -> None:
    if not payload:
        return
    try:
        result = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("火山 TTS SessionFinished 响应不是有效 JSON") from exc
    if result.get("status_code") not in (None, 20000000):
        raise RuntimeError(
            f"火山 TTS 会话失败 ({result.get('status_code')}): {result.get('message', '未知错误')}"
        )


async def _transcode_to_m4a(input_path: str, output_path: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", input_path,
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "1",
        output_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        Path(output_path).unlink(missing_ok=True)
        raise RuntimeError(f"TTS 音频转码失败: {stderr.decode(errors='replace')[-500:]}")
