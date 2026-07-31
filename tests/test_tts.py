"""Doubao Speech TTS 1.0 protocol tests."""

import json
import struct
from types import SimpleNamespace

import aiohttp
import pytest

import config
from tools import tts


def _response_frame(message_type: int, event: int, payload: bytes) -> bytes:
    session_id = b"session-1"
    return b"".join(
        (
            bytes((0x11, (message_type << 4) | 0x04, 0x10, 0x00)),
            struct.pack(">I", event),
            struct.pack(">I", len(session_id)),
            session_id,
            struct.pack(">I", len(payload)),
            payload,
        )
    )


def _error_frame(code: int, detail: str) -> bytes:
    payload = detail.encode()
    return b"".join(
        (
            bytes((0x11, 0xF0, 0x10, 0x00)),
            struct.pack(">I", code),
            struct.pack(">I", len(payload)),
            payload,
        )
    )


class FakeWebSocket:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def send_bytes(self, data):
        self.sent.append(data)

    def exception(self):
        return None


class FakeSession:
    def __init__(self, websocket, capture, **_kwargs):
        self.websocket = websocket
        self.capture = capture

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def ws_connect(self, endpoint, **kwargs):
        self.capture.update(endpoint=endpoint, **kwargs)
        return self.websocket


def test_tts_request_uses_official_v3_send_text_frame():
    request = tts._build_tts_request("你好", "voice-1.0")

    assert request[:4] == bytes((0x11, 0x10, 0x10, 0x00))
    payload_size = struct.unpack(">I", request[4:8])[0]
    payload = json.loads(request[8:])
    assert payload_size == len(request) - 8
    assert payload["req_params"] == {
        "text": "你好",
        "speaker": "voice-1.0",
        "audio_params": {"format": "mp3", "sample_rate": 48000},
    }


@pytest.mark.asyncio
async def test_volcano_tts_uses_seed_tts_1_0_and_waits_for_session_finished(monkeypatch):
    monkeypatch.setattr(config, "VOLCANO_TTS_API_KEY", "tts-key")
    voice = tts.VoiceProfile("测试音色", "voice-1.0", ())
    audio_frames = [
        _response_frame(0xB, 352, b"first"),
        _response_frame(0xB, 352, b"second"),
        _response_frame(0x9, 152, b'{"status_code":20000000,"message":"ok"}'),
    ]
    websocket = FakeWebSocket(
        [SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=data) for data in audio_frames]
    )
    capture = {}
    monkeypatch.setattr(
        tts.aiohttp,
        "ClientSession",
        lambda **kwargs: FakeSession(websocket, capture, **kwargs),
    )

    audio = await tts.VolcanoTTSEngine(voice)._request_audio("测试")

    assert audio == b"firstsecond"
    assert capture["endpoint"] == config.VOLCANO_TTS_ENDPOINT
    assert capture["headers"]["X-Api-Key"] == "tts-key"
    assert capture["headers"]["X-Api-Resource-Id"] == "seed-tts-1.0"
    assert len(websocket.sent) == 1


def test_volcano_tts_rejects_service_error_frame():
    with pytest.raises(RuntimeError, match="45000000"):
        tts._parse_tts_frame(_error_frame(45000000, "invalid speaker"))


def test_volcano_tts_requires_dedicated_credentials(monkeypatch):
    monkeypatch.setattr(config, "VOLCANO_TTS_API_KEY", None)

    with pytest.raises(RuntimeError, match="VOLCANO_TTS_API_KEY"):
        tts.VolcanoTTSEngine(tts.VoiceProfile("测试音色", "voice-1.0", ()))


def test_seed_tts_1_0_resource_is_not_environment_overridable():
    assert config.VOLCANO_TTS_RESOURCE_ID == "seed-tts-1.0"


@pytest.mark.parametrize(
    ("storyboard", "expected_name", "expected_speaker"),
    [
        (
            {"mood": "warm premium contemplative", "shots": [{"scene_description": "清晨咖啡店"}]},
            "亲切女声",
            "zh_female_qinqienvsheng_moon_bigtts",
        ),
        (
            {"style": "cinematic", "mood": "grim tense", "title": "末日求生"},
            "悬疑解说",
            "zh_male_changtianyi_mars_bigtts",
        ),
        (
            {"mood": "energetic triumphant", "title": "青春运动品牌"},
            "阳光青年",
            "zh_male_yangguangqingnian_moon_bigtts",
        ),
        (
            {"style": "futuristic", "mood": "premium", "title": "高端科技产品"},
            "知性女声",
            "zh_female_zhixingnvsheng_mars_bigtts",
        ),
        (
            {"style": "documentary", "title": "历史知识科普"},
            "解说小明",
            "zh_male_jieshuoxiaoming_moon_bigtts",
        ),
        (
            {"style": "cinematic", "mood": "balanced"},
            "Vivi",
            "zh_female_vv_mars_bigtts",
        ),
    ],
)
def test_voice_is_selected_from_storyboard_context(storyboard, expected_name, expected_speaker):
    voice = tts.select_voice_for_storyboard(storyboard)

    assert voice.name == expected_name
    assert voice.speaker == expected_speaker
