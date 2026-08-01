"""Storyboard LLM JSON response boundary tests."""

import json
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.storyboard import (
    _action_structure_guidance,
    _build_correction_prompt,
    _call_llm_for_storyboard,
    _parse_json_response,
    generate_storyboard,
)


def _response(
    content: str,
    *,
    finish_reason: str = "stop",
    completion_tokens: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )],
        usage=(
            SimpleNamespace(completion_tokens=completion_tokens)
            if completion_tokens is not None
            else None
        ),
    )


class _Completions:
    def __init__(self, responses: list[SimpleNamespace]):
        self._responses = iter(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self._responses)


def _client(*contents: str | SimpleNamespace) -> tuple[SimpleNamespace, _Completions]:
    completions = _Completions([
        content if isinstance(content, SimpleNamespace) else _response(content)
        for content in contents
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_fenced_malformed_json_is_not_silently_rewritten():
    malformed = '```json\n{"title":"猫咖","shots":[{"scene_description":"猫说"你好""}]}\n```'

    with pytest.raises(json.JSONDecodeError):
        _parse_json_response(malformed)


def test_llm_retries_malformed_json_once_with_structured_output():
    malformed = '```json\n{"title":"猫咖","shots":[{"scene_description":"猫说"你好""}]}\n```'
    repaired = '{"title":"猫咖","shots":[]}'
    client, completions = _client(malformed, repaired)

    result = _call_llm_for_storyboard(client, "输出 JSON", "制作猫咖短视频")

    assert result == {"title": "猫咖", "shots": []}
    assert len(completions.calls) == 2
    assert all(
        call["response_format"] == {"type": "json_object"}
        for call in completions.calls
    )
    repair_message = completions.calls[1]["messages"][-1]["content"]
    repair_system = completions.calls[1]["messages"][0]["content"]
    assert "JSONDecodeError" in repair_message
    assert "猫说\"你好\"" in repair_message
    assert "只修复 JSON 语法" in repair_system
    assert repair_system != "输出 JSON"


def test_llm_stops_after_one_failed_json_repair():
    malformed = '{"title":"猫咖","shots":[}'
    client, completions = _client(malformed, malformed)

    with pytest.raises(ValueError, match="修复失败.*已重试 1 次"):
        _call_llm_for_storyboard(client, "输出 JSON", "制作猫咖短视频")

    assert len(completions.calls) == 2


def test_llm_does_not_try_json_repair_when_response_is_truncated():
    truncated = _response(
        '{"title":"robot cleanup","shots":[{"shot_id":1',
        finish_reason="length",
        completion_tokens=4096,
    )
    client, completions = _client(truncated)

    with pytest.raises(ValueError, match="输出被截断.*4096 tokens"):
        _call_llm_for_storyboard(client, "输出 JSON", "制作30秒动作视频")

    assert len(completions.calls) == 1


def test_storyboard_calls_set_an_explicit_completion_budget():
    malformed = '{"title":"猫咖","shots":[}'
    repaired = '{"title":"猫咖","shots":[]}'
    client, completions = _client(malformed, repaired)

    _call_llm_for_storyboard(client, "输出 JSON", "制作30秒短视频")

    assert len(completions.calls) == 2
    assert all(call["max_completion_tokens"] == 16384 for call in completions.calls)


def test_storyboard_call_accepts_lower_temperature_for_contract_correction():
    client, completions = _client('{"title":"fight","shots":[]}')

    _call_llm_for_storyboard(
        client,
        "输出 JSON",
        "修正动作槽位",
        temperature=0.2,
    )

    assert completions.calls[0]["temperature"] == 0.2


def test_fifteen_second_action_guidance_allocates_conflict_to_edge_shots():
    guidance = _action_structure_guidance("action", 15)

    assert "3 个镜头" in guidance
    assert "冲突中建立" in guidance
    assert "动作中收束" in guidance
    assert "至少 2 个镜头" in guidance


def test_generate_storyboard_normalizes_positive_duration_before_validation(
    monkeypatch,
):
    draft = {
        "title": "robot cleanup",
        "total_duration": 30,
        "shots": [{
            "shot_id": 1,
            "duration": 3,
            "scene_description": "ruined downtown street",
            "prompt_en": "combat robot advances through a ruined downtown street",
        }],
    }
    monkeypatch.setattr("pipeline.storyboard.OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(
        "pipeline.storyboard._call_llm_for_storyboard",
        lambda *_args, **_kwargs: deepcopy(draft),
    )
    monkeypatch.setattr(
        "pipeline.storyboard._validate_storyboard_richness",
        lambda *_args, **_kwargs: ([], False),
    )

    storyboard = generate_storyboard(
        "制作一个30秒的视频，机器人在末日城市清除丧尸",
        target_duration=30,
    )

    assert storyboard["shots"][0]["duration"] == 4


def test_generate_storyboard_does_not_mask_non_positive_duration(monkeypatch):
    draft = {
        "title": "invalid duration",
        "shots": [{
            "shot_id": 1,
            "duration": 0,
            "scene_description": "ruined downtown street",
            "prompt_en": "combat robot advances through a ruined downtown street",
        }],
    }
    monkeypatch.setattr("pipeline.storyboard.OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(
        "pipeline.storyboard._call_llm_for_storyboard",
        lambda *_args, **_kwargs: deepcopy(draft),
    )

    with pytest.raises(ValueError, match="duration"):
        generate_storyboard("制作一个30秒动作视频", target_duration=30)


def test_action_density_correction_is_surgical_and_explicit():
    prompt = _build_correction_prompt(
        "用户需求: 15 秒机器人清除丧尸",
        {"shots": [{"shot_id": 1}, {"shot_id": 2}, {"shot_id": 3}]},
        ["🚨 动作重心不足: 仅 1/3 个镜头"],
    )

    assert "只修改导致告警的字段" in prompt
    assert "冲突中建立" in prompt
    assert "动作中收束" in prompt
    assert "至少 2 个镜头" in prompt


def test_short_action_generation_recovers_with_one_surgical_correction(monkeypatch):
    def shot(shot_id: int, duration: int, action: str, framing: str) -> dict:
        return {
            "shot_id": shot_id,
            "duration": duration,
            "scene_id": "post_apoc_downtown",
            "scene_description": "【末日市中心】同一街道中的连续冲突",
            "prompt_en": "cinematic detail " * 40,
            "primary_action": action,
            "start_state": {
                "location": "post apocalyptic downtown street",
                "subject": "subjects hold their starting positions",
                "action_phase": "start",
                "camera": framing,
            },
            "end_state": {
                "location": "post apocalyptic downtown street",
                "subject": "the visible action reaches its result",
                "action_phase": "end",
                "camera": framing,
            },
            "camera": {
                "speed": "slow",
                "start_framing": framing,
                "end_framing": framing,
            },
            "mood": f"action mood {shot_id}",
            "characters": [],
            "extract_character_ref": False,
        }

    first_draft = {
        "title": "robot cleanup",
        "shots": [
            shot(1, 4, "smoke drifts across the ruined street", "wide shot"),
            shot(2, 5, "robot fires one burst at zombies", "close-up"),
            shot(3, 6, "robot scans the cleared street", "medium shot"),
        ],
    }
    corrected = deepcopy(first_draft)
    corrected["shots"][0]["primary_action"] = "zombies charge at the robot"
    responses = iter((first_draft, corrected))
    calls = []

    def fake_call(client, system_prompt, user_prompt, **kwargs):
        calls.append((user_prompt, kwargs))
        return deepcopy(next(responses))

    monkeypatch.setattr("pipeline.storyboard.OpenAI", lambda **kwargs: object())
    monkeypatch.setattr("pipeline.storyboard._call_llm_for_storyboard", fake_call)

    result = generate_storyboard(
        "制作一个15秒的视频，机器人末日清除丧尸",
        target_duration=15,
    )

    assert result["shots"][0]["primary_action"] == "zombies charge at the robot"
    assert result["shots"][2]["primary_action"] == "robot scans the cleared street"
    assert "动作槽位契约" in calls[0][0]
    assert "冲突中建立" in calls[1][0]
    assert calls[1][1]["temperature"] == 0.2


def test_storyboard_stops_after_one_failed_contract_correction(monkeypatch):
    invalid = {
        "title": "broken",
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "scene_description": "test scene",
            "prompt_en": "word " * 80,
            "characters": [],
        }],
    }
    calls = []

    def fake_call(*args, **kwargs):
        calls.append(1)
        return json.loads(json.dumps(invalid))

    monkeypatch.setattr("pipeline.storyboard.OpenAI", lambda **kwargs: object())
    monkeypatch.setattr("pipeline.storyboard._call_llm_for_storyboard", fake_call)

    with pytest.raises(ValueError, match="自动修正 1 次后仍未通过"):
        generate_storyboard("制作一个测试视频")

    assert len(calls) == 2
