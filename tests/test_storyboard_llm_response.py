"""Storyboard LLM JSON response boundary tests."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.storyboard import _call_llm_for_storyboard, _parse_json_response


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _Completions:
    def __init__(self, contents: list[str]):
        self._contents = iter(contents)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _response(next(self._contents))


def _client(*contents: str) -> tuple[SimpleNamespace, _Completions]:
    completions = _Completions(list(contents))
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
    assert "JSONDecodeError" in repair_message
    assert "猫说\"你好\"" in repair_message


def test_llm_stops_after_one_failed_json_repair():
    malformed = '{"title":"猫咖","shots":[}'
    client, completions = _client(malformed, malformed)

    with pytest.raises(ValueError, match="修复失败.*已重试 1 次"):
        _call_llm_for_storyboard(client, "输出 JSON", "制作猫咖短视频")

    assert len(completions.calls) == 2
