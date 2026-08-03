"""Storyboard LLM JSON response boundary tests."""

import json
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.storyboard import (
    _build_correction_prompt,
    _call_llm_for_storyboard,
    _parse_json_response,
    generate_storyboard,
)
from pipeline.causality import blocking_geometry_issues
from pipeline.models import validate_storyboard, validate_storyboard_draft


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


def test_llm_draft_projection_drops_unknown_keys_but_runtime_validation_stays_strict():
    draft = {
        "shots": [{
            "shot_id": 1,
            "duration": 5,
            "scene_description": "test scene",
            "prompt_en": "cinematic test shot",
            "interaction_geometry": {
                "effect_phase": "none",
                "reaction_zone": "an unsupported invented field",
            },
        }],
    }

    projected = validate_storyboard_draft(draft)

    assert "reaction_zone" not in projected["shots"][0]["interaction_geometry"]
    with pytest.raises(ValueError, match="reaction_zone"):
        validate_storyboard(draft)


def test_generate_storyboard_applies_planned_duration_before_validation(
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

    assert storyboard["shots"][0]["duration"] == 5


def test_generate_storyboard_normalizes_misplaced_top_level_metadata(monkeypatch):
    draft = {
        "title": "robot cleanup",
        "total_duration": 30,
        "shots": [{
            "shot_id": 1,
            "duration": 6,
            "total_duration": 30,
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

    assert storyboard["total_duration"] == 30
    assert "total_duration" not in storyboard["shots"][0]


def test_generate_storyboard_replaces_llm_duration_with_executable_plan(monkeypatch):
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

    monkeypatch.setattr(
        "pipeline.storyboard._validate_storyboard_richness",
        lambda *_args, **_kwargs: ([], False),
    )

    storyboard = generate_storyboard("制作一个30秒动作视频", target_duration=30)

    assert storyboard["shots"][0]["duration"] == 5


def test_action_density_correction_is_surgical_and_explicit():
    prompt = _build_correction_prompt(
        "用户需求: 15 秒机器人清除丧尸",
        {"shots": [{"shot_id": 1}, {"shot_id": 2}, {"shot_id": 3}]},
        ["🚨 动作重心不足: 仅 1/3 个镜头"],
    )

    assert "只修改导致告警的创意字段" in prompt
    assert "冲突中建立" in prompt
    assert "动作中收束" in prompt
    assert "至少 2 个镜头" in prompt


def test_missing_interaction_mode_routes_to_causality_correction():
    prompt = _build_correction_prompt(
        "用户需求: 抽象交互短片",
        {"shots": [{"shot_id": 1}]},
        ["🚨 Shot 1: 可见因果交互缺少 interaction_mode"],
    )

    assert "direct_contact" in prompt
    assert "directed_path" in prompt
    assert "area_effect" in prompt
    assert "indirect_effect" in prompt
    assert "准备/瞄准/充能用 setup" in prompt
    assert "aftermath 不得创建新作用" in prompt


def test_short_action_generation_receives_immutable_production_slots(monkeypatch):
    narrative_states = (
        ("the threat is approaching", "the street conflict is active"),
        ("the street conflict is active", "the immediate threat is reduced"),
        ("the immediate threat is reduced", "the route is visibly secured"),
    )

    def shot(shot_id: int, duration: int, action: str, framing: str) -> dict:
        state_before, state_after = narrative_states[shot_id - 1]
        return {
            "shot_id": shot_id,
            "duration": duration,
            "scene_id": "post_apoc_downtown",
            "scene_description": "【末日市中心】同一街道中的连续冲突",
            "prompt_en": "cinematic detail " * 40,
            "primary_action": action,
            "narrative_beat": {
                "function": ("setup", "progress", "payoff")[shot_id - 1],
                "state_before": state_before,
                "state_change": f"shot {shot_id} visibly changes the situation",
                "state_after": state_after,
            },
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
            "interaction_geometry": {
                "interaction_mode": "none",
                "effect_phase": "none",
                "outcome_scope": "none",
                "effect_motion": "none",
            },
            "mood": f"action mood {shot_id}",
            "characters": [],
            "extract_character_ref": False,
        }

    draft = {
        "title": "robot cleanup",
        "story_arc": {
            "goal": "secure the route",
            "stakes": "the route remains dangerous",
            "turning_point": "the threat enters direct conflict",
            "resolution": "the route is visibly secured",
        },
        "shots": [
            shot(1, 4, "smoke drifts across the ruined street", "wide shot"),
            shot(2, 5, "robot fires one burst at zombies", "close-up"),
            shot(3, 6, "robot scans the cleared street", "medium shot"),
        ],
    }
    calls = []

    def fake_call(client, system_prompt, user_prompt, **kwargs):
        calls.append((user_prompt, kwargs))
        return deepcopy(draft)

    monkeypatch.setattr("pipeline.storyboard.OpenAI", lambda **kwargs: object())
    monkeypatch.setattr("pipeline.storyboard._call_llm_for_storyboard", fake_call)
    monkeypatch.setattr(
        "pipeline.storyboard._validate_storyboard_richness",
        lambda *_args, **_kwargs: ([], False),
    )

    result = generate_storyboard(
        "制作一个15秒的视频，机器人末日清除丧尸",
        target_duration=15,
    )

    assert result["shots"][0]["primary_action"] == "smoke drifts across the ruined street"
    assert result["shots"][2]["primary_action"] == "robot scans the cleared street"
    assert all(
        shot["interaction_geometry"]["effect_phase"] == "active"
        for shot in result["shots"]
    )
    assert "生产计划" in calls[0][0]
    assert "effect_phase=active" in calls[0][0]
    assert len(calls) == 1


def test_contract_correction_preserves_accepted_spatial_fields(monkeypatch):
    blocking = {
        "robot": {
            "frame_position": "screen-left foreground",
            "body_orientation": "three-quarter toward screen-right",
            "facing_target": "zombies",
            "eyeline_target": "zombies",
            "travel_direction": "left-to-right",
            "action_target": "zombies",
        },
        "zombies": {
            "frame_position": "screen-right midground",
            "body_orientation": "three-quarter toward screen-left",
            "facing_target": "robot",
            "eyeline_target": "robot",
            "travel_direction": "right-to-left",
            "action_target": "robot",
        },
    }
    original = {
        "title": "robot cleanup",
        "characters": [
            {"name": "robot", "reference_mode": "identity"},
            {"name": "zombies", "reference_mode": "group"},
        ],
        "shots": [{
            "shot_id": 1,
            "duration": 6,
            "scene_id": "ruined_city_street",
            "scene_description": "Ruined city street firefight",
            "prompt_en": "cinematic action detail " * 30,
            "continuity_from_previous": "none",
            "coverage_role": "action_subject",
            "required_visible_entities": ["robot", "zombies"],
            "primary_action": "robot aims at the approaching zombies",
            "action_beats": [{
                "phase": "trigger",
                "actor": "robot",
                "action": "aims its rifle",
                "target": "zombies",
                "visible_result": "zombies remain visible downrange",
            }],
            "start_state": {
                "location": "ruined city street",
                "subject": "robot faces zombies",
                "action_phase": "aiming",
                "camera": "wide shot",
            },
            "end_state": {
                "location": "ruined city street",
                "subject": "robot keeps zombies in sight",
                "action_phase": "ready to fire",
                "camera": "wide shot",
            },
            "camera": {
                "speed": "fixed",
                "start_framing": "wide shot",
                "end_framing": "wide shot",
                "screen_positions": {
                    "robot": "screen-left foreground",
                    "zombies": "screen-right midground",
                },
                "axis_change": "establish",
            },
            "blocking": blocking,
            "characters": ["robot", "zombies"],
        }],
    }
    corrected = deepcopy(original)
    corrected["shots"][0].update({
        "coverage_role": "interaction",
        "required_visible_entities": ["robot"],
        "primary_action": "robot fires one burst at the zombies",
        "camera": {
            "speed": "fixed",
            "start_framing": "medium shot",
            "end_framing": "medium shot",
        },
    })
    corrected["shots"][0].pop("blocking")
    responses = iter((original, corrected))
    richness = iter(((['🚨 动作景别层次不足'], True), ([], False)))

    monkeypatch.setattr("pipeline.storyboard.OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(
        "pipeline.storyboard._call_llm_for_storyboard",
        lambda *_args, **_kwargs: deepcopy(next(responses)),
    )
    monkeypatch.setattr(
        "pipeline.storyboard._validate_storyboard_richness",
        lambda *_args, **_kwargs: next(richness),
    )

    storyboard = generate_storyboard(
        "制作一个30秒的视频，机器人在末日城市清除丧尸",
        target_duration=30,
    )
    shot = storyboard["shots"][0]

    assert shot["coverage_role"] == "interaction"
    assert shot["primary_action"] == "robot fires one burst at the zombies"
    assert shot["camera"]["start_framing"] == "wide shot"
    assert shot["camera"]["screen_positions"] == original["shots"][0]["camera"]["screen_positions"]
    assert shot["camera"]["axis_change"] == "establish"
    assert shot["required_visible_entities"] == ["robot", "zombies"]
    assert shot["blocking"]["robot"]["frame_position"] == (
        blocking["robot"]["frame_position"]
    )
    assert shot["blocking"]["robot"]["travel_direction"] == (
        blocking["robot"]["travel_direction"]
    )
    assert shot["blocking"]["robot"]["body_orientation"] == (
        "three-quarter toward screen-right and background, away from camera"
    )
    assert not blocking_geometry_issues(shot)


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
