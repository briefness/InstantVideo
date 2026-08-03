"""Hard generation gate for defects that would waste paid video calls."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from pipeline.causality import (
    blocking_geometry_issues,
    causal_storyboard_issues,
    causality_readiness_issues,
    interaction_mode,
)
from pipeline.narrative import narrative_readiness_issues
from pipeline.participants import visible_character_names
from pipeline.production_plan import production_plan_issues


class GenerationReadinessError(ValueError):
    """The production plan is incomplete or references unavailable assets."""


def storyboard_readiness_issues(storyboard: dict) -> list[str]:
    """Return deterministic plan defects without making provider calls."""
    issues: list[str] = production_plan_issues(storyboard)
    issues.extend(narrative_readiness_issues(storyboard))
    story_arc = storyboard.get("story_arc")
    issues.extend(causal_storyboard_issues(
        storyboard,
        required=(
            isinstance(story_arc, dict)
            and any(str(value).strip() for value in story_arc.values())
        ),
    ))
    characters = {
        character.get("name"): character
        for character in storyboard.get("characters", [])
        if character.get("name")
    }
    character_names = list(characters)
    shots = storyboard.get("shots", [])
    for shot in shots:
        shot_id = shot.get("shot_id", "?")
        prompt = str(shot.get("prompt_en", "")).strip()
        duration = shot.get("duration")
        if not prompt:
            issues.append(f"Shot {shot_id}: prompt_en 为空")
        if isinstance(duration, bool) or not isinstance(duration, int) or not 4 <= duration <= 15:
            issues.append(f"Shot {shot_id}: duration 必须是 4-15 秒整数")

        declared_characters = shot.get("characters", [])
        visible_characters = visible_character_names(shot, character_names)
        hidden_characters = [
            name for name in visible_characters if name not in declared_characters
        ]
        if hidden_characters:
            issues.append(
                f"Shot {shot_id}: characters 未包含实际可见角色 "
                f"{', '.join(hidden_characters)}"
            )

        unknown = sorted(set(declared_characters) - set(characters))
        if unknown and characters:
            issues.append(f"Shot {shot_id}: 未定义角色 {', '.join(unknown)}")

        if shot.get("extract_character_ref"):
            identity_characters = [
                name
                for name in visible_characters
                if characters.get(name, {}).get("reference_mode", "identity") == "identity"
            ]
            if len(visible_characters) != 1 or len(identity_characters) != 1:
                issues.append(
                    f"Shot {shot_id}: 角色身份参考只能从单一 identity 角色镜头提取"
                )

        if _is_multi_character_action(shot):
            blocking = shot.get("blocking", {})
            required = (
                "frame_position",
                "body_orientation",
                "facing_target",
                "eyeline_target",
                "action_target",
            )
            for name in shot.get("characters", []):
                intent = blocking.get(name, {}) if isinstance(blocking, dict) else {}
                if not isinstance(intent, dict) or any(
                    not str(intent.get(field, "")).strip() for field in required
                ):
                    issues.append(f"Shot {shot_id}: 角色 {name} 的动作 blocking 不完整")
            if not shot.get("action_beats"):
                issues.append(f"Shot {shot_id}: 多角色动作镜头缺少 action_beats")

        issues.extend(blocking_geometry_issues(shot))

        issues.extend(coverage_readiness_issues(shot))

    return issues


def shot_readiness_issues(
    shot: dict,
    *,
    previous_frame: str | None,
    previous_shot: dict | None = None,
    character_refs: Mapping[str, str],
) -> list[str]:
    """Validate runtime dependencies immediately before provider submission."""
    issues: list[str] = []
    shot_id = shot.get("shot_id", "?")
    continuity = shot.get("continuity_from_previous")
    from pipeline.storyboard import _should_use_previous_tail_reference

    needs_tail = continuity == "seamless" or _should_use_previous_tail_reference(
        shot,
        previous_shot,
        has_identity_reference=any(
            bool(character_refs.get(name)) for name in shot.get("characters", [])
        ),
    )
    if needs_tail and not previous_frame:
        issues.append(f"Shot {shot_id}: 连续性镜头缺少已接受的上一镜尾帧")

    for name in shot.get("characters", []):
        ref = character_refs.get(name)
        if ref and not _reference_exists(ref):
            issues.append(f"Shot {shot_id}: 角色 {name} 的身份参考不存在: {ref}")
    return issues


def ensure_storyboard_ready(storyboard: dict) -> None:
    issues = storyboard_readiness_issues(storyboard)
    if issues:
        raise GenerationReadinessError("生成就绪校验失败:\n- " + "\n- ".join(issues))


def ensure_shot_ready(
    shot: dict,
    *,
    previous_frame: str | None,
    previous_shot: dict | None = None,
    character_refs: Mapping[str, str],
) -> None:
    issues = shot_readiness_issues(
        shot,
        previous_frame=previous_frame,
        previous_shot=previous_shot,
        character_refs=character_refs,
    )
    if issues:
        raise GenerationReadinessError("生成就绪校验失败:\n- " + "\n- ".join(issues))


def _reference_exists(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:")) or Path(value).is_file()


def _is_multi_character_action(shot: dict) -> bool:
    if len(shot.get("characters", [])) < 2:
        return False
    from pipeline.storyboard import _advances_action_conflict

    return _advances_action_conflict(str(shot.get("primary_action", "")))


def coverage_readiness_issues(
    shot: dict,
    *,
    require_causality_contract: bool = False,
) -> list[str]:
    """Return only coverage and interaction defects for one shot."""
    shot_id = shot.get("shot_id", "?")
    issues = causality_readiness_issues(
        shot,
        require_for_visible_interaction=require_causality_contract,
    )
    beats = [
        beat for beat in shot.get("action_beats", [])
        if isinstance(beat, dict) and str(beat.get("actor", "")).strip()
    ]
    active_actors = list(dict.fromkeys(str(beat["actor"]).strip() for beat in beats))
    if len(active_actors) > 1:
        issues.append(
            f"Shot {shot_id}: 多个主动动作执行者 ({', '.join(active_actors)})；"
            "每镜只保留一个主动作，目标反应写入 visible_result"
        )

    geometry = shot.get("interaction_geometry", {})
    geometry = geometry if isinstance(geometry, dict) else {}
    effect_phase = str(geometry.get("effect_phase", "")).strip()
    visible_interaction = None
    if effect_phase == "active" or effect_phase in {"", "unspecified"}:
        visible_interaction = next(
            (
                beat for beat in beats
                if str(beat.get("target", "")).strip()
                and str(beat.get("visible_result", "")).strip()
            ),
            {} if effect_phase == "active" else None,
        )
    if not geometry.get("must_share_frame") and visible_interaction is None:
        return issues

    actor = str(
        geometry.get("actor")
        or (visible_interaction or {}).get("actor", "")
    ).strip()
    target = str(
        geometry.get("target")
        or (visible_interaction or {}).get("target", "")
    ).strip()
    required_visible = set(shot.get("required_visible_entities", []))
    mode = interaction_mode(shot)
    target_only_visibility = (
        mode in {"area_effect", "indirect_effect"}
        and not geometry.get("must_share_frame")
    )
    if target_only_visibility and not target:
        issues.append(f"Shot {shot_id}: {mode} 交互缺少 target")
    elif target_only_visibility and target not in required_visible:
        issues.append(
            f"Shot {shot_id}: {mode} 的反应 target 必须在 required_visible_entities"
        )
    elif not target_only_visibility and (not actor or not target):
        issues.append(f"Shot {shot_id}: 交互镜头缺少 actor 或 target")
    elif not target_only_visibility and (
        actor not in required_visible or target not in required_visible
    ):
        issues.append(
            f"Shot {shot_id}: 同框交互的 actor/target 必须都在 required_visible_entities"
        )

    camera = shot.get("camera", {})
    camera = camera if isinstance(camera, dict) else {}
    framing = " ".join(str(camera.get(key, "")).lower() for key in (
        "start_framing", "end_framing", "composition",
    ))
    if _is_extreme_closeup(framing) and len(required_visible) > 1:
        issues.append(
            f"Shot {shot_id}: 极近景无法清晰承载多个必需同框主体；"
            "改为中景交互镜头，或拆成武器特写与目标反应"
        )

    target_position = str(
        geometry.get("target_screen_position")
        or camera.get("screen_positions", {}).get(target, "")
    ).lower()
    has_visible_result = any(
        str(beat.get("visible_result", "")).strip() for beat in beats
    )
    if (
        has_visible_result
        and any(marker in target_position for marker in ("distant", "far", "远处", "远景"))
        and (_is_extreme_closeup(framing) or "shallow depth" in framing)
    ):
        issues.append(
            f"Shot {shot_id}: 远处目标与近景浅焦无法同时清晰展示命中结果"
        )
    return issues


def _is_extreme_closeup(value: str) -> bool:
    return any(marker in value for marker in (
        "extreme close-up", "extreme close up", "macro", "微距", "极近景",
    ))
