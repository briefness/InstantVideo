"""Hard generation gate for defects that would waste paid video calls."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import config
from pipeline.camera_contract import camera_contract_issues
from pipeline.causality import (
    blocking_geometry_issues,
    causal_storyboard_issues,
    causality_readiness_issues,
    compile_action_contract,
    interaction_geometry,
    interaction_mode,
)
from pipeline.narrative import narrative_readiness_issues
from pipeline.participants import (
    structured_entity_reference_issues,
    visible_character_names,
)
from pipeline.production_plan import production_plan_issues
from pipeline.provider_capabilities import capability_contract_issues
from pipeline.visual_invariants import compile_visual_invariants


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
    theme_elements = storyboard.get("theme_elements", [])
    shots = storyboard.get("shots", [])
    for shot in shots:
        shot_id = shot.get("shot_id", "?")
        issues.extend(camera_contract_issues(shot))
        prompt = str(shot.get("prompt_en", "")).strip()
        duration = shot.get("duration")
        if not prompt:
            issues.append(f"Shot {shot_id}: prompt_en 为空")
        if isinstance(duration, bool) or not isinstance(duration, int) or not 4 <= duration <= 15:
            issues.append(f"Shot {shot_id}: duration 必须是 4-15 秒整数")

        issues.extend(structured_entity_reference_issues(
            shot,
            character_names,
            theme_elements,
        ))

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
                    issues.append(
                        f"Shot {shot_id}: 角色调度不完整: 角色 {name} 的动作 blocking 不完整"
                    )
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


def provider_submission_issues(
    prompt: str,
    shot: dict,
    storyboard: dict,
    *,
    prompt_profile: str,
    model: str,
    resolution: str,
    duration: int,
) -> list[str]:
    """Validate the final paid request after compilation and before reservation."""
    shot_id = shot.get("shot_id", "?")
    issues: list[str] = []
    if shot.get('result_admission') == 'review-required-v1':
        issues.append(f'Shot {shot_id}: 结果候选尚未完成必要性审查，不能提交视频任务')
    issues.extend(
        f"Provider capability contract 不完整: {issue}"
        for issue in capability_contract_issues()
    )
    normalized_prompt = " ".join(str(prompt or "").split())
    if not normalized_prompt:
        issues.append(f"Shot {shot_id}: 最终 provider prompt 为空")

    if prompt_profile not in {"normal", "policy_safe"}:
        issues.append(f"Shot {shot_id}: 未知 prompt profile: {prompt_profile}")

    aspect_ratio = str(storyboard.get("aspect_ratio", config.DEFAULT_RATIO))
    if aspect_ratio not in config.SUPPORTED_ASPECT_RATIOS:
        issues.append(f"Shot {shot_id}: provider 不支持画幅 {aspect_ratio}")
    if resolution not in config.SUPPORTED_RESOLUTIONS:
        issues.append(f"Shot {shot_id}: provider 不支持分辨率 {resolution}")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, int)
        or not config.MIN_SHOT_DURATION <= duration <= config.MAX_SHOT_DURATION
    ):
        issues.append(
            f"Shot {shot_id}: provider 时长必须是 "
            f"{config.MIN_SHOT_DURATION}-{config.MAX_SHOT_DURATION} 秒整数"
        )

    requested_resolution = str(
        storyboard.get("resolution", config.DEFAULT_RESOLUTION)
    )
    chain = config.GENERATION_CHAINS.get(requested_resolution)
    matching_profiles = [
        profile
        for profile in chain or []
        if profile.get("model") == model
        and profile.get("resolution") == resolution
    ]
    if not matching_profiles:
        issues.append(
            f"Shot {shot_id}: model={model}、resolution={resolution} "
            f"不属于请求分辨率 {requested_resolution} 的 generation chain"
        )
    elif isinstance(duration, int) and not isinstance(duration, bool):
        max_duration = min(
            int(profile.get("max_duration", 0)) for profile in matching_profiles
        )
        if duration > max_duration:
            issues.append(
                f"Shot {shot_id}: duration={duration} 超出当前 generation profile "
                f"上限 {max_duration}"
            )

    phase = str(interaction_geometry(shot).get("effect_phase", "")).strip()
    if normalized_prompt and prompt_profile == "normal" and phase in {
        "none", "setup", "active", "aftermath",
    }:
        contract = compile_action_contract(shot)
        missing_parts = [
            part
            for part in contract.prompt_parts
            if " ".join(str(part).split()) not in normalized_prompt
        ]
        if missing_parts:
            issues.append(
                f"Shot {shot_id}: 最终 provider prompt 丢失 "
                f"{len(missing_parts)} 项 Action Contract 投影"
            )
    if normalized_prompt:
        invariants = compile_visual_invariants(shot)
        missing_locks = [
            part
            for part in invariants.prompt_parts
            if " ".join(part.split()) not in normalized_prompt
        ]
        if missing_locks:
            issues.append(
                f"Shot {shot_id}: 最终 provider prompt 丢失 "
                f"{len(missing_locks)} 项 Positive Locks 投影"
            )
    return issues


def ensure_provider_submission_ready(
    prompt: str,
    shot: dict,
    storyboard: dict,
    *,
    prompt_profile: str,
    model: str,
    resolution: str,
    duration: int,
) -> None:
    """Fail closed before a new provider task can consume paid authorization."""
    issues = provider_submission_issues(
        prompt,
        shot,
        storyboard,
        prompt_profile=prompt_profile,
        model=model,
        resolution=resolution,
        duration=duration,
    )
    if issues:
        raise GenerationReadinessError(
            "Provider 提交预检失败:\n- " + "\n- ".join(issues)
        )


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
    from pipeline.action_timing import sequence_events, sequence_required, event_shot
    events = sequence_events(shot)
    if events:
        # 复用同框、路径、景别等规则；每个事件自行声明作用者与反应目标。
        return [issue for event in events for issue in coverage_readiness_issues(
            {**shot, **event_shot(event), "production_slot": None,
             "shot_id": f"{shot_id}/event{event['event_id']}"},
            require_causality_contract=require_causality_contract)]
    if len(active_actors) > 1 and not sequence_required(shot):
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
