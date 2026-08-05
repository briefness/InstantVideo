"""Deterministic shot topology compiled before creative storyboard writing."""

from __future__ import annotations

from copy import deepcopy
from math import ceil

import config


_ALL_EFFECT_PHASES = ["none", "setup", "active", "aftermath"]


def build_production_plan(content_focus: str, target_duration: int) -> dict:
    """Allocate executable slots without deciding story-specific content."""
    duration = max(config.MIN_SHOT_DURATION, int(target_duration))
    shot_count = min(10, max(1, ceil(duration / 6)))
    durations = _allocate_durations(duration, shot_count)
    slots = [
        _build_slot(content_focus, index, shot_count, slot_duration)
        for index, slot_duration in enumerate(durations)
    ]
    return {
        "version": "production-plan-v1",
        "content_focus": content_focus,
        "target_duration": duration,
        "planned_duration": sum(durations),
        "slots": slots,
    }


def apply_production_plan(storyboard: dict, plan: dict) -> None:
    """Project immutable production decisions onto an LLM-authored storyboard."""
    storyboard["production_plan"] = deepcopy(plan)
    storyboard["content_focus"] = plan["content_focus"]
    storyboard["total_duration"] = plan["planned_duration"]
    slots = plan["slots"]
    for index, shot in enumerate(storyboard.get("shots", [])):
        if not isinstance(shot, dict) or index >= len(slots):
            continue
        slot = deepcopy(slots[index])
        shot["shot_id"] = slot["shot_id"]
        shot["duration"] = slot["duration"]
        shot["production_slot"] = slot

        camera = shot.get("camera")
        camera = dict(camera) if isinstance(camera, dict) else {}
        framing = {
            "wide": "wide shot",
            "medium": "medium shot",
            "close_detail": "close-up detail shot",
        }[slot["framing_family"]]
        camera["start_framing"] = framing
        camera["end_framing"] = framing
        shot["camera"] = camera

        narrative = shot.get("narrative_beat")
        narrative = dict(narrative) if isinstance(narrative, dict) else {}
        narrative["function"] = slot["narrative_function"]
        shot["narrative_beat"] = narrative
        shot["coverage_role"] = slot["coverage_roles"][0]

        geometry = shot.get("interaction_geometry")
        geometry = dict(geometry) if isinstance(geometry, dict) else {}
        allowed_phases = slot["allowed_effect_phases"]
        if len(allowed_phases) == 1:
            geometry["effect_phase"] = allowed_phases[0]
        elif geometry.get("effect_phase") not in allowed_phases:
            geometry["effect_phase"] = allowed_phases[0]
        if slot.get("outcome_scope"):
            geometry["outcome_scope"] = slot["outcome_scope"]
        shot["interaction_geometry"] = geometry

        reference_policy = slot["reference_policy"]
        if reference_policy == "independent":
            shot["continuity_from_previous"] = "none"
            shot["composition_change"] = "large"
        else:
            shot["continuity_from_previous"] = "intentional_cut"
            previous_family = (
                slots[index - 1]["framing_family"] if index > 0 else None
            )
            shot["composition_change"] = (
                "small"
                if slot["framing_family"] == previous_family
                else "medium"
            )


def production_plan_issues(storyboard: dict) -> list[str]:
    """Return topology drift that must stop execution before provider calls."""
    plan = storyboard.get("production_plan")
    if not isinstance(plan, dict):
        return []
    slots = plan.get("slots")
    shots = storyboard.get("shots")
    if not isinstance(slots, list) or not isinstance(shots, list):
        return ["ProductionPlan 缺少可执行的 slots 或 shots"]

    issues: list[str] = []
    if len(shots) != len(slots):
        issues.append(
            f"ProductionPlan 镜头数量为 {len(slots)}，分镜实际为 {len(shots)}"
        )
    if storyboard.get("content_focus") != plan.get("content_focus"):
        issues.append("ProductionPlan content_focus 与分镜不一致")

    for index, (shot, slot) in enumerate(zip(shots, slots)):
        shot_id = slot.get("shot_id", "?")
        if shot.get("shot_id") != shot_id:
            issues.append(f"Slot {shot_id}: shot_id 被改写")
        if shot.get("duration") != slot.get("duration"):
            issues.append(
                f"Shot {shot_id}: duration 必须保持计划值 {slot.get('duration')} 秒"
            )
        if shot.get("production_slot") != slot:
            issues.append(f"Shot {shot_id}: production_slot 与顶层计划不一致")

        narrative = shot.get("narrative_beat")
        narrative = narrative if isinstance(narrative, dict) else {}
        if narrative.get("function") != slot.get("narrative_function"):
            issues.append(f"Shot {shot_id}: narrative function 偏离计划槽位")

        geometry = shot.get("interaction_geometry")
        geometry = geometry if isinstance(geometry, dict) else {}
        phase = geometry.get("effect_phase")
        if phase not in slot.get("allowed_effect_phases", []):
            issues.append(
                f"Shot {shot_id}: effect_phase={phase} 不属于计划允许阶段 "
                f"{slot.get('allowed_effect_phases')}"
            )
        planned_scope = slot.get("outcome_scope")
        if planned_scope and geometry.get("outcome_scope") != planned_scope:
            issues.append(
                f"Shot {shot_id}: outcome_scope 必须保持计划值 {planned_scope}"
            )

        if shot.get("coverage_role") not in slot.get("coverage_roles", []):
            issues.append(
                f"Shot {shot_id}: coverage_role={shot.get('coverage_role')} "
                f"不符合计划槽位 {slot.get('coverage_roles')}"
            )
        if slot.get("requires_visible_result") and not _has_visible_result(shot):
            issues.append(f"Shot {shot_id}: 计划结果槽位缺少 visible_result")
        if not _framing_matches(shot, slot.get("framing_family")):
            issues.append(
                f"Shot {shot_id}: 镜头景别不符合计划的 {slot.get('framing_family')} 覆盖"
            )

        expected_continuity = "none" if index == 0 else "intentional_cut"
        expected_change = "large" if index == 0 else (
            "small"
            if slot.get("framing_family") == slots[index - 1].get("framing_family")
            else "medium"
        )
        if shot.get("continuity_from_previous") != expected_continuity:
            issues.append(f"Shot {shot_id}: continuity_from_previous 偏离参考计划")
        if shot.get("composition_change") != expected_change:
            issues.append(f"Shot {shot_id}: composition_change 偏离参考计划")
    return issues


def format_production_plan(plan: dict) -> str:
    """Render the immutable topology as concise LLM instructions."""
    lines = [
        f"必须严格输出 {len(plan['slots'])} 个镜头；shot_id、duration、"
        "narrative function 和下列槽位约束不可增删或改写："
    ]
    for slot in plan["slots"]:
        lines.append(
            f"- Shot {slot['shot_id']}: {slot['duration']}s; "
            f"narrative={slot['narrative_function']}; "
            f"effect_phase={'/'.join(slot['allowed_effect_phases'])}; "
            f"outcome_scope={slot.get('outcome_scope') or 'story-dependent'}; "
            f"coverage={'/'.join(slot['coverage_roles'])}; "
            f"framing={slot['framing_family']}; "
            f"visible_result={'required' if slot['requires_visible_result'] else 'when applicable'}; "
            f"reference={slot['reference_policy']}"
        )
    return "\n".join(lines)


def reference_policy(shot: dict) -> str | None:
    slot = shot.get("production_slot")
    return slot.get("reference_policy") if isinstance(slot, dict) else None


def classify_framing(value: object) -> str | None:
    """Normalize editorial shot-size language for every plan consumer."""
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    if any(marker in text for marker in (
        "extreme close", "medium close", "close up", "close shot",
        "close framing", "macro", "detail", "特写", "近景", "细节",
    )):
        return "close_detail"
    if any(marker in text for marker in (
        "medium", "mid shot", "waist", "full body", "full shot", "中景", "全身",
    )):
        return "medium"
    if any(marker in text for marker in (
        "extreme wide", "wide", "long shot", "establishing", "远景", "全景",
    )):
        return "wide"
    return None


def _allocate_durations(target_duration: int, shot_count: int) -> list[int]:
    target = min(
        shot_count * config.MAX_SHOT_DURATION,
        max(shot_count * config.MIN_SHOT_DURATION, target_duration),
    )
    base, remainder = divmod(target, shot_count)
    return [base + (index < remainder) for index in range(shot_count)]


def _build_slot(focus: str, index: int, count: int, duration: int) -> dict:
    action = focus == "action"
    allowed_phases = list(_ALL_EFFECT_PHASES)
    requires_visible_result = False
    outcome_scope: str | None = None
    if action:
        phase = _action_phase(index, count)
        allowed_phases = [phase]
        outcome_scope = "none" if phase == "setup" else "single"
        requires_visible_result = phase != "setup"

    framing_families = _framing_families(count)
    reference_policy = (
        "independent"
        if index == 0
        else "state_if_same_scene"
        if framing_families[index] == framing_families[index - 1]
        else "state_and_identity"
    )
    # A planned intentional cut is the only place where a long tail chain may
    # reset. It remains a no-op when no accepted canonical identity exists.
    if (
        index > 0
        and index < count - 1
        and index % (config.MAX_REFERENCE_CHAIN_DEPTH + 1) == 0
    ):
        reference_policy = "identity_only"

    return {
        "shot_id": index + 1,
        "duration": duration,
        "narrative_function": _narrative_function(index, count),
        "allowed_effect_phases": allowed_phases,
        "outcome_scope": outcome_scope,
        "requires_visible_result": requires_visible_result,
        "coverage_roles": _coverage_roles(focus, index, count),
        "framing_family": framing_families[index],
        "reference_policy": reference_policy,
    }


def _narrative_function(index: int, count: int) -> str:
    if count == 1 or index == count - 1:
        return "payoff"
    if index == 0:
        return "setup"
    if index == count - 2:
        return "turn"
    return "progress"


def _coverage_roles(focus: str, index: int, count: int) -> list[str]:
    if focus == "action":
        phase = _action_phase(index, count)
        if phase == "setup":
            return ["establish"]
        if phase == "aftermath":
            return ["aftermath"]
        if index == 0:
            return ["interaction"]
        if count >= 5 and index == count // 2:
            return ["target_reaction"]
        return ["action_subject" if index % 2 else "interaction"]
    if focus == "product":
        if index == 0:
            return ["establish", "action_subject"]
        if index == count - 1:
            return ["aftermath", "action_subject"]
        return ["insert", "action_subject", "interaction"]
    if index == 0:
        return ["establish", "action_subject"]
    if index == count - 1:
        return ["aftermath", "action_subject", "target_reaction"]
    return ["action_subject", "interaction", "insert", "target_reaction"]


def _action_phase(index: int, count: int) -> str:
    """Give action requests a deterministic cinematic progression by slot count."""
    if count <= 2:
        return "active"
    if count == 3:
        return "aftermath" if index == count - 1 else "active"
    if index == 0:
        return "setup"
    if index == count - 1:
        return "aftermath"
    return "active"


def _framing_families(count: int) -> list[str]:
    if count == 1:
        return ["medium"]
    if count == 2:
        return ["wide", "medium"]
    cycle = ("wide", "medium", "close_detail")
    return [cycle[index % len(cycle)] for index in range(count)]


def _has_visible_result(shot: dict) -> bool:
    return any(
        isinstance(beat, dict) and str(beat.get("visible_result", "")).strip()
        for beat in shot.get("action_beats", [])
    )


def _framing_matches(shot: dict, family: str | None) -> bool:
    camera = shot.get("camera")
    camera = camera if isinstance(camera, dict) else {}
    value = next(
        (
            camera.get(key)
            for key in ("start_framing", "end_framing", "composition")
            if str(camera.get(key, "")).strip()
        ),
        "",
    )
    return classify_framing(value) == family
