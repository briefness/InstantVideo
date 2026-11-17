"""Deterministic shot topology compiled before creative storyboard writing."""

from __future__ import annotations

from copy import deepcopy
from math import ceil

import config


_ALL_EFFECT_PHASES = ["none", "setup", "active", "aftermath"]
CANONICAL_FRAMING = {'wide': 'wide shot', 'medium': 'medium shot', 'close_detail': 'close-up detail shot'}


def build_production_plan(content_focus: str, target_duration: int, *, intent: dict | None = None,
                          contract_version: str = 'production-plan-v8-request-shape',
                          duration_allocation: list[int] | None = None,
                          duration_limits: dict | None = None) -> dict:
    """Allocate executable slots without deciding story-specific content."""
    if intent is not None:
        from pipeline.models import DirectorIntentSpec
        intent = DirectorIntentSpec.model_validate(intent).model_dump(exclude_none=True)
    if duration_limits is not None and (intent is None or contract_version != 'production-plan-v8-request-shape'):
        raise ValueError('冻结执行容量仅用于具备原始意图的新时长计划')
    if contract_version == 'production-plan-v8-request-shape':
        from pipeline.production_shape import allocate_requested_shape
        duration = target_duration
        durations = allocate_requested_shape(duration, intent, duration_allocation,
                                             duration_limits=duration_limits)
        shot_count = len(durations)
    else:
        if duration_allocation is not None:
            raise ValueError('历史计划不能套用新的创作时长分配')
        duration = max(config.MIN_SHOT_DURATION, int(target_duration))
        shot_count = min(10, max(1, ceil(duration / 6)))
        durations = _allocate_durations(duration, shot_count)
    slots = [
        _build_slot(content_focus, index, shot_count, slot_duration)
        for index, slot_duration in enumerate(durations)
    ]
    if intent is not None:
        if contract_version not in {'production-plan-v3-intent-timing', 'production-plan-v4-events', 'production-plan-v5-payoff-events', 'production-plan-v6-performance', 'production-plan-v7-film-rhythm', 'production-plan-v8-request-shape'}:
            raise ValueError('不支持的意图计划版本')
        if intent['content_focus'] != content_focus:
            raise ValueError('意图与生产计划重心不一致')
        if contract_version in {'production-plan-v6-performance', 'production-plan-v7-film-rhythm', 'production-plan-v8-request-shape'}:
            for slot in slots:
                slot['choreography'] = ('exact' if intent['motion_control'] == 'reference_motion'
                                        else intent.get('choreography', 'exact'))
                slot['creative_brief'] = intent['source_request']
                if intent.get('authoring_protocol') == 'scene-units-v1':
                    if slot['choreography'] != 'flexible':
                        raise ValueError('场景动作协议仅用于自由创作，不能替换精确或参考复现合同')
                    slot['authoring_protocol'] = intent['authoring_protocol']
                    # 题材不决定物理事件：准备镜可以发生作用，收尾也可以只有表演。
                    # 明确的密集交互任务在下方继续冻结其预算。
                    slot.update(allowed_effect_phases=list(_ALL_EFFECT_PHASES),
                                outcome_scope=None, requires_visible_result=False)
                    slot['coverage_roles'] = list(dict.fromkeys([
                        *slot['coverage_roles'], 'action_subject', 'interaction']))
                if intent.get('motion_timing'):
                    slot['motion_timing'] = deepcopy(intent['motion_timing'])
        if intent.get('creative_direction'):
            for slot in slots:
                slot['creative_direction'] = deepcopy(intent['creative_direction'])
        if intent.get('ending_requirement'):
            slots[-1]['required_outcome'] = intent['ending_requirement']
        if intent['pacing'] == 'dense':
            for index, slot in enumerate(slots):
                slot.update(allowed_effect_phases=['active'], outcome_scope='single',
                            requires_visible_result=True, coverage_roles=['interaction'],
                            min_interaction_seconds=round(slot['duration'] * .7, 2),
                            result_hold_seconds=.5 if index == len(slots) - 1 else 0)
                # 密集交互末镜必须看清双方和结果，不能强制微距细节覆盖。
                slot['framing_family'] = 'wide' if index % 2 == 0 else 'medium'
                if contract_version in {'production-plan-v4-events', 'production-plan-v5-payoff-events', 'production-plan-v6-performance', 'production-plan-v7-film-rhythm', 'production-plan-v8-request-shape'}:
                    slot['min_interaction_events'] = max(2, min(6, slot['duration'] // 2))
            if contract_version in {'production-plan-v5-payoff-events', 'production-plan-v6-performance', 'production-plan-v7-film-rhythm', 'production-plan-v8-request-shape'} and intent.get('ending_requirement') and len(slots) > 1:
                # 决胜作用包含完整受击反应；把其余名额放回交锋段，整片配额不减少。
                extra = slots[-1]['min_interaction_events'] - 1
                slots[-1]['min_interaction_events'] = 1
                recipients = list(range(1, len(slots) - 1)) or [0]
                for index in range(extra):
                    slots[recipients[index % len(recipients)]]['min_interaction_events'] += 1
    plan = {
        "version": contract_version if intent else "production-plan-v2-short-payoff",
        "content_focus": content_focus,
        "target_duration": duration,
        "planned_duration": sum(durations),
        "slots": slots,
    }
    if intent:
        plan['director_intent'] = deepcopy(intent)
        if (contract_version in {'production-plan-v7-film-rhythm', 'production-plan-v8-request-shape'} and intent['pacing'] == 'dense'
                and all(s.get('choreography') == 'flexible' for s in slots)):
            # Preserve the film's existing density requirement, but let authored tasks
            # allocate it across shots. Exact/reference reproduction retains local windows.
            plan['min_interaction_seconds'] = round(sum(s['min_interaction_seconds'] for s in slots), 2)
            for slot in slots:
                slot['min_interaction_seconds'] = 0.
    if duration_allocation is not None:
        plan['duration_allocation'] = list(duration_allocation)
    if duration_limits is not None:
        plan['duration_limits'] = deepcopy(duration_limits)
    return plan


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
        framing = CANONICAL_FRAMING[slot['framing_family']]
        camera["start_framing"] = framing
        if (classify_framing(camera.get("end_framing")) is None
                and shot.get('generation_direction') != 'authored-staging-v1'):
            camera["end_framing"] = framing
        shot["camera"] = camera

        narrative = shot.get("narrative_beat")
        narrative = dict(narrative) if isinstance(narrative, dict) else {}
        narrative["function"] = slot["narrative_function"]
        shot["narrative_beat"] = narrative
        if shot.get("coverage_role") not in slot["coverage_roles"]:
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
    if any(s.get('completion_policy') in ('source-goals-v1', 'goal-allocation-v1') for s in storyboard.get('shots', [])):
        from pipeline.completion_intent import validate_goal_coverage
        try:
            validate_goal_coverage(storyboard)
        except ValueError as exc:
            return [str(exc)]
    if plan.get('version') in {'production-plan-v3-intent-timing', 'production-plan-v4-events', 'production-plan-v5-payoff-events', 'production-plan-v6-performance', 'production-plan-v7-film-rhythm', 'production-plan-v8-request-shape'}:
        try:
            from pipeline.models import ProductionPlanSpec
            intent = plan['director_intent']
            expected = build_production_plan(plan['content_focus'], plan['target_duration'], intent=intent,
                contract_version=plan['version'], duration_allocation=plan.get('duration_allocation'),
                duration_limits=plan.get('duration_limits'))
            if ProductionPlanSpec.model_validate(expected) != ProductionPlanSpec.model_validate(plan):
                return ['ProductionPlan 时间预算或槽位偏离冻结意图，不能降低动作要求']
        except (KeyError, TypeError, ValueError):
            return ['ProductionPlan 缺少有效的冻结意图，不能降级为关键词规划']
    slots = plan.get("slots")
    shots = storyboard.get("shots")
    if not isinstance(slots, list) or not isinstance(shots, list):
        return ["ProductionPlan 缺少可执行的 slots 或 shots"]

    from pipeline.film_rhythm import planned_film_timing_issues
    issues: list[str] = planned_film_timing_issues(storyboard)
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
    direction = (plan.get('director_intent') or {}).get('creative_direction')
    if direction:
        import json
        from pipeline.creative_direction import creative_authoring_context, AUTHORING_DIRECTION
        lines.append(AUTHORING_DIRECTION + json.dumps(creative_authoring_context(direction), ensure_ascii=False))
    for slot in plan["slots"]:
        if slot.get('motion_timing'):
            from pipeline.motion_timing import motion_timing_instruction
            lines.append(motion_timing_instruction(slot['motion_timing']))
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
        if slot.get('min_interaction_seconds'):
            lines.append(
                f"  action_beats 必须用 start_seconds/end_seconds/activity 连续覆盖0到{slot['duration']}秒；"
                f"activity=interaction 的真实主体攻防不少于{slot['min_interaction_seconds']}秒。"
                "transition 表示入场/接近，hold 表示停顿/收势；运镜、摆姿势不算 interaction。"
                "每个 interaction 节拍都写清 actor、target、动作和可见结果，不能用一个动作标签计入整镜时长。"
            )
        if slot.get('min_interaction_events'):
            lines.append(
                f"  至少{slot['min_interaction_events']}个独立交互事件，分别占一个 interaction 节拍；"
                "每个事件自带完整 interaction_geometry（actor/target 与节拍一致，effect_phase=active）"
                "及 effect_at_seconds/reaction_at_seconds，必须在节拍时间窗内先作用后反应。"
                "允许相邻事件交换主动执行者；不要把多个招式/物理过程塞成一段自由描述或拉长一次接触。"
                "整镜 interaction_geometry 描述终结事件，各事件使用自己的作用模式和影响范围。最多12个总节拍。"
            )
        if slot.get('result_hold_seconds'):
            lines.append(f"  末镜必须先有 peak/interaction 建立结果，再用至少{slot['result_hold_seconds']}秒的 aftermath/hold 展示结果持续；不能直接从交战跳到已经倒地。")
        if slot.get('required_outcome'):
            lines.append(f"  Shot {slot['shot_id']} 必须因果可见地实现用户结局：{slot['required_outcome']}")
    intent = plan.get('director_intent') or {}
    if intent:
        lines.append('冻结的用户意图：' + str(intent))
        if intent.get('ending_requirement') and not any(s.get('required_outcome') for s in plan['slots']):
            lines.append('本批不含全片结局槽位；不得提前完成全片结局。')
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
        # 三个槽位的短片需要在末镜完成决胜动作，不能整段强制为静态余波。
        if count == 3 and index == count - 1:
            allowed_phases = ["active", "aftermath"]

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
        "coverage_roles": (
            ["interaction", "aftermath"]
            if action and count == 3 and index == count - 1
            else _coverage_roles(focus, index, count)
        ),
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
