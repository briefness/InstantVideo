"""Time occupancy arithmetic used by the shared Action Contract compiler."""
from __future__ import annotations

import math

MIN_INTERACTION_MS = 350
MIN_PASSIVE_MS = 100


def passive_floor_ms(activities: list[str], result_hold_seconds: float) -> int:
    """Shared compiler capacity bound; no statement about observed performance."""
    floor = sum(MIN_PASSIVE_MS for kind in activities if kind != 'interaction')
    if activities and activities[-1] == 'hold':
        floor += max(0, round(result_hold_seconds * 1000) - MIN_PASSIVE_MS)
    return floor

RESPONSE_MODES = {
    'impact': ('direct_contact', 'directed_path', 'area_effect', 'indirect_effect'),
    'evasion': ('directed_path',),
    'deflection': ('direct_contact', 'directed_path'),
}


def _number(value: object) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def sequence_required(shot: dict) -> bool:
    slot = shot.get('production_slot')
    beats = shot.get('action_beats')
    return bool((isinstance(slot, dict) and slot.get('min_interaction_events')) or (
        isinstance(beats, list) and any(isinstance(b, dict) and any(
            b.get(k) is not None for k in ('interaction_geometry', 'effect_at_seconds', 'reaction_at_seconds', 'response_kind', 'flow_from_previous')
        ) for b in beats)))


def action_sequence_issues(shot: dict) -> list[str]:
    """Validate event ownership and cause/reaction times, without interpreting prose."""
    if not sequence_required(shot):
        return []
    from pipeline.models import InteractionGeometrySpec
    label = f"Shot {shot.get('shot_id', '?')}"
    beats = shot.get('action_beats')
    if not isinstance(beats, list) or not beats or any(not isinstance(b, dict) for b in beats):
        return [f'{label}: 动作事件必须是非空节拍列表']
    issues, count = [], 0
    for index, beat in enumerate(beats, 1):
        raw = beat.get('interaction_geometry')
        if beat.get('activity') != 'interaction':
            if any(beat.get(k) is not None for k in ('interaction_geometry', 'effect_at_seconds', 'reaction_at_seconds', 'response_kind', 'flow_from_previous')):
                issues.append(f'{label}: 节拍{index}不是交互，不能携带新的作用事件')
            continue
        count += 1
        flow = beat.get('flow_from_previous')
        if flow is not None and (
            not isinstance(flow, str) or not flow.strip() or len(flow) > 800 or index == 1
            or not any(isinstance(entity, str) and entity and entity in (
                beats[index - 2].get('actor'), beats[index - 2].get('target'))
                for entity in (beat.get('actor'), beat.get('target')))
        ):
            issues.append(
                f'{label}: action_beats[{index - 1}].flow_from_previous 无效；'
                '它只描述本镜内前一节拍的共同参与者和具体动作。首节拍必须省略此字段；'
                '跨镜衔接应通过 start_state/open_motion 和 continuity_from_previous 描述')
        try:
            geometry = InteractionGeometrySpec.model_validate(raw).model_dump()
        except (ValueError, TypeError):
            issues.append(f'{label}: 交互事件{index}缺少独立的完整 interaction_geometry')
            continue
        field_checks = {
            'actor (须与本节拍 actor 相同)': geometry['actor'] == beat.get('actor'),
            'target (须与本节拍 target 相同)': geometry['target'] == beat.get('target'),
            'effect_phase (须为 active)': geometry['effect_phase'] == 'active',
            'interaction_mode': geometry['interaction_mode'] in ('direct_contact', 'directed_path', 'area_effect', 'indirect_effect'),
            'outcome_scope (single/subset/all)': geometry['outcome_scope'] in ('single', 'subset', 'all'),
            'effect_motion (static/sweep/expand/propagate)': geometry['effect_motion'] in ('static', 'sweep', 'expand', 'propagate'),
        }
        invalid_fields = [field for field, valid in field_checks.items() if not valid]
        invalid_fields.extend(f'{key} (不可为空)' for key in
                              ('actor', 'target', 'source', 'effect_region', 'reaction_scope', 'unaffected_behavior')
                              if not geometry.get(key, '').strip())
        if invalid_fields:
            issues.append(f'{label}: action_beats[{index - 1}].interaction_geometry 字段不完整/不一致：' + ', '.join(invalid_fields))
        if ((geometry['interaction_mode'] == 'direct_contact' and not geometry['must_share_frame'])
                or (geometry['interaction_mode'] == 'directed_path' and not geometry['line_of_action_visible'])):
            issues.append(f'{label}: 交互事件{index}缺少同框接触或可见作用路径')
        response = beat.get('response_kind') if beat.get('response_kind') is not None else 'impact'
        if not isinstance(response, str) or geometry['interaction_mode'] not in RESPONSE_MODES.get(response, ()):
            issues.append(f'{label}: 交互事件{index}的命中/闪避/拨挡与作用路径不一致')
        times = [beat.get(k) for k in ('start_seconds', 'effect_at_seconds', 'reaction_at_seconds', 'end_seconds')]
        if (not all(_number(t) for t in times) or not times[0] <= times[1] < times[2] <= times[3]):
            issues.append(f'{label}: 交互事件{index}必须在自身时间窗内先作用、后反应')
    slot = shot.get('production_slot') or {}
    if not isinstance(slot, dict):
        return issues + [f'{label}: 事件计划格式无效']
    minimum = slot.get('min_interaction_events', 0)
    if type(minimum) is not int or minimum < 0 or minimum > 12 or count < minimum:
        issues.append(f'{label}: 独立交互事件数量{count}不足计划{minimum}；拉长一次接触不能代替连续交互')
    return issues


def timing_required(shot: dict) -> bool:
    slot = shot.get('production_slot') or {}
    beats = shot.get('action_beats', [])
    if not isinstance(slot, dict) or not isinstance(beats, list):
        return True
    return bool(sequence_required(shot) or slot.get('min_interaction_seconds') or slot.get('result_hold_seconds') or any(
        any(beat.get(key) is not None for key in ('start_seconds', 'end_seconds', 'activity'))
        for beat in beats if isinstance(beat, dict)))


def action_timing_issues(shot: dict) -> list[str]:
    if not timing_required(shot):
        return []
    label = f"Shot {shot.get('shot_id', '?')}"
    beats, duration = shot.get('action_beats', []), shot.get('duration')
    slot = shot.get('production_slot') or {}
    if (not isinstance(slot, dict) or not isinstance(beats, list) or not beats
            or any(not isinstance(beat, dict) for beat in beats)
            or not _number(duration) or duration <= 0):
        return [f'{label}: 缺少可执行的动作时间轴']
    issues, cursor, interaction, interaction_end = action_sequence_issues(shot), 0., 0., 0.
    for index, beat in enumerate(beats):
        start, end, activity = (beat.get(k) for k in ('start_seconds', 'end_seconds', 'activity'))
        if (not _number(start) or not _number(end) or start < 0 or end <= start
                or end > duration or activity not in ('interaction', 'transition', 'hold')
                or beat.get('phase') not in ('trigger', 'peak', 'aftermath')):
            issues.append(f'{label}: 节拍{index + 1}时间/活动类型无效')
            continue
        previous = beats[index - 1] if index else {}
        # 仅明确承接且作用时刻有序的交互可搭接，不能用重叠掩盖乱序或空档。
        linked_overlap = bool(
            index and isinstance(beat.get('flow_from_previous'), str) and beat['flow_from_previous'].strip()
            and activity == previous.get('activity') == 'interaction'
            and _number(previous.get('start_seconds')) and previous['start_seconds'] <= start < cursor
            and end >= cursor and _number(previous.get('effect_at_seconds'))
            and _number(beat.get('effect_at_seconds'))
            and previous['effect_at_seconds'] < beat['effect_at_seconds']
        )
        if abs(start - cursor) > 1e-6 and not linked_overlap:
            issues.append(f'{label}: 动作时间轴有空隙或重叠')
        cursor = end
        if activity == 'interaction':
            if end - start + 1e-6 < .35:
                issues.append(f'{label}: 单个交互节拍少于0.35秒，无法清楚展示作用和反应')
            if not beat.get('actor') or not beat.get('target') or not beat.get('visible_result'):
                issues.append(f'{label}: 交互时段缺少执行者、目标或可见结果')
            if beat.get('phase') == 'aftermath':
                issues.append(f'{label}: aftermath 不能包含新的交互作用；把决胜动作放入 peak')
            interaction += max(0., end - max(start, interaction_end))
            interaction_end = max(interaction_end, end)
    if abs(cursor - duration) > 1e-6:
        issues.append(f'{label}: 动作时间轴必须完整覆盖镜头时长')
    minimum = slot.get('min_interaction_seconds', 0)
    hold = slot.get('result_hold_seconds', 0)
    if not _number(minimum) or not _number(hold) or minimum < 0 or hold < 0 or minimum + hold > duration:
        issues.append(f'{label}: 动作时间预算无效')
    else:
        if interaction + 1e-6 < minimum:
            issues.append(f'{label}: 有效交互时长{interaction:g}秒不足计划{minimum:g}秒；入场、运镜和停顿不能冒充动作')
        if hold:
            last = beats[-1]
            established = any(b.get('phase') == 'peak' and b.get('activity') == 'interaction'
                              and b.get('visible_result') for b in beats[:-1])
            if (not established or last.get('phase') != 'aftermath' or last.get('activity') != 'hold'
                    or not last.get('visible_result') or not _number(last.get('start_seconds'))
                    or not _number(last.get('end_seconds'))
                    or last['end_seconds'] - last['start_seconds'] + 1e-6 < hold):
                issues.append(f'{label}: 结局必须先由可见作用建立，再留出至少{hold:g}秒持续确认')
    return issues


def timing_projection(shot: dict) -> dict:
    if not timing_required(shot) or action_timing_issues(shot):
        return {}
    return {'version': 'action-timing-v1', 'duration': shot['duration'],
            'intervals': [{key: beat[key] for key in ('start_seconds', 'end_seconds', 'activity', 'phase')}
                          for beat in shot['action_beats']],
            'min_interaction_seconds': (shot.get('production_slot') or {}).get('min_interaction_seconds', 0),
            'result_hold_seconds': (shot.get('production_slot') or {}).get('result_hold_seconds', 0)}


def sequence_events(shot: dict) -> list[dict]:
    """Canonical event data consumed exclusively through the Action Contract."""
    if not sequence_required(shot) or action_timing_issues(shot):
        return []
    return [dict(event_id=index, **{k: beat[k] for k in (
        'actor', 'target', 'action', 'visible_result', 'start_seconds', 'end_seconds',
        'effect_at_seconds', 'reaction_at_seconds', 'interaction_geometry')},
        **{k: beat[k] for k in ('flow_from_previous', 'response_kind', 'outcomes', 'scene_changes', 'result_directions') if beat.get(k) is not None})
        for index, beat in enumerate(shot['action_beats'], 1) if beat.get('activity') == 'interaction']


def event_shot(event: dict) -> dict:
    """Project one event onto the existing single-effect compiler without recursion."""
    return {'interaction_geometry': event['interaction_geometry'],
            'primary_action': event['action'],
            'action_beats': [{'phase': 'peak', **{k: event[k] for k in ('actor', 'target', 'action', 'visible_result')}}],
            'end_state': {'action_phase': event['visible_result']}}


def timing_constraint(shot: dict) -> str:
    projection = timing_projection(shot)
    if not projection:
        return ''
    windows = '; '.join(f"{w['start_seconds']:g}-{w['end_seconds']:g}s {w['phase']}/{w['activity']}"
                        for w in projection['intervals'])
    flow = (
        "Explicitly linked overlapping interactions unfold together; preserve momentum without resetting "
        "between beats. Count overlapping time only once, not as extra action duration. "
        if any(beat.get('flow_from_previous') for beat in shot['action_beats']) else ''
    )
    return (f"Timed action schedule: {windows}. Follow the ordered action beats within these windows. "
            f"Show at least {projection['min_interaction_seconds']:g}s of visible subject interaction; "
            "camera motion, posing, anticipation and an outcome held still do not count. "
            + flow +
            "Establish the contracted result through visible cause and reaction before holding it; "
            "do not start a new strike in an aftermath interval.")
