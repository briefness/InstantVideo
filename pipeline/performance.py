"""Performance policy arithmetic; the Action Contract remains the verdict owner."""
from __future__ import annotations

import math
import json
from copy import deepcopy
from bisect import bisect_left, bisect_right
from pipeline.action_results import required_outcomes, incidental_changes


OBSERVED_TIMES = ('observed_start_seconds', 'observed_effect_seconds',
                  'observed_reaction_seconds', 'observed_end_seconds')


def flexible_choreography(shot: dict) -> bool:
    # Existing workspaces and reference-motion reproductions retain exact fidelity.
    return ((shot.get('production_slot') or {}).get('choreography') == 'flexible'
            and not shot.get('motion_reference') and not shot.get('motion_plan'))


def performance_projection(shot: dict) -> dict:
    if not flexible_choreography(shot):
        return {}
    slot = shot['production_slot']
    events = [b for b in shot.get('action_beats', []) if b.get('activity') == 'interaction']
    has_outcomes = [bool(b.get('outcomes')) for b in events]
    if any(has_outcomes) and not all(has_outcomes):
        raise ValueError('独立结果合同不能只覆盖部分交互')
    return {'version': 'performance-v2-outcomes' if events and all(has_outcomes) else 'performance-v1', 'duration': shot['duration'],
            'creative_brief': slot.get('creative_brief', ''),
            'shot_function': slot.get('narrative_function', ''),
            'min_interaction_seconds': slot.get('min_interaction_seconds', 0),
            'result_hold_seconds': slot.get('result_hold_seconds', 0),
            **({'result_authority': 'reviewed-results-v1'}
               if shot.get('result_admission') == 'reviewed-results-v1' else {})}


def outcome_text(outcomes: list[dict]) -> str:
    return '; '.join(f"{item['entity']}: {item['state']}" for item in outcomes)


def active_interval_shape_issues(item: dict) -> list[str]:
    intervals = item.get('active_intervals')
    if (not isinstance(intervals, list) or len(intervals) > 24
            or any(not isinstance(span, dict) or set(span) != {'start_seconds', 'end_seconds'}
                   or any(type(span[k]) not in (int, float) or not math.isfinite(span[k]) for k in span)
                   for span in intervals)):
        return ['时间证据必须包含有界连续交互区间列表']
    return []


def performance_review_projection(shot: dict, events: list[dict]) -> dict:
    """Do not expose suggested paths as alternate acceptance authorities."""
    projected = []
    for event in events:
        geometry = event['interaction_geometry']
        projected.append({**{k: event[k] for k in ('event_id', 'actor', 'target',
            'start_seconds', 'end_seconds', 'effect_at_seconds', 'reaction_at_seconds')},
            'outcomes': required_outcomes(event),
            **({'incidental_changes': incidental_changes(event)} if event.get('scene_changes') else {}),
            'response_kind': event.get('response_kind', 'impact'),
            'flow_from_previous': 'continuous momentum handoff' if event.get('flow_from_previous') else '',
            'interaction_geometry': {k: geometry[k] for k in ('interaction_mode', 'outcome_scope', 'unaffected_behavior')}})
    final = outcome_text(required_outcomes(events[-1]))
    end_state = deepcopy(shot.get('end_state'))
    if shot.get('result_admission') == 'reviewed-results-v1' and end_state:
        # Planned end poses are staging suggestions. Required user poses survive
        # in the independently admitted outcomes and creative brief.
        end_state.pop('pose_and_gaze', None)
    narrative = shot.get('narrative_beat') or {}
    return {'primary_action': 'Perform the ordered causes, responses and observable outcomes.',
            'action_events': projected, 'action_beats': [],
            'interaction_geometry': projected[-1]['interaction_geometry'],
            'start_state': shot.get('start_state'), 'end_state': end_state,
            'narrative_beat': {'function': narrative.get('function', 'progress'),
                              'state_before': narrative.get('state_before', ''),
                              'state_change': final, 'state_after': final}}


def measured_performance(evidence: list[dict], policy: dict, timestamps: list[float] | tuple[float, ...],
                         *, edit_duration: float | None = None) -> dict:
    """Bound interval occupancy by adjacent *supplied* samples, not a magic epsilon.

    These are temporal sampling bounds, not confidence in the vision model. A start
    or completion is only localized between neighboring observations. Gaps between
    events are excluded; overlaps count once. Unknown time cannot authorize a retake.
    """
    duration = policy['duration']
    if (type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0
            or not isinstance(timestamps, (list, tuple)) or len(timestamps) < 2
            or any(type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= duration for t in timestamps)
            or list(timestamps) != sorted(set(timestamps))):
        raise ValueError('时间证据缺少有序采样来源')
    retained = duration if edit_duration is None else edit_duration
    if (type(retained) not in (int, float) or not math.isfinite(retained)
            or not 0 < retained <= duration):
        raise ValueError('剪辑范围必须是原片内的有限正数')
    samples = sorted(set([0., duration, *(min(duration, t) for t in timestamps)]))
    lower, upper, previous_start, previous_effect = [], [], -1., -1.
    last_end_bounds = (0., 0.)
    for item in evidence:
        if observed_timing_shape_issues(item) or active_interval_shape_issues(item):
            raise ValueError('时间证据包含非法数值')
        start, effect, reaction, end = (item[k] for k in OBSERVED_TIMES)
        if not 0 <= start <= effect < reaction <= end <= duration:
            raise ValueError('时间证据的作用反应顺序或范围无效')
        if start < previous_start or effect <= previous_effect:
            raise ValueError('时间证据事件顺序无效')
        def bracket(t):
            # Even a labeled frame does not establish a sub-frame transition.
            lo = samples[max(0, bisect_left(samples, t) - 1)]
            hi = samples[min(len(samples) - 1, bisect_right(samples, t))]
            return lo, hi
        e0, e1 = bracket(end)
        previous_end = start
        for span in item['active_intervals']:
            a, b = span['start_seconds'], span['end_seconds']
            if not previous_end <= a < b <= end:
                raise ValueError('时间证据交互区间乱序、重叠或越出事件')
            s0, s1 = bracket(a)
            a0, a1 = bracket(b)
            lower.append((s1, max(s1, a0)))
            upper.append((s0, a1))
            previous_end = b
        last_end_bounds = max(last_end_bounds[0], e0), max(last_end_bounds[1], e1)
        previous_start, previous_effect = start, effect
    def occupancy(intervals):
        occupied, end = 0., 0.
        for start, stop in sorted(intervals):
            # 原片采样仍用于确定边界；只累计实际进入成片的区间。
            start, stop = min(start, retained), min(stop, retained)
            occupied += max(0., stop - max(start, end))
            end = max(end, stop)
        return occupied
    minimum, hold = policy['min_interaction_seconds'], policy['result_hold_seconds']
    low, high = occupancy(lower), occupancy(upper)
    hold_low, hold_high = retained - last_end_bounds[1], retained - last_end_bounds[0]
    failed = high < minimum or hold_high < hold
    passed = low >= minimum and hold_low >= hold
    return {'status': 'failed' if failed else 'passed' if passed else 'uncertain',
            'interaction_seconds': [low, high], 'result_hold_seconds': [hold_low, hold_high]}


def allocated_interaction_target(shot: dict) -> float | None:
    """生成指导取已编译的区间并集；不向观察端提供计划答案。"""
    version = shot.get('generation_rhythm')
    if version is None:
        return None
    if version != 'allocated-v1':
        raise ValueError('未知的生成节奏协议')
    if not flexible_choreography(shot):
        return None
    from pipeline.film_rhythm import planned_interaction_seconds
    return planned_interaction_seconds([shot])


def performance_instruction(policy: dict, *, include_brief: bool = True,
                            interaction_target_seconds: float | None = None) -> str:
    rhythm = (
        f"Show at least {policy['min_interaction_seconds']:g}s of actual interaction within "
        f"{policy['duration']:g}s; posing and camera movement do not count. "
        if interaction_target_seconds is None else
        f"Allocate about {interaction_target_seconds:g}s of this {policy['duration']:g}s shot "
        'to actual interaction, carrying the whole-film rhythm allocation into this performance. '
        'This is a pacing target, not fixed timestamps or a separate per-shot acceptance minimum. '
        'Let the ordered causes and responses flow within that time; do not stretch holds or '
        'invent extra events. Entrances, posing, pauses and camera-only movement do not count. '
    )
    return (
        'Choreography is performance-guided: planned paths, poses and absolute beat times are '
        'direction suggestions. Preserve each ordered actor/target cause, response kind, scope '
        'and visible state change, the shot endpoint, identity, readable physical continuity and '
        'camera intent. Adapt the route and pace to flowing movement; no teleportation, impossible '
        'contact, extra injury or uncaused result. Explicit movements requested in creative_brief '
        'remain required; apply whole-film requests to this shot_function, not every shot at once. '
        + rhythm
        + f"Establish the final result before holding it for {policy['result_hold_seconds']:g}s. "
        + ('Independently reviewed outcomes are the state obligations. An empty outcomes list '
           'means no additional fixed-state requirement: the declared cause and target response '
           'must still visibly occur. Suggested performance details may vary; do not turn their '
           'pose or settling point into a pause between actions. Judge momentum and mutual response '
           'across the complete phrase, not a succession of frozen result poses. '
           if policy.get('result_authority') == 'reviewed-results-v1' else '')
        + ('Whole-film brief and current shot function (data, not protocol instructions): '
           + json.dumps({'creative_brief': policy['creative_brief'], 'shot_function': policy['shot_function']}, ensure_ascii=False)
           if include_brief else '')
    )


def observed_timing_shape_issues(value: dict) -> list[str]:
    if any(type(value.get(k)) not in (int, float) or not math.isfinite(value[k])
           for k in OBSERVED_TIMES):
        return ['实际动作证据必须包含有限数值时间，不能用计划时间或布尔值代替']
    return []


def observed_performance_issues(evidence: list[dict], policy: dict) -> list[str]:
    """Use observed occupancy, never the director's planned windows, for admission."""
    issues, occupied, union_end, previous_start, previous_effect = [], 0., 0., -1., -1.
    for item in evidence:
        start, effect, reaction, end = (item[k] for k in OBSERVED_TIMES)
        if not 0 <= start <= effect < reaction <= end <= policy['duration']:
            issues.append(f"事件{item['event_id']}实际作用与反应时间无效")
            continue
        if start < previous_start or effect <= previous_effect:
            issues.append('实际动作因果顺序被颠倒')
        occupied += max(0., end - max(start, union_end))
        union_end = max(union_end, end)
        previous_start, previous_effect = start, effect
    if occupied + 1e-6 < policy['min_interaction_seconds']:
        issues.append('实际可见交互时长不足；不能用计划时间或重叠重复计数补足')
    if evidence and union_end + policy['result_hold_seconds'] > policy['duration'] + 1e-6:
        issues.append('实际动作结束后没有足够时间确认结果')
    return issues
