"""把已编译动作装配为拍摄指令；空间关联不产生新的验收义务。"""
from __future__ import annotations

from copy import deepcopy
import re

from pipeline.action_results import (
    required_outcomes, functional_outcomes, directed_outcomes, incidental_changes, result_claims,
)
from pipeline.action_timing import beat_timing_instruction
from pipeline.performance import allocated_interaction_target, flexible_choreography
from pipeline.result_observations import completion_contract
from pipeline.creative_direction import creative_direction_projection, creative_direction_instruction
from pipeline.motion_timing import motion_timing_projection, motion_timing_instruction


VERSION = 'scene-instructions-v1'
from pipeline.observed_boundary import EXECUTION_VERSION, instruction as observed_instruction
VERSIONS = {VERSION, EXECUTION_VERSION}


def local_goal_instructions(catalog: dict) -> list[str]:
    """保留目标与本镜范围；实现动作不能替代目标或被重新标成必要功能。"""
    from pipeline.assessment_scope import local_assessment
    from pipeline.support_witness import POLICIES
    if catalog.get('evidence_policy') in POLICIES:
        from pipeline.goal_allocation import criterion_instruction
        return [criterion_instruction(c) for c in catalog['criteria'].values() if c['kind']=='completion']
    criteria = local_assessment(catalog)['criteria'] if catalog else {}
    return [('; '.join(u['action'] for u in c['execution_scope']['units'])
             if c['kind'] == 'contribution' else c['action'])
            for c in criteria.values() if c['kind'] in {'completion', 'contribution'}]


def shooting_projection(shot: dict, storyboard: dict) -> dict:
    """引用已有实体 ID，保存来源；不从自然语言猜测角色、目的地或可见性。"""
    # 编译稿已经确定实体主键；这里不能再使用历史描述性别名归一化。
    declared = [*(c['name'] for c in storyboard.get('characters', [])),
                *shot.get('characters', []), *shot.get('key_props', []),
                *shot.get('continuity_props', []), *storyboard.get('theme_elements', [])]
    ids = sorted(set(declared), key=lambda value: (-len(value), value))
    # ASCII ID 可紧邻中文；A 不应误匹配 Panel/A2，长 ID 优先占据自己的文本范围。
    alternatives = [
        ('(?<![A-Za-z0-9_])' if re.match(r'[A-Za-z0-9_]', name[0]) else '')
        + re.escape(name)
        + ('(?![A-Za-z0-9_])' if re.match(r'[A-Za-z0-9_]', name[-1]) else '')
        for name in ids if name
    ]
    pattern = re.compile('|'.join(alternatives)) if alternatives else None
    positions = (shot.get('camera') or {}).get('screen_positions') or {}
    actions = []
    for index, beat in enumerate(shot.get('action_beats', []), 1):
        geometry = beat.get('interaction_geometry') or {}
        references = {key: beat.get(key, '') for key in ('actor', 'target', 'action', 'visible_result')}
        references.update({f'interaction_geometry/{key}': geometry.get(key, '')
                           for key in ('source', 'effect_region')})
        references.update({f'{path}/state': claim['state'] for path, claim in result_claims(beat).items()
                           if path not in (beat.get('result_directions') or [])})
        anchors = {}
        for path, value in references.items():
            matches = ([value] if path in ('actor', 'target') and value in ids else
                       [m.group() for m in pattern.finditer(value)] if pattern and value else [])
            for entity in dict.fromkeys(matches):
                entry = anchors.setdefault(entity, {'placement': positions.get(entity, ''), 'sources': []})
                entry['sources'].append(path)
        actions.append({
            'beat_id': index,
            **{key: deepcopy(beat.get(key, '')) for key in (
                'actor', 'target', 'activity', 'action', 'visible_result', 'start_seconds', 'end_seconds',
                'temporal_relation', 'flow_from_previous', 'effect_at_seconds', 'reaction_at_seconds',
                'response_kind')},
            'geometry': deepcopy(geometry), 'spatial_anchors': anchors,
            'required_states': required_outcomes(beat),
            'functional_responses': functional_outcomes(beat),
            'performance_details': directed_outcomes(beat),
            'incidental_changes': incidental_changes(beat),
        })
    return deepcopy({
        'version': shot.get('generation_execution', VERSION), 'shot_id': shot['shot_id'], 'duration': shot['duration'],
        **({'observed_origin_instruction': observed_instruction(shot)}
           if shot.get('generation_execution') == EXECUTION_VERSION else {}),
        'timing': ('event-relative' if shot.get('execution_semantics') == 'intent-bounded-v1' and flexible_choreography(shot)
                   else 'suggested-windows'), 'actions': actions,
        'completion': completion_contract(shot),
        'interaction_seconds': allocated_interaction_target(shot),
        'minimum_interaction_seconds': (shot.get('production_slot') or {}).get('min_interaction_seconds', 0),
        'result_hold_seconds': (shot.get('production_slot') or {}).get('result_hold_seconds', 0),
        'required_final_outcome': (shot.get('production_slot') or {}).get('required_outcome', ''),
        'creative_direction': creative_direction_projection(shot),
        'motion_timing': motion_timing_projection(shot),
        'creative_context': ((shot.get('production_slot') or {}).get('creative_brief', '')
                             if shot.get('result_admission') != 'reviewed-results-v1' else ''),
        'visual_description': shot.get('visual_description', ''), 'mood': shot.get('mood', ''),
        'camera': shot.get('camera') or {}, 'blocking': shot.get('blocking') or {},
        'observed_entry_path': shot.get('observed_entry_path', ''),
    })


def shooting_instruction_parts(projection: dict) -> list[str]:
    """给生成器的是局部执行上下文，不是审查协议或全片计划。"""
    from pipeline.goal_allocation import criterion_instruction
    from pipeline.performance import outcome_text
    from pipeline.camera_contract import compile_camera
    from pipeline.models import CameraSpec, CharacterBlockingSpec
    from pipeline.causality import event_execution_instruction
    camera = CameraSpec.model_validate(projection['camera']).model_dump()
    camera_parts, _ = compile_camera({'camera': camera})
    parts = list(camera_parts)
    for key in ('composition', 'speed', 'end_framing'):
        if camera.get(key):
            parts.append(f'Camera {key}: {camera[key]}')
    # 已关联的锚点就近呈现；其余空间布局保留，但不升级为每镜必须可见。
    linked = {name for action in projection['actions'] for name in action['spatial_anchors']}
    remaining = [f'{name}: {value}' for name, value in camera['screen_positions'].items() if name not in linked]
    if remaining:
        parts.append('Composition across the shot: ' + '; '.join(remaining))
    for name, raw in projection['blocking'].items():
        blocking = CharacterBlockingSpec.model_validate(raw).model_dump()
        parts.append(f'Performer staging for {name}: ' + '; '.join(f'{k}: {v}' for k, v in blocking.items() if v))
    criteria = projection['completion'].get('criteria', {})
    goals = (local_goal_instructions(projection['completion']) if projection['version'] == EXECUTION_VERSION
             else [criterion_instruction(c) for c in criteria.values() if c['kind'] == 'completion'])
    if goals:
        parts.append('Required observable functions: ' + '; '.join(goals))
    if projection['required_final_outcome']:
        parts.append('Required final visible outcome: ' + projection['required_final_outcome'])
    if projection.get('observed_origin_instruction'):
        parts.append(projection['observed_origin_instruction'])
    parts.append(f"Play this {projection['duration']:g}s shot as connected action. "
                 + ('Preserve the following event dependencies at the requested performer tempo.'
                    if projection['timing'] == 'event-relative' else
                    'The following time windows guide pacing, not held poses; preserve their sequence and overlap.'))
    occupancy = projection['interaction_seconds']
    if occupancy is not None:
        parts.append(f"Allocate about {occupancy:g}s of this {projection['duration']:g}s shot "
                     'to the authored interactions at the requested performer tempo; do not stretch '
                     'gestures or substitute pauses and camera movement for the interaction')
    elif projection['minimum_interaction_seconds']:
        parts.append(f"Keep authored interactions active for at least {projection['minimum_interaction_seconds']:g}s")
    if projection['result_hold_seconds']:
        parts.append(f"Leave {projection['result_hold_seconds']:g}s for the established result to read")
    if projection['observed_entry_path']:
        parts.append('Approach the first existing action from the observed start: ' + projection['observed_entry_path'])
    for action in projection['actions']:
        clock = ('' if projection['timing'] == 'event-relative'
                 else f"{action['start_seconds']:g}-{action['end_seconds']:g}s; ")
        text = f"Beat {action['beat_id']}: {clock}{action['actor']} {action['action']}"
        if action['target']:
            text += f"; target={action['target']}"
        if action['temporal_relation']:
            text += '; ' + beat_timing_instruction(action, action['beat_id'])
        if action['activity'] == 'interaction':
            text += '; response=' + (action['response_kind'] or 'impact')
            response = event_execution_instruction(action)
            if response:
                text += '; ' + response
        geometry = action['geometry']
        if geometry:
            text += '; ' + '; '.join(
                f'{label}: {geometry[key]}' for key, label in (
                    ('source', 'physical source'), ('effect_region', 'action region'),
                    ('actor_screen_position', 'actor placement through this action'),
                    ('target_screen_position', 'target placement through this action'),
                    ('interaction_mode', 'physical mode'), ('outcome_scope', 'affected scope'),
                    ('reaction_scope', 'response'), ('unaffected_behavior', 'unaffected surroundings'),
                ) if geometry.get(key))
            # 场的运动不能被当作人物/物体整体运动；保留其物理含义而非暴露无主语的 static。
            if geometry.get('effect_motion') not in (None, '', 'none'):
                text += f"; motion of the causal effect field (not actor/target travel): {geometry['effect_motion']}"
            if geometry.get('must_share_frame'):
                text += '; show the causal participants and action region together'
            if geometry.get('line_of_action_visible'):
                text += '; keep the line of action visible'
            if geometry.get('occlusion_policy'):
                text += f"; occlusion: {geometry['occlusion_policy']}"
        placements = [f"{name}: {anchor['placement']}" for name, anchor in action['spatial_anchors'].items()
                      if anchor['placement']]
        if placements:
            text += '; spatial context for this action: ' + '; '.join(placements)
        if action['required_states']:
            text += '; required visible state change: ' + outcome_text(action['required_states'])
        label = 'suggested visible realization' if projection['completion'].get('intent_policy') else 'visible result'
        text += f'; {label}: ' + action['visible_result']
        for key, label in (('functional_responses', 'suggested response realization'),
                           ('performance_details', 'replaceable performance details'),
                           ('incidental_changes', 'optional scene details')):
            if action[key]:
                text += f'; {label}: ' + outcome_text(action[key])
        parts.append(text)
    for key, label in (('creative_context', 'Creative context'),
                       ('visual_description', 'Authored visual direction'), ('mood', 'Shot mood')):
        if projection[key]:
            parts.append(f'{label}: ' + projection[key])
    parts.extend(p for p in (creative_direction_instruction(projection['creative_direction']),
                            motion_timing_instruction(projection['motion_timing'])) if p)
    return parts
