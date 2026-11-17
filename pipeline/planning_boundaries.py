"""以相邻片段的原始证据审查时间进展，不用全片批准代替接缝判断。"""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


VERSION = 'planning-boundaries-v1'
INSTRUCTION = '''存在boundary_review时，用boundaries替代continuity_valid。逐项判断固定键对应的真实剪辑接缝。
previous提供前镜全部已完成动作及终态，next提供后镜起态及动作；必须读到前镜末尾，不能只引用前镜中途动作。
state_relation判断后镜起态是否与前镜终态及用户要求相容；action_progression判断后镜是否推进或保持已经建立的状态。
supported表示原稿证据支持，contradicted表示矛盾，undetermined表示证据不足。不能用无明显问题代替支持。
切换机位或特写不意味着时间倒退。成片按顺序使用每段完整时长，不会自动裁去重复动作。
已完成的交接、落地、到达等不能在下一镜重新变成尚未完成；持续状态、周期性动作、换场和用户明确要求的非线性叙事则按完整语境判断，不以文字相同判为重复。
previous_evidence和next_evidence分别引用该接缝两侧的固定路径，并解释两侧证据之间的关系；不能补写回放、画外变化或隐含剪辑来替原稿解围。
没有接缝的批次不需要该字段。跨批次的previous_shot只提供接缝左侧证据，不重审其动作。
'''


def boundary_review_context(storyboard: dict) -> dict | None:
    shots = storyboard['shots']
    if not any(s.get('authoring_protocol') == 'scene-units-v1'
               for s in (storyboard.get('production_plan') or {}).get('slots', [])):
        return None
    previous = (storyboard.get('review_scope') or {}).get('previous_shot')
    left_path = '/review_scope/previous_shot'
    boundaries = {}
    for index, shot in enumerate(shots):
        right_path = f'/shots/{index}'
        if previous is not None:
            key = f"{previous['shot_id']}->{shot['shot_id']}"
            if key in boundaries or previous['shot_id'] >= shot['shot_id']:
                raise ValueError('接缝镜头必须唯一且按实际播放顺序排列')
            boundaries[key] = {
                'previous': _source_fields(previous, left_path, 'end_state'),
                'next': _source_fields(shot, right_path, 'start_state'),
                'previous_scene': previous.get('scene_id'), 'next_scene': shot.get('scene_id'),
                'edit': shot.get('continuity_from_previous'),
            }
        previous, left_path = shot, right_path
    if not boundaries:
        return None
    return {'version': VERSION, 'playback': 'ordered_full_clips', 'boundaries': boundaries}


def _source_fields(shot: dict, prefix: str, state_name: str) -> dict:
    fields = {}
    for name, value in (shot.get(state_name) or {}).items():
        if isinstance(value, str) and value.strip():
            fields[f'{prefix}/{state_name}/{name}'] = value
    for index, beat in enumerate(shot.get('action_beats', [])):
        for name in ('action', 'visible_result'):
            value = beat.get(name)
            if isinstance(value, str) and value.strip():
                fields[f'{prefix}/action_beats/{index}/{name}'] = value
    return fields


class BoundaryJudgment(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, hide_input_in_errors=True)
    state_relation: Literal['supported', 'contradicted', 'undetermined']
    action_progression: Literal['supported', 'contradicted', 'undetermined']
    previous_evidence: list[str] = Field(min_length=1, max_length=24)
    next_evidence: list[str] = Field(min_length=1, max_length=24)
    reason: str = Field(min_length=1, max_length=1600)


def boundary_response_schema(base: dict, context: dict) -> dict:
    schema = deepcopy(base)
    items = {}
    for key, pair in context['boundaries'].items():
        judgment = BoundaryJudgment.model_json_schema()
        for side in ('previous', 'next'):
            paths = list(pair[side])
            if not paths:
                raise ValueError('接缝缺少原始状态或动作证据')
            judgment['properties'][f'{side}_evidence']['items'] = {'type': 'string', 'enum': paths}
        items[key] = judgment
    schema['properties'].pop('continuity_valid')
    schema['required'].remove('continuity_valid')
    schema['properties']['boundaries'] = {'type': 'object', 'properties': items,
                                        'required': list(items), 'additionalProperties': False}
    schema['required'].append('boundaries')
    return schema


def decode_boundary_review(raw: dict, context: dict) -> tuple[dict, dict]:
    values = raw.get('boundaries')
    if ('continuity_valid' in raw or not isinstance(values, dict)
            or set(values) != set(context['boundaries'])):
        raise ValueError('连续性必须逐接缝提供两侧证据，不能只返回整体批准或遗漏接缝')
    evidence, failures = {}, []
    for key, pair in context['boundaries'].items():
        judgment = BoundaryJudgment.model_validate(values[key])
        if not judgment.reason.strip():
            raise ValueError('接缝判断必须解释原始证据关系')
        sources = {}
        for side in ('previous', 'next'):
            refs = getattr(judgment, f'{side}_evidence')
            if len(refs) != len(set(refs)) or any(ref not in pair[side] for ref in refs):
                raise ValueError('接缝证据必须分别引用对应前镜和后镜，不能重复或跨接缝借用')
            sources[side] = {ref: pair[side][ref] for ref in refs}
        evidence[key] = {**judgment.model_dump(), 'source_fields': sources}
        if judgment.state_relation != 'supported' or judgment.action_progression != 'supported':
            failures.append(f'接缝 {key}: {judgment.reason}')
    result = deepcopy(raw)
    result.pop('boundaries')
    result['continuity_valid'] = not failures
    if failures:
        result['reason'] = '；'.join(failures)[:1600]
    return result, evidence


def authoring_progress(plan: dict, completed: list[dict], slots: list[dict]) -> dict:
    """只投影已写出的事实，不从自然语言猜测已完成了哪些用户事件。"""
    return {'version': 'authoring-progress-v1',
            'completed': [{'shot_id': s['shot_id'], 'scene_id': s['scene_id'],
                           'state_after': s['story_state_after']} for s in completed],
            'current_boundary': deepcopy(completed[-1]['end_state']) if completed else None,
            'remaining_slots': deepcopy(slots[len(completed):]),
            'total_duration': sum(s['duration'] for s in plan['slots'])}
