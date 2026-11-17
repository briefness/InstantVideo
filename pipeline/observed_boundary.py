"""已验收世界状态是续拍起点；摄影目标与它分开，观察问题共享同一来源。"""
from copy import deepcopy
import json
import re


EXECUTION_VERSION = 'scene-instructions-v2'
VERSION = 'observed-origin-v1'
STATE_FIELDS = ('location', 'subject', 'action_phase', 'camera', 'screen_direction',
                'pose_and_gaze', 'prop_state', 'open_motion', 'lighting')


def origin(shot: dict) -> dict | None:
    value = shot.get('observed_origin')
    if value is None:
        return None
    if (shot.get('generation_execution') != EXECUTION_VERSION or not isinstance(value, dict)
            or set(value) != {'version', 'source_sha256', 'state'} or value['version'] != VERSION
            or not isinstance(value['source_sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', value['source_sha256'])):
        raise ValueError('续拍真实起点缺少有效来源')
    state = value['state']
    if (not isinstance(state, dict) or not set(state) <= set(STATE_FIELDS)
            or not state.get('location') or not state.get('subject')
            or any(not isinstance(v, str) or len(v) > 2000 for v in state.values())):
        raise ValueError('续拍真实起点结构无效')
    start = shot.get('start_state') or {}
    if any(start.get(k) != v for k, v in state.items() if k != 'camera'):
        raise ValueError('续拍起点与已验收状态不一致')
    return deepcopy(value)


def instruction(shot: dict) -> str:
    from pipeline.temporal_evidence import motion_instruction
    from pipeline.spatial_guidance import guidance_instruction
    bound = origin(shot)
    if not bound:
        return ''
    physical = {k: v for k, v in bound['state'].items() if k != 'camera'}
    spatial = guidance_instruction(shot)
    return ('Opening world state observed in the accepted prior frame: '
            + json.dumps(physical, ensure_ascii=False, sort_keys=True)
            + '. This is the physical origin of the next action, not a pose to hold throughout. '
            'Authored starting poses and prop placements cannot reset it; realize the local work from this origin. '
            'The current camera plan selects the new framing independently. Previous observed camera: '
            + bound['state'].get('camera', 'unspecified') + '. ' + motion_instruction(shot)
            + (' ' + spatial if spatial else ''))


def assessment_criteria(shot: dict) -> dict:
    bound = origin(shot)
    if not bound:
        return {}
    common = {'kind': 'state', 'beat_id': 0, 'source_sha256': bound['source_sha256'],
              'previous_observed_state': bound['state']}
    result = {'boundary/physical_progress': {**common, 'domain': 'continuity',
        'state': 'The current opening continues the previously observed physical state without an '
                 'unexplained reset or replay of completed work. Describe the previous endpoint and '
                 'the current opening separately, then any visible transition into the local action. '
                 'A different incidental gaze or resting pose is not a violation; invented progress '
                 'or reversed physical results are. Do not infer the opening from the current ending.'}}
    if shot.get('continuity_from_previous') == 'intentional_cut' and shot.get('composition_change') in {'medium', 'large'}:
        result['boundary/camera_change'] = {**common, 'domain': 'camera', 'camera': deepcopy(shot['camera']),
            'state': 'The intentional cut realizes the current camera plan with a readable framing or '
                     'viewpoint change from the prior shot. Describe actual framing of both sides and '
                     'their visible difference. Subject or prop movement alone is not camera change; '
                     'do not repeat a planned framing label as an observed fact.'}
    return result
