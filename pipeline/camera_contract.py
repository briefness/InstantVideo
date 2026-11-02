"""Bounded camera intent shared by request compilation, review and provenance."""
from __future__ import annotations

import re

from pipeline.production_plan import classify_framing


_MOVES = {
    'fixed': ('fixed', 'static', 'locked off', '固定', '固定镜头'),
    'dolly in': ('dolly in', 'push in', '推进', '推镜'),
    'dolly out': ('dolly out', 'pull out', 'pull back', '拉远', '拉镜'),
    'pan left': ('pan left', '左摇'),
    'pan right': ('pan right', '右摇'),
    'pan': ('pan', '摇镜'),
    'tilt up': ('tilt up', '上摇'),
    'tilt down': ('tilt down', '下摇'),
    'tracking': ('tracking', 'track', 'follow', '跟拍'),
    'truck left': ('truck left', 'track left', '左移'),
    'truck right': ('truck right', 'track right', '右移'),
    'crane up': ('crane up', '升镜'),
    'crane down': ('crane down', '降镜'),
    'orbit': ('orbit', 'arc', '环绕', '弧形环绕'),
    'zoom in': ('zoom in', '变焦推进'),
    'zoom out': ('zoom out', '变焦拉远'),
    'handheld': ('handheld', '手持'),
}
_ALIASES = {alias: move for move, aliases in _MOVES.items() for alias in aliases}
SUPPORTED_CAMERA_MOVEMENTS = tuple(_MOVES)
_SPEEDS = {'slow': 'slow', 'gentle': 'slow', 'subtle': 'slow', 'smooth': 'smooth',
           'fast': 'fast', '缓慢': 'slow', '快速': 'fast'}
_FRAMING = {'wide': 'wide shot', 'medium': 'medium shot', 'close_detail': 'close detail shot'}


def camera_movement(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or len(value) > 120:
        return '', ''
    text = ' '.join(value.casefold().replace('_', ' ').replace('-', ' ').split())
    speed = ''
    for prefix, label in _SPEEDS.items():
        if text.startswith(prefix + ' '):
            text, speed = text[len(prefix) + 1:], label
            break
    return _ALIASES.get(text, ''), speed


def camera_contract_issues(shot: dict) -> list[str]:
    camera = shot.get('camera')
    if not isinstance(camera, dict) or not camera.get('primary_movement'):
        return []
    movement, _ = camera_movement(camera['primary_movement'])
    if movement:
        return []
    return [f"Shot {shot.get('shot_id', '?')}: 运镜无法执行；primary_movement 请使用 "
            + ', '.join(_MOVES) + '（可加 slow/smooth/fast 前缀）']


def compile_camera(shot: dict) -> tuple[tuple[str, ...], dict]:
    camera = shot.get('camera')
    if not isinstance(camera, dict):
        return (), {}
    movement, speed = camera_movement(camera.get('primary_movement'))
    declared_speed = camera.get('speed')
    if not speed and isinstance(declared_speed, str):
        speed = _SPEEDS.get(declared_speed.strip().casefold(), '')
    parts, projection = [], {}
    if movement:
        projection['camera_movement'] = movement
        if movement == 'fixed':
            parts.append('positive lock - camera remains fixed throughout the shot')
        else:
            projection['camera_speed'] = speed or 'controlled'
            parts.append(f"camera movement: {speed or 'controlled'} {movement}; "
                         'keep the action and required participants readable during the move')
    framing = {}
    for boundary in ('start', 'end'):
        value = camera.get(f'{boundary}_framing')
        if isinstance(value, str) and value.strip():
            family = classify_framing(value)
            if family in _FRAMING:
                framing[boundary] = _FRAMING[family]
    if framing:
        projection['camera_framing'] = framing
        parts.append('camera framing: ' + ', '.join(f'{key} at {value}' for key, value in framing.items()))
    composition = camera.get('composition')
    if isinstance(composition, str):
        text = composition.casefold().replace('-', ' ')
        cues = [cue for cue in ('low angle', 'high angle', 'eye level', 'over the shoulder',
                               'full body', 'two shot', 'deep focus')
                if re.search(r'\b' + re.escape(cue) + r'\b', text)]
        if cues:
            projection['camera_composition'] = cues
            parts.append('camera composition: ' + ', '.join(cues))
    axis = camera.get('axis_change')
    if axis in {'hold', 'reestablish'}:
        projection['camera_axis'] = axis
        parts.append('preserve the established 180-degree screen axis' if axis == 'hold'
                     else 'visibly reestablish spatial geography before changing the screen axis')
    return tuple(parts), projection
