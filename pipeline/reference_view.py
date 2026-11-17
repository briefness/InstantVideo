"""把已观察的场景状态、新取景意图与供应商执行能力分开。

不生成看不见的内容，不改变原片。可直接使用的派生图能约束首帧；需要官方
原件的通道仅携带原视频和视窗定位。原始尾帧继续用于接缝审查，来源不可替换。
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictFloat

from pipeline.camera_contract import compile_camera
from pipeline.production_plan import classify_framing


VERSION = 'observed-camera-view-v1'
SELECTION_VERSION = 'optional-camera-view-v1'
_VIEW_INSTRUCTION_TEMPLATE = (
    'The supplied image is the actual accepted ending frame. camera_view requests a tighter '
    'initial view from that same world state. Return view_window=null if this image cannot supply '
    'the planned camera view without inventing unseen content, changing the viewpoint, or losing '
    'a required participant, explicit method, prop or working area. Otherwise choose a normalized '
    'left, top, scale window; its width and height are each scale times the source dimensions. '
    'regions must bind every listed entity to its actually visible necessary region in this image '
    '({region_scope}). '
    'All regions must fit inside the window. Do not use tiny fragments to claim a subject is retained. '
    'The program will crop these exact pixels as the next first frame; it will not retouch people, '
    'move objects or fill missing background. Keep the planned events, camera and outcomes unchanged. '
    'This is an initial camera setup, not evidence that the future video passes review.'
)

SELECTION_INSTRUCTION = (
    'Camera view is an optional execution proposal, separate from required continuity compatibility. '
    'A missing or unusable proposal does not mean the story cannot continue: the program retains '
    'the original actions and camera requirement on its native reference route. '
    'Use view_window=null when no useful tighter view is available. When proposing regions, follow '
    'the planned framing: identify the visible subject parts and working area actually needed there. '
    'Maintaining an identity does not require its face or full body in every shot; a detail shot may '
    'show only the relevant hands or object unless the user explicitly requires more. '
    'Never remove required entities, methods or working areas just to fit a window. '
    'Do not change compatible to false merely because this optional crop cannot be supplied.'
)
INSTRUCTION = _VIEW_INSTRUCTION_TEMPLATE.format(
    region_scope='recognizable person plus the hands/work area needed for the next action, or the relevant object')
OPTIONAL_INSTRUCTION = _VIEW_INSTRUCTION_TEMPLATE.format(
    region_scope='the visible subject parts and working area needed for this planned framing') + ' ' + SELECTION_INSTRUCTION


class ViewWindow(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True, allow_inf_nan=False)
    left: float = Field(strict=True, ge=0, le=1)
    top: float = Field(strict=True, ge=0, le=1)
    scale: float = Field(strict=True, gt=0, lt=1)
    regions: dict[str, tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat]]


def _digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _image(path: Path):
    import cv2
    if not path.is_file() or path.stat().st_size > 20_000_000:
        raise ValueError('取景来源缺失或超出图像容量')
    image = cv2.imread(str(path))
    if image is None or image.size == 0 or image.shape[0] * image.shape[1] > 40_000_000:
        raise ValueError('取景来源不是可用的有界图像')
    return image


def camera_view_context(shot: dict, previous: dict, tail: Path, *, legacy: bool = False) -> dict | None:
    if (shot.get('completion_policy') != 'goal-allocation-v1'
            or shot.get('continuity_from_previous') != 'intentional_cut'
            or shot.get('motion_reference') or shot.get('motion_plan')):
        return None
    camera = compile_camera(shot)[1]
    if camera.get('camera_axis') != 'hold':
        return None
    target = classify_framing((shot.get('camera') or {}).get('start_framing', ''))
    if legacy:
        # 历史提交的上下文和指纹必须按原协议重建。
        rank = {'wide': 0, 'medium': 1, 'close_detail': 2}
        source = classify_framing((previous.get('observed_end_state') or {}).get('camera', ''))
        if source not in rank or target not in rank or rank[target] <= rank[source]:
            return None
    elif target not in {'medium', 'close_detail'}:
        return None
    if any(c in camera.get('camera_composition', []) for c in ('low angle', 'high angle', 'over the shoulder', 'full body')):
        return None
    entities = list(dict.fromkeys([*shot.get('characters', []), *shot.get('required_visible_entities', [])]))
    if not entities:
        return None
    image = _image(tail)
    context = {'version': VERSION, 'camera': camera, 'camera_plan': deepcopy(shot['camera']), 'region_entities': entities,
               'source_size': [image.shape[1], image.shape[0]]}
    if not legacy:
        from pipeline.provider_materials import view_execution_capability
        # 摄影可达性由实际像素提案判断，不把自由文字的景别解析当作视觉事实。
        context['execution_capability'] = view_execution_capability(str(tail))
        if shot.get('presentation_policy') == 'local-reframe-v1':
            context['presentation_policy'] = shot['presentation_policy']
    return context


def view_instruction(context: dict) -> str:
    """同一取景意图按实际素材通道描述，不向编导承诺不存在的像素控制。"""
    if context.get('presentation_policy') == 'local-reframe-v1':
        return (
            'Inspect the supplied image against camera_plan. Return view_window=null if the '
            'planned view needs another angle or unseen content. Otherwise return only regions: '
            'one normalized [left,top,right,bottom] box per region_entity, protecting the visible '
            'subject parts and working area necessary for the planned framing. Do not choose '
            'crop coordinates or discard required participants. The program solves a containing '
            'window and may crop the generated shot locally only if its actual coordinates align. '
            'This does not prove the resulting framing or action is correct; the actual edited '
            'video must pass the unchanged review. ' + SELECTION_INSTRUCTION)
    capability = context.get('execution_capability')
    if capability is None or capability['mode'] == 'first_frame_pixels':
        return OPTIONAL_INSTRUCTION
    if capability['mode'] != 'original_video_guidance':
        raise ValueError('未知的取景执行能力')
    return (
        'The supplied image is an observation of the accepted scene state. Inspect its actual pixels '
        'against camera_plan; the observed camera prose is not a measured framing category. '
        'Propose left, top and scale in the source image normalized coordinate system only when a '
        'tighter view can represent the planned initial composition. Width and height both use scale. '
        'Bind every region_entity to the visible subject parts, objects and working area needed by '
        'that framing, all inside the window. Return view_window=null when the view requires unseen '
        'content, a different viewpoint, or loss of necessary action space. '
        'Execution uses the complete official original video plus a timestamp and this view window '
        'as reference guidance. No cropped pixels are sent as a first frame, and this proposal cannot '
        'guarantee the generated framing. The actual output still has to meet camera_plan. '
        + SELECTION_INSTRUCTION
    )


def view_schema(context: dict) -> dict:
    region = {'type': 'array', 'items': {'type': 'number', 'minimum': 0, 'maximum': 1},
              'minItems': 4, 'maxItems': 4}
    if context.get('presentation_policy') == 'local-reframe-v1':
        return {'anyOf': [{'type': 'null'}, {'type': 'object', 'properties': {
            'regions': {'type': 'object', 'properties': {e: region for e in context['region_entities']},
                'required': context['region_entities'], 'additionalProperties': False}},
            'required': ['regions'], 'additionalProperties': False}]}
    return {'anyOf': [{'type': 'null'}, {'type': 'object', 'properties': {
        'left': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'top': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'scale': {'type': 'number', 'exclusiveMinimum': 0, 'exclusiveMaximum': 1},
        'regions': {'type': 'object', 'properties': {e: region for e in context['region_entities']},
                    'required': context['region_entities'], 'additionalProperties': False}},
        'required': ['left', 'top', 'scale', 'regions'], 'additionalProperties': False}]}


def validate_view(value: object, context: dict) -> dict | None:
    if value is None:
        return None
    import math
    window = ViewWindow.model_validate(value).model_dump(mode='json')
    left, top, scale = (window[k] for k in ('left', 'top', 'scale'))
    if left + scale > 1 + 1e-9 or top + scale > 1 + 1e-9:
        raise ValueError('取景窗口超出实际画面')
    if set(window['regions']) != set(context['region_entities']):
        raise ValueError('取景缺少必要人物或对象的画面依据')
    for box in window['regions'].values():
        x1, y1, x2, y2 = box
        if (not all(math.isfinite(v) for v in box)
                or not left <= x1 < x2 <= left + scale + 1e-9
                or not top <= y1 < y2 <= top + scale + 1e-9):
            raise ValueError('必要人物、动作区域或对象被取景裁掉')
    if min(context['source_size']) * scale < 64:
        raise ValueError('取景后的有效像素不足')
    return window


def bind_view(window: dict, context: dict, source_hash: str) -> dict:
    return {**context, 'source_sha256': source_hash, 'window': validate_view(window, context)}


def select_view(proposal: object, context: dict) -> dict:
    """验证提案以决定能力采用，不把提案改写为有效值或改写必要摄影要求。"""
    from pydantic import ValidationError
    decision = {'version': SELECTION_VERSION, 'mode': 'native_reference',
                'reason': 'not_proposed', 'window': None, 'issues': [],
                'proposal_sha256': hashlib.sha256(json.dumps(
                    proposal, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    try:
        if context.get('presentation_policy') == 'local-reframe-v1' and proposal is not None:
            from pipeline.presentation_media import solve_window
            window = solve_window(proposal, context)
        else:
            window = validate_view(proposal, context)
    except ValidationError as exc:
        decision.update(reason='invalid_proposal', issues=[
            {'field': '/'.join(map(str, error['loc'])), 'code': error['type']}
            for error in exc.errors(include_input=False, include_url=False, include_context=False)])
    except (ValueError, TypeError, KeyError):
        # 只有无副作用的提案几何校验可以放弃；源素材/缓存/IO错误不得走此分支。
        decision.update(reason='invalid_proposal', issues=[{'field': 'window', 'code': 'unusable_geometry'}])
    else:
        if window is not None:
            mode = ('original_video_guidance' if context.get('execution_capability', {}).get('mode')
                    == 'original_video_guidance' else 'derived_frame')
            decision.update(mode=mode, reason='usable_proposal', window=window)
    return decision


def materialize_view(shot: dict, tail: str, root: Path) -> str:
    import cv2
    spec = shot['reference_view']
    source = Path(tail)
    if (spec.get('version') != VERSION or spec.get('source_sha256') != _digest(source)
            or spec.get('camera') != compile_camera(shot)[1]
            or spec.get('camera_plan') != shot.get('camera')):
        raise ValueError('取景计划与已观察来源或当前摄影合同不一致')
    if 'execution_capability' in spec:
        from pipeline.provider_materials import view_execution_capability
        if spec['execution_capability'] != view_execution_capability(tail):
            raise ValueError('素材执行能力与取景计划不一致；未改变参考通道')
    image = _image(source)
    if spec['source_size'] != [image.shape[1], image.shape[0]]:
        raise ValueError('取景来源尺寸发生变化')
    window = validate_view(spec['window'], spec)
    if window is None:
        raise ValueError('未提供可执行取景窗口')
    key = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    directory = Path(root) / 'reference_views'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{key}.jpg'
    receipt = path.with_suffix('.json')
    if path.exists() or receipt.exists():
        if not matches_view(shot, tail, str(path)):
            raise ValueError('派生取景缓存不完整或已变化；未重新生成参考')
        from pipeline.provider_materials import bind_view
        bind_view(str(path), tail, window)
        return str(path)
    height, width = image.shape[:2]
    x, y = round(window['left'] * width), round(window['top'] * height)
    w, h = round(window['scale'] * width), round(window['scale'] * height)
    crop = image[y:y+h, x:x+w]
    if crop.shape[:2] != (h, w):
        raise ValueError('实际取景像素越界')
    temporary = directory / f'{key}.{uuid4().hex}.jpg'
    try:
        if not cv2.imwrite(str(temporary), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise ValueError('无法写入取景参考')
        from pipeline.run_state import RunWorkspace
        temporary.replace(path)
        RunWorkspace._atomic_write_json(receipt, {'spec': spec, 'output_sha256': _digest(path),
                                                   'source_sha256': spec['source_sha256']})
    finally:
        temporary.unlink(missing_ok=True)
    from pipeline.provider_materials import bind_view
    bind_view(str(path), tail, window)
    return str(path)


def matches_view(shot: dict, tail: str, reference: str) -> bool:
    spec = shot.get('reference_view')
    if not spec:
        return False
    try:
        path = Path(reference)
        record = json.loads(path.with_suffix('.json').read_text())
        return (record.get('spec') == spec and spec['source_sha256'] == _digest(Path(tail))
                and record['output_sha256'] == _digest(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False
