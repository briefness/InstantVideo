"""原始 Take 与本地呈现分离；确定性剪辑不是语义通过证明。

仅执行同轴固定机位的收紧取景。参考区域属于先前观察坐标，必须有实际
输出的同构图测量才能复用；其他情况保留原片交由原有审查判断。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, StrictFloat

VERSION = 'local-reframe-v1'


class ProtectedRegions(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True, allow_inf_nan=False)
    regions: dict[str, tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat]]


def solve_window(proposal: object, context: dict) -> dict | None:
    """固定长宽比的最小包围窗；无空间解就返回空，不修剪保护区。"""
    from pipeline.reference_view import validate_view
    regions = ProtectedRegions.model_validate(proposal).model_dump(mode='json')['regions']
    if not regions or set(regions) != set(context['region_entities']):
        raise ValueError('取景区域缺少必要实体')
    for x1, y1, x2, y2 in regions.values():
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError('取景区域无效')
    left = min(b[0] for b in regions.values())
    top = min(b[1] for b in regions.values())
    right = max(b[2] for b in regions.values())
    bottom = max(b[3] for b in regions.values())
    scale = max(right - left, bottom - top)
    if scale >= 1:
        return None
    return validate_view({'left': max(0., min(left, 1 - scale, (left + right - scale) / 2)),
        'top': max(0., min(top, 1 - scale, (top + bottom - scale) / 2)),
        'scale': scale, 'regions': regions}, context)


def _hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _identity(video: str, shot: dict, root: Path) -> tuple[dict, Path]:
    identity = {'version': VERSION, 'source_sha256': _hash(Path(video)),
        'camera': shot.get('camera'), 'shot_id': shot.get('shot_id'),
        'continuity': shot.get('continuity_from_previous')}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return identity, Path(root) / 'presentations' / f'{key}.json'


def _record(video: str, shot: dict, root: Path) -> tuple[dict, Path]:
    identity, path = _identity(video, shot, root)
    try:
        if path.is_symlink() or path.stat().st_size > 100_000:
            raise ValueError('呈现记录超出容量')
        record = json.loads(path.read_text())
        output = path.with_suffix('.mp4') if record['mode'] == 'crop' else Path(video)
        box = record['source_box']
        if (output.is_symlink() or record['identity'] != identity or record['mode'] not in {'crop', 'native'}
                or not isinstance(box, list) or len(box) != 4
                or not all(type(v) in (int, float) and math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1
                or (record['mode'] == 'native' and box != [0., 0., 1., 1.])
                or _hash(output) != record['output_sha256']):
            raise ValueError('呈现记录与实际素材不一致')
        return record, output
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError('呈现素材缺失或已变化；不能改用未经审查的原片') from exc


def resolve_presentation(video: str, shot: dict | None, root: Path) -> str:
    if not shot or shot.get('presentation_policy') != VERSION:
        return video
    return str(_record(video, shot, root)[1])


def source_box(video: str, shot: dict | None, root: Path, box=None) -> list | None:
    """将呈现画面的观察区域映回官方原件，不能冒充新的官方产物。"""
    if not shot or shot.get('presentation_policy') != VERSION:
        return box
    rect = _record(video, shot, root)[0]['source_box']
    if box is None:
        return None if rect == [0., 0., 1., 1.] else rect
    x, y, r, b = rect
    return [x + box[0] * (r - x), y + box[1] * (b - y),
            x + box[2] * (r - x), y + box[3] * (b - y)]


def ensure_presentation(video: str, shot: dict, previous_frame: str | None, root: Path) -> str:
    if shot.get('presentation_policy') != VERSION:
        return video
    from pipeline.camera_contract import compile_camera
    from pipeline.reference_view import validate_view, VERSION as VIEW_VERSION
    from pipeline.run_state import RunWorkspace
    from tools.camera_measurement import measure_video_composition
    from tools.ffmpeg_ops import get_video_info

    spec = shot.get('reference_view')
    if spec and previous_frame and spec.get('source_sha256') != _hash(Path(previous_frame)):
        raise ValueError('呈现取景的观察来源已变化；未复用缓存')
    identity, receipt = _identity(video, shot, root)
    if receipt.exists():
        record, _ = _record(video, shot, root)
        if record.get('reference_view') != shot.get('reference_view'):
            raise ValueError('呈现取景提案已变化；未重新剪辑')
        return resolve_presentation(video, shot, root)
    output = receipt.with_suffix('.mp4')
    if output.exists():
        raise ValueError('呈现缓存不完整；未自动重做剪辑')
    record = {'identity': identity, 'mode': 'native', 'reason': 'unsupported_or_unproposed',
              'reference_view': shot.get('reference_view'),
              'source_box': [0., 0., 1., 1.], 'output_sha256': identity['source_sha256']}
    camera = compile_camera(shot)[1]
    if (spec and previous_frame and shot.get('continuity_from_previous') == 'intentional_cut'
            and camera.get('camera_movement') == 'fixed' and camera.get('camera_axis') == 'hold'
            and not shot.get('motion_plan') and not shot.get('motion_reference')):
        if (spec.get('version') != VIEW_VERSION
                or spec['source_sha256'] != _hash(Path(previous_frame)) or spec['camera'] != camera
                or spec['camera_plan'] != shot.get('camera')):
            raise ValueError('呈现取景与已观察来源或摄影合同不一致')
        window = validate_view(spec['window'], spec)
        if window is not None:
            measured = measure_video_composition(previous_frame, video, 0.)
            record.update(measurement=measured, reason='source_coordinates_not_verified')
            if measured['status'] == 'unchanged':
                info = get_video_info(video)
                stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
                width, height = stream['width'], stream['height']
                # 向外取整保证不丢掉保护区域，最多增加一个色度采样单元。
                x, y = (math.floor(window[k] * n / 2) * 2
                        for k, n in [('left', width), ('top', height)])
                right = min(width, math.ceil((window['left'] + window['scale']) * width / 2) * 2)
                bottom = min(height, math.ceil((window['top'] + window['scale']) * height / 2) * 2)
                receipt.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(f'{output.stem}.{uuid4().hex}.mp4')
                try:
                    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', video,
                        '-map', '0:v:0', '-map', '0:a?', '-vf',
                        f'crop={right-x}:{bottom-y}:{x}:{y},scale={width}:{height},setsar=1',
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                        '-c:a', 'copy', str(temporary)], check=True, capture_output=True, timeout=120)
                    temporary.replace(output)
                finally:
                    temporary.unlink(missing_ok=True)
                record.update(mode='crop', reason='same_axis_constrained_crop',
                    source_box=[x/width, y/height, right/width, bottom/height], output_sha256=_hash(output))
    receipt.parent.mkdir(parents=True, exist_ok=True)
    RunWorkspace._atomic_write_json(receipt, record)
    return resolve_presentation(video, shot, root)
