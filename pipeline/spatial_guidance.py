"""任务相关的可视空间提示；不把示意图当首帧、事实或硬性控制。"""
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path

VERSION = 'grounded-diagram-v1'
INSTRUCTION = (
    'spatial_context contains the unchanged authored local work. Return spatial_guidance for up to four '
    'unambiguous spatial destinations/contact areas already required by that work, grounded in the actual '
    'last observed frame. Empty list is valid when there is no such work, the destination is occluded, '
    'the view is not groundable, or input mode cannot accept an extra reference image. Never invent a '
    'surface, a new relation or a path to satisfy the schema. entity is the existing moving/acting entity; '
    'target_entity is its distinct destination/support/contact entity, not a nearby landmark. '
    'source_box bounds the currently visible entity. target_region is four ordered convex vertices '
    'of the visible usable target surface/interior/contact area, not its whole object silhouette. '
    'Coordinates are normalized in the prior frame. Use an exact substring of that beat action or '
    'required states as requirement_quote. on means supported by a surface, in means inside a container, '
    'at means reaching a location, contact means physical contact. Do not use a gaze or resting pose as '
    'a new goal. The program will draw a separate geometric diagram, never retouch the original video. '
    'It is advisory visual guidance, not evidence that the generated action happened. Camera framing '
    'may change; maintain world relationships rather than copying these screen coordinates literally.'
)


def guidance_context(shot: dict) -> dict:
    from pipeline import spatial_observation
    if shot.get('spatial_control') == spatial_observation.VERSION:
        return spatial_observation.context(shot)
    from pipeline.action_results import required_outcomes
    beats = []
    entities = set((shot.get('camera') or {}).get('screen_positions') or {})
    for index, beat in enumerate(shot.get('action_beats', []), 1):
        states = required_outcomes(beat) if beat.get('outcomes') else []
        row = {k: deepcopy(beat.get(k, '')) for k in ('actor', 'target', 'action', 'start_seconds', 'end_seconds')}
        row.update(beat_id=index, required_states=states)
        beats.append(row)
        entities.update(x for x in (beat.get('actor'), beat.get('target')) if x)
        entities.update(x['entity'] for x in states)
    return {'version': VERSION, 'beats': beats, 'entities': sorted(entities),
            'camera': deepcopy(shot.get('camera') or {}),
            'available': shot.get('continuity_from_previous') == 'intentional_cut'}


def guidance_schema(context: dict) -> dict:
    from pipeline import spatial_observation
    if context['version'] == spatial_observation.VERSION:
        return spatial_observation.schema(context)
    number = {'type': 'number', 'minimum': 0, 'maximum': 1}
    point = {'type': 'array', 'items': number, 'minItems': 2, 'maxItems': 2}
    fields = {'beat_id': {'type': 'integer', 'minimum': 1, 'maximum': max(1, len(context['beats']))},
              'entity': {'type': 'string', 'enum': context['entities'] or ['']},
              'target_entity': {'type': 'string', 'enum': context['entities'] or ['']},
              'relation': {'type': 'string', 'enum': ['on', 'in', 'at', 'contact']},
              'source_box': {'type': 'array', 'items': number, 'minItems': 4, 'maxItems': 4},
              'target_region': {'type': 'array', 'items': point, 'minItems': 4, 'maxItems': 4},
              'requirement_quote': {'type': 'string', 'minLength': 1, 'maxLength': 160}}
    return {'type': 'array', 'maxItems': 4, 'items': {'type': 'object', 'properties': fields,
            'required': list(fields), 'additionalProperties': False}}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def bind_guidance(proposal: object, context: dict, source_sha256: str) -> dict:
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in source_sha256)):
        raise ValueError('空间示意缺少已观察画面的来源')
    from pipeline import spatial_observation
    if context['version'] == spatial_observation.VERSION:
        return spatial_observation.bind(proposal, context, source_sha256)
    if not isinstance(proposal, list) or len(proposal) > 4 or (proposal and not context['available']):
        raise ValueError('空间示意超出当前输入能力')
    used = set()
    fields = set(guidance_schema(context)['items']['properties'])
    for item in proposal:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError('空间示意结构无效')
        index = item['beat_id']
        if type(index) is not int or not 1 <= index <= len(context['beats']) or index in used:
            raise ValueError('空间示意动作来源重复或无效')
        used.add(index)
        beat = context['beats'][index - 1]
        if (any(not isinstance(item[k], str) for k in ('entity', 'target_entity', 'relation'))
                or item['entity'] not in {beat['actor'], beat['target']}
                or item['target_entity'] not in context['entities'] or item['entity'] == item['target_entity']
                or item['relation'] not in {'on', 'in', 'at', 'contact'}):
            raise ValueError('空间示意不能改变参与者或引入实体')
        quote = item['requirement_quote']
        sources = [beat['action'], *(s['state'] for s in beat['required_states'])]
        if not isinstance(quote, str) or not quote.strip() or len(quote) > 160 or not any(quote in s for s in sources):
            raise ValueError('空间示意没有对应的原始动作要求')
        box, points = item['source_box'], item['target_region']
        if (not isinstance(box, list) or len(box) != 4 or not isinstance(points, list) or len(points) != 4
                or any(not isinstance(p, list) or len(p) != 2 for p in points)):
            raise ValueError('空间示意坐标结构无效')
        numbers = [*box, *(v for p in points for v in p)]
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in numbers):
            raise ValueError('空间示意坐标必须位于观察画面内')
        if not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError('作用对象区域无效')
        turns = []
        for i in range(4):
            a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
            turns.append((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
        if not (all(t > 1e-8 for t in turns) or all(t < -1e-8 for t in turns)):
            raise ValueError('目标区域必须是可定位的非退化凸区域')
    return {'version': VERSION, 'source_sha256': source_sha256,
            'context_sha256': _digest(context), 'items': deepcopy(proposal)}


def checked_guidance(shot: dict) -> dict | None:
    value = shot.get('spatial_guidance')
    if value is None:
        return None
    from pipeline import spatial_observation
    marker = shot.get('spatial_control')
    if marker not in {VERSION, spatial_observation.VERSION} or not isinstance(value, dict):
        raise ValueError('空间示意协议不一致')
    proposal = (spatial_observation.response_from_bound(value)
                if marker == spatial_observation.VERSION else value.get('items'))
    expected = bind_guidance(proposal, guidance_context(shot), value.get('source_sha256'))
    if value != expected:
        raise ValueError('空间示意与当前动作或镜头来源不一致')
    origin = shot.get('observed_origin')
    if origin and origin['source_sha256'] != value['source_sha256']:
        raise ValueError('空间示意与观察起点不一致')
    return value


def active_guidance(shot: dict) -> dict | None:
    """可选观察图服从已选素材通道，不改变主摄影输入。"""
    from pipeline.spatial_observation import VERSION as LAYOUT_VERSION
    bound = checked_guidance(shot)
    view = shot.get('reference_view')
    if (bound and bound['version'] == LAYOUT_VERSION and view is not None
            and (view.get('execution_capability') or {}).get('mode') != 'original_video_guidance'):
        return None
    return bound


def reference_kind(shot: dict) -> str | bool:
    bound = active_guidance(shot)
    return bound['version'] if bound and bound['items'] else False


def _legacy_image_bytes(bound: dict) -> bytes:
    """已提交旧协议的字节必须稳定，不能以新解释覆盖历史素材。"""
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (960, 540), '#111827')
    draw = ImageDraw.Draw(image)
    draw.text((16, 10), 'SPATIAL GUIDE - reference coordinates, NOT a frame to copy', fill='white')
    count = len(bound['items'])
    for n, item in enumerate(bound['items']):
        width, height = 920 / count, 390
        left, top = 20 + n * width, 65
        # 归一化坐标使用方形坐标域，避免多项并排时把几何关系压扁。
        size = min(width - 15, height)
        xy = lambda p: (left + p[0] * size, top + p[1] * size)
        box = item['source_box']
        start = xy(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        polygon = [xy(p) for p in item['target_region']]
        end = (sum(p[0] for p in polygon) / 4, sum(p[1] for p in polygon) / 4)
        draw.rectangle([left, top, left + size, top + size], outline='#475569')
        draw.polygon(polygon, fill='#134e4a', outline='#5eead4', width=3)
        draw.rectangle([*xy(box[:2]), *xy(box[2:])], outline='#fbbf24', width=3)
        draw.line([start, end], fill='#f8fafc', width=3)
        draw.ellipse([end[0]-5, end[1]-5, end[0]+5, end[1]+5], fill='#5eead4')
        # 标签只使用编号；完整实体映射进入同一请求，避免字体/名称截断改变含义。
        draw.text((left, 42), f"Beat {item['beat_id']} / {item['relation'].upper()}", fill='white')
        draw.text((left, 465), f"{n+1}A: current entity box (yellow)", fill='#fbbf24')
        draw.text((left, 485), f"{n+1}B: target region (green)", fill='#5eead4')
    payload = io.BytesIO()
    image.save(payload, format='PNG')
    return payload.getvalue()


def materialize_guidance(shot: dict, root: Path) -> Path | None:
    """只画几何信息，无原片像素或人物；固定排版不约束成片镜头。"""
    from pipeline import spatial_observation
    bound = active_guidance(shot)
    if not bound or not bound['items']:
        return None
    data = (spatial_observation.image_bytes(bound) if bound['version'] == spatial_observation.VERSION
            else _legacy_image_bytes(bound))
    directory = Path(root) / 'spatial_guidance'
    if directory.is_symlink():
        raise ValueError('空间示意目录不能指向工作区外部')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (_digest(bound) + '.png')
    if path.is_symlink() or (path.exists() and path.read_bytes() != data):
        raise ValueError('空间示意缓存已变化；未覆盖或提交')
    if not path.exists():
        # 同一工作区由运行锁持有；排它创建避免覆盖任何已有内容。
        with path.open('xb') as handle:
            handle.write(data)
    return path


def append_guidance(shot: dict, refs: list[str], role: str | None, tail: str | None, root: Path):
    bound = active_guidance(shot)
    if not bound or not bound['items']:
        return refs, role
    if role != 'reference_image' or not tail or hashlib.sha256(Path(tail).read_bytes()).hexdigest() != bound['source_sha256']:
        raise ValueError('空间示意缺少可兼容的实际观察参考；未替换原素材')
    from pipeline.provider_materials import material_plan
    combined = [*refs, str(materialize_guidance(shot, root))]
    # 上游多个定位可能属于同一个原视频；上限按供应商实际素材清单核对。
    if material_plan(combined, role) is None and len(combined) > 9:
        raise ValueError('空间示意与现有参考超出图片上限；未删除参考或提交')
    return combined, role


def guidance_instruction(shot: dict) -> str:
    bound = active_guidance(shot)
    if not bound or not bound['items']:
        return ''
    from pipeline import spatial_observation
    if bound['version'] == spatial_observation.VERSION:
        return spatial_observation.instruction(bound)
    labels = '; '.join(f"{i}A={x['entity']}; {i}B={x['target_entity']} ({x['relation']})"
                       for i, x in enumerate(bound['items'], 1))
    return ('Use the attached SPATIAL GUIDE only to disambiguate the existing local action destination. '
            + labels + '. Yellow is the prior observed entity box; green is the intended target region, '
            'not an observed completed result. The connecting line associates endpoints, not a required trajectory. '
            'Keep appearance and scene from the original video and execute the current camera plan; '
            'do not render the diagram, labels or its layout in the output.')


def reference_clauses(count: int, role: str | None, has_state: bool, *, observed_layout: bool = False) -> list[str]:
    """几何图不属于身份参考，三个职责不能由额外图片数量混推。"""
    if role != 'reference_image' or type(count) is not int or count < 2:
        raise ValueError('空间示意缺少独立的参考位置')
    purpose = ('observed object locations only, not new goals, desired framing, identity or style'
               if observed_layout else 'geometric destination guidance only, not identity or style')
    clauses = [f'Image {count} is the SPATIAL GUIDE: ' + purpose]
    if has_state:
        clauses.append('Image 1 supplies the accepted prior physical state, not the new camera framing')
    first = 2 if has_state else 1
    for index in range(first, count):
        clauses.append(f'Image {index} supplies identity and appearance only, not action destination or camera framing')
    return clauses
