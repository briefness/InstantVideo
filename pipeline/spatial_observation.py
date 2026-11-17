"""空间观察是只读输入；不允许定位器重新创造动作关系。"""
from copy import deepcopy
import hashlib
import io
import json
import math
import re

VERSION = 'observed-layout-v1'
INSTRUCTION = (
    'spatial_context is a read-only localization task. Return spatial_guidance as an object '
    'with exactly the supplied entity IDs as keys. Each value is null if the entity is not '
    'visibly localizable in the last observed frame, otherwise four ordered convex vertices '
    'of its visible extent in normalized source-frame coordinates. A partially visible '
    'object may use only its visible area; never substitute its actor, holder or nearby object. '
    'The authored work is context, not evidence that its future positions already exist. '
    'Do not return relations, action destinations, action prose or a new camera plan. '
    'The resulting map only labels observed objects; it cannot redefine authored work. '
    'Return an empty object when there are no requested entities.'
)


def context(shot: dict) -> dict:
    registry = set((shot.get('camera') or {}).get('screen_positions') or {})
    registry.update(shot.get('key_props') or [])
    registry.update(shot.get('continuity_props') or [])
    registry.update(shot.get('characters') or [])
    entities, work = set(), []
    for index, beat in enumerate(shot.get('action_beats', []), 1):
        intent = beat.get('completion_intent') or {}
        # 独立审定的本镜工作拥有用途；表演文字不能自己产生空间控制任务。
        if intent.get('kind') != 'allocated' or not intent.get('requirements'):
            continue
        focus = beat.get('target') or beat.get('actor')
        if focus:
            entities.add(focus)
        region = (beat.get('interaction_geometry') or {}).get('effect_region', '')
        for name in registry:
            if re.search(r'(?<![A-Za-z0-9_])' + re.escape(name) + r'(?![A-Za-z0-9_])', region):
                entities.add(name)
        work.append({'beat_id': index, 'actor': beat.get('actor'), 'target': beat.get('target'),
                     'action': beat.get('action'), 'action_region': region,
                     'requirements': deepcopy(intent['requirements'])})
    available = shot.get('continuity_from_previous') == 'intentional_cut' and len(entities) <= 8
    return {'version': VERSION, 'work': work, 'entities': sorted(entities) if available else [],
            'available': available, 'camera': deepcopy(shot.get('camera') or {}),
            'coordinate_scope': 'prior_observation_only'}


def schema(value: dict) -> dict:
    point = {'type': 'array', 'items': {'type': 'number', 'minimum': 0, 'maximum': 1},
             'minItems': 2, 'maxItems': 2}
    region = {'anyOf': [{'type': 'null'}, {'type': 'array', 'items': point,
                                        'minItems': 4, 'maxItems': 4}]}
    return {'type': 'object', 'properties': {name: deepcopy(region) for name in value['entities']},
            'required': value['entities'], 'additionalProperties': False}


def bind(proposal: object, value: dict, source_sha256: str) -> dict:
    if not isinstance(proposal, dict) or set(proposal) != set(value['entities']):
        raise ValueError('空间观察只能定位已分配对象，不能返回新的动作或关系')
    items, unlocated = [], []
    for entity in value['entities']:
        points = proposal[entity]
        if points is None:
            unlocated.append(entity)
            continue
        if (not isinstance(points, list) or len(points) != 4
                or any(not isinstance(p, list) or len(p) != 2 for p in points)):
            raise ValueError('空间观察必须是原画面区域或明确不可定位')
        if any(type(n) not in (int, float) or not math.isfinite(n) or not 0 <= n <= 1
               for p in points for n in p):
            raise ValueError('空间观察坐标超出来源画面')
        turns = []
        for i in range(4):
            a, b, c = points[i], points[(i+1) % 4], points[(i+2) % 4]
            turns.append((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
        if not (all(t > 1e-8 for t in turns) or all(t < -1e-8 for t in turns)):
            raise ValueError('空间观察区域退化或交叉')
        items.append({'entity': entity, 'region': deepcopy(points)})
    digest = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {'version': VERSION, 'source_sha256': source_sha256, 'context_sha256': digest,
            'items': items, 'unlocated': unlocated}


def response_from_bound(bound: dict) -> dict:
    if not isinstance(bound.get('items'), list) or not isinstance(bound.get('unlocated'), list):
        raise ValueError('空间观察缓存结构无效')
    result = {}
    for item in bound['items']:
        if not isinstance(item, dict) or set(item) != {'entity', 'region'} or not isinstance(item['entity'], str):
            raise ValueError('空间观察缓存包含非观察字段')
        if item['entity'] in result:
            raise ValueError('空间观察对象重复')
        result[item['entity']] = item['region']
    for entity in bound['unlocated']:
        if not isinstance(entity, str) or entity in result:
            raise ValueError('空间观察可见性自相矛盾')
        result[entity] = None
    return result


def image_bytes(bound: dict) -> bytes:
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (960, 540), '#111827')
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), 'SPATIAL GUIDE - OBSERVED OBJECT MAP, not desired state or camera framing', fill='white')
    draw.rectangle((24, 65, 440, 481), outline='#475569')
    for index, item in enumerate(bound['items'], 1):
        polygon = [(24 + x * 416, 65 + y * 416) for x, y in item['region']]
        draw.polygon(polygon, outline='#5eead4', width=3)
        point = (sum(x for x, _ in polygon)/4, sum(y for _, y in polygon)/4)
        draw.text(point, str(index), fill='white')
        draw.text((472, 75 + (index-1)*40), f'{index}: observed region (entity mapping in request)', fill='white')
    draw.text((24, 505), 'Normalized source coordinates. No arrows, destinations or future poses are encoded.', fill='white')
    payload = io.BytesIO()
    image.save(payload, format='PNG')
    return payload.getvalue()


def instruction(bound: dict) -> str:
    labels = '; '.join(f"{i}={x['entity']}" for i, x in enumerate(bound['items'], 1))
    missing = ', '.join(bound['unlocated'])
    return ('The SPATIAL GUIDE is only an observed object map in the prior frame: ' + labels
            + '. It defines no new destination, action, trajectory or composition. '
            + (f'Not localized in that observation: {missing}; do not substitute another object. ' if missing else '')
            + 'Execute the unchanged authored work and current camera plan using the original video for appearance. '
              'Do not render this diagram or copy its layout into the film.')
