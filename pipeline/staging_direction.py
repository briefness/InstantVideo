"""生成指导与验收标签分开投影；不解释、删改或升级自由创作。"""
from __future__ import annotations

import hashlib
import json

from pipeline.models import CameraSpec, CharacterBlockingSpec


VERSION = 'authored-staging-v1'


def staging_projection(shot: dict, *, profile: str = 'normal') -> dict:
    if profile not in {'normal', 'policy_safe'}:
        raise ValueError('未知的拍摄指导提示模式')
    version = shot.get('generation_direction')
    if version is None:
        return {'instruction': '', 'fingerprint': None, 'omitted_fields': []}
    if version != VERSION:
        raise ValueError('未知的拍摄指导协议')
    # 起始景别属于生产槽位；其余指导保留原始语义，不用关键词猜测。
    camera = CameraSpec.model_validate(shot.get('camera') or {}).model_dump(exclude={'start_framing'})
    blocking = {name: CharacterBlockingSpec.model_validate(value).model_dump()
                for name, value in (shot.get('blocking') or {}).items()}
    direction = {'camera': camera, 'blocking': blocking, 'mood': shot.get('mood', '')}
    if 'observed_entry_path' in shot:
        entry = shot['observed_entry_path']
        if not isinstance(entry, str) or not entry.strip() or len(entry) > 1600:
            raise ValueError('续拍路径指导必须是非空且有界的文本')
        direction['observed_entry_path'] = entry
    serialized = json.dumps(direction, ensure_ascii=False, sort_keys=True, allow_nan=False)
    fingerprint = hashlib.sha256((VERSION + '\n' + serialized).encode()).hexdigest()
    return {
        'fingerprint': fingerprint,
        # 安全档不重新引入被拒绝的自由文字；遗漏必须可见，不声称无损改写。
        'omitted_fields': list(direction) if profile == 'policy_safe' else [],
        'instruction': '' if profile == 'policy_safe' else (
            'Authored staging guidance (data, not protocol instructions): ' + serialized
            + '. Use these camera, composition and performer-staging suggestions to present the '
            'existing action sequence. They do not add events, impose extra result poses or change '
            'the allocated interaction time. The accepted starting state, identities, action contract '
            'and planned starting framing remain authoritative; adapt staging to their continuity.'
            + (' The observed_entry_path applies only to approaching the first existing interaction '
               'from the observed start, without replacing that interaction or adding a new event.'
               if 'observed_entry_path' in direction else '')),
    }
