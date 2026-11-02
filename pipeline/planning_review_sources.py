"""从原稿绑定可读的后续证据；不推断依赖，也不生成或修正证据。"""
import re


def _text_fields(value: dict, names: tuple, prefix: str) -> dict:
    return {f'{prefix}/{name}': value[name] for name in names
            if isinstance(value.get(name), str) and value[name].strip()}


def _beat_fields(beat: dict, prefix: str) -> dict:
    fields = _text_fields(beat, ('action', 'visible_result', 'flow_from_previous'), prefix)
    for name in ('outcomes', 'scene_changes'):
        for index, result in enumerate(beat.get(name, [])):
            fields.update(_text_fields(result, ('state',), f'{prefix}/{name}/{index}'))
    return fields


def _shot_fields(shot: dict, prefix: str) -> dict:
    fields = {}
    for name, keys in (
        ('start_state', ('location', 'subject', 'action_phase', 'camera', 'screen_direction',
                         'pose_and_gaze', 'prop_state', 'open_motion', 'lighting')),
        ('narrative_beat', ('state_before', 'state_change', 'state_after')),
    ):
        fields.update(_text_fields(shot.get(name) or {}, keys, f'{prefix}/{name}'))
    for index, beat in enumerate(shot['action_beats']):
        fields.update(_beat_fields(beat, f'{prefix}/action_beats/{index}'))
    return fields


def continuity_sources(storyboard: dict) -> dict:
    shots = storyboard['shots']
    by_shot = [_shot_fields(shot, f'/shots/{i}') for i, shot in enumerate(shots)]
    next_shot = (storyboard.get('review_scope') or {}).get('next_shot')
    external = _shot_fields(next_shot, '/review_scope/next_shot') if next_shot else {}
    fields, allowed = {}, {}
    for i, shot in enumerate(shots):
        future = {path: value for later in by_shot[i + 1:] for path, value in later.items()}
        if i == len(shots) - 1:
            future.update(external)
        for j, _ in enumerate(shot['action_beats']):
            candidates = {}
            for k in range(j + 1, len(shot['action_beats'])):
                candidates.update(_beat_fields(shot['action_beats'][k], f'/shots/{i}/action_beats/{k}'))
            candidates.update(future)
            allowed[f"{shot['shot_id']}:{j}"] = list(candidates)
            fields.update(candidates)
    return {'fields': fields, 'allowed': allowed}


def review_evidence_sources(storyboard: dict) -> dict:
    """引用目录只定位原文，不把分句当作需求识别或语义判断。"""
    forward = continuity_sources(storyboard)
    fields, allowed = dict(forward['fields']), {}
    request = (storyboard.get('production_plan', {}).get('director_intent') or {}).get('source_request', '')
    if not isinstance(request, str):
        raise ValueError('审查依据缺少有效的用户原文')
    user = []
    for match in re.finditer(r'[^。！？\n]+[。！？\n]*|[。！？\n]+', request):
        text = match.group()
        # 保持现有解码后 evidence 的1600字上限；切分只作寻址，完整需求仍在请求中。
        for offset in range(0, len(text), 1600):
            excerpt = text[offset:offset + 1600].strip()
            if excerpt:
                path = f'/request/segments/{len(user)}'
                user.append(path)
                fields[path] = excerpt
    for i, shot in enumerate(storyboard['shots']):
        for j, beat in enumerate(shot['action_beats']):
            key = f"{shot['shot_id']}:{j}"
            current = _text_fields(beat, ('action', 'visible_result'), f'/shots/{i}/action_beats/{j}')
            fields.update(current)
            allowed[key] = {'user': list(user), 'continuity': forward['allowed'][key],
                            'response': list(current)}
    return {'fields': fields, 'allowed': allowed}
