"""已编译动作的语义边界；旧格式的修复不属于此阶段。"""
from copy import deepcopy

from pipeline.models import (
    ActionBeatSpec, InteractionGeometrySpec, ContinuityStateSpec,
    NarrativeBeatSpec, CharacterBlockingSpec, CameraSpec,
)


def authored_semantics(shot: dict) -> bool:
    return shot.get('action_semantics') == 'authored-v1'


def semantic_snapshot(storyboard: dict) -> dict:
    """按类型补齐默认值，比较显式语义而非 JSON 序列化差异。"""
    result = {}
    for index, shot in enumerate(storyboard['shots']):
        if not authored_semantics(shot):
            continue
        snapshot = {key: deepcopy(shot.get(key)) for key in (
            'action_semantics', 'primary_action', 'scene_id', 'duration',
            'characters', 'required_visible_entities',
        )}
        snapshot['action_beats'] = [ActionBeatSpec.model_validate(beat).model_dump(exclude_none=True)
                                    for beat in shot['action_beats']]
        for key, model in (
            ('interaction_geometry', InteractionGeometrySpec),
            ('start_state', ContinuityStateSpec), ('end_state', ContinuityStateSpec),
            ('narrative_beat', NarrativeBeatSpec), ('camera', CameraSpec),
        ):
            snapshot[key] = model.model_validate(shot.get(key) or {}).model_dump(exclude_none=True)
        snapshot['blocking'] = {key: CharacterBlockingSpec.model_validate(value).model_dump(exclude_none=True)
                                for key, value in shot.get('blocking', {}).items()}
        result[index] = snapshot
    return result


def assert_semantics_preserved(before: dict, storyboard: dict) -> None:
    after = semantic_snapshot(storyboard)
    if before == after:
        return
    differences = [f'shots[{index}].{key}' for index in before.keys() | after.keys()
                   for key in before.get(index, {}).keys() | after.get(index, {}).keys()
                   if before.get(index, {}).get(key) != after.get(index, {}).get(key)]
    raise ValueError('编译阶段改写已确定的动作语义: ' + ', '.join(differences))
