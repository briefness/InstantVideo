"""故事登记身份，逐镜声明用途；投影到已有创作合同，不重写创作。"""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SceneLocation(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, hide_input_in_errors=True)
    id: str = Field(min_length=1, max_length=128, pattern=r'^\S+$')
    description: str = Field(min_length=1, max_length=1600)


class PropRole(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, hide_input_in_errors=True)
    entity: str = Field(min_length=1, max_length=128)
    role: Literal['visible', 'persistent', 'visible_and_persistent']


def contextual_shot_schema(schema: dict, locations: list[SceneLocation], entities: list[str]) -> dict:
    result = deepcopy(schema)
    shot = result['properties']['shot']
    # 与动作参与者共用已冻结的目录；对象位置和角色站位不是可自由命名的区域表。
    names = result['$defs']['PassiveUnit']['properties']['actor']['enum']
    for mapping, owners, description in (
        (result['$defs']['CameraSpec']['properties']['screen_positions'], [*names, *entities],
         '实体ID到画面位置的映射；区域、路径和构图描述写在值或composition中。'),
        (shot['properties']['blocking'], names, '角色ID到起始站位的映射。'),
    ):
        value_schema = mapping['additionalProperties']
        mapping.update(properties={name: deepcopy(value_schema) for name in owners},
                       additionalProperties=False, description=description)
    shot['properties']['scene_id'] = {'type': 'string', 'enum': [s.id for s in locations]}
    for name in ('key_props', 'continuity_props'):
        shot['properties'].pop(name)
        if name in shot['required']:
            shot['required'].remove(name)
    role = PropRole.model_json_schema()
    role['properties']['entity']['enum'] = entities
    shot['properties']['prop_roles'] = {'type': 'array', 'maxItems': 40, 'items': role}
    # 没有可用对象时不能构造空 enum；明确要求空列表，局部校验同样拒绝任意引用。
    if not entities:
        role['properties']['entity'].pop('enum')
        shot['properties']['prop_roles']['maxItems'] = 0
    shot['required'].append('prop_roles')
    # 只发送本协议可达的定义，避免旧合同的字段说明混入新创作任务。
    definitions = result.pop('$defs')
    used = {}
    def visit(node):
        if isinstance(node, dict):
            ref = node.get('$ref')
            if isinstance(ref, str) and ref.startswith('#/$defs/'):
                name = ref.removeprefix('#/$defs/')
                if name not in used:
                    used[name] = definitions[name]
                    visit(used[name])
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(result)
    result['$defs'] = used
    return result


def bind_authored_context(raw: dict, locations: list[SceneLocation], entities: list[str]) -> dict:
    if raw.get('scene_id') not in {s.id for s in locations}:
        raise ValueError('镜头必须引用故事已登记的物理地点，不能逐镜重新命名')
    if 'key_props' in raw or 'continuity_props' in raw:
        raise ValueError('新创作只声明 prop_roles，不能同时返回程序投影的两份道具列表')
    values = raw.get('prop_roles')
    if not isinstance(values, list) or len(values) > 40:
        raise ValueError('每镜必须给出有界的 prop_roles 列表')
    roles = [PropRole.model_validate(value) for value in values]
    ids = [r.entity for r in roles]
    if len(ids) != len(set(ids)) or any(name not in entities for name in ids):
        raise ValueError('道具用途必须引用已登记对象，每个对象只能声明一次')
    result = deepcopy(raw)
    result.pop('prop_roles')
    result['key_props'] = [r.entity for r in roles if r.role != 'persistent']
    result['continuity_props'] = [r.entity for r in roles if r.role != 'visible']
    return result
