"""由现有槽位绑定必需作用，降为单一动作列表；不补写或重排创作。"""
from copy import deepcopy
import math

from pipeline.action_authoring import AuthoredShot, InteractionUnit, PassiveUnit
from pipeline.action_timing import RESPONSE_MODES
from pipeline.action_results import CandidateResolution, entity_resolution_schema
from pipeline.camera_contract import SUPPORTED_CAMERA_MOVEMENTS, camera_contract_issues
from pipeline.production_plan import CANONICAL_FRAMING
from pipeline.action_vocabulary import INDEPENDENT_ACTIVITIES


def _layout(slot: dict) -> tuple[list[str], bool, int]:
    limit = next(item.max_length for item in AuthoredShot.model_fields['action_plan'].metadata
                 if hasattr(item, 'max_length'))
    scene_units = isinstance(slot, dict) and slot.get('authoring_protocol') == 'scene-units-v1'
    count = slot.get('min_interaction_events', 0 if scene_units else None) if isinstance(slot, dict) else None
    hold = slot.get('result_hold_seconds', 0) if isinstance(slot, dict) else None
    if (type(count) is not int or not (0 if scene_units else 1) <= count <= limit
            or type(hold) not in (int, float) or not math.isfinite(hold) or hold < 0):
        raise ValueError('动作创作槽位缺少有效的必需交互数或结果停留预算')
    terminal = hold > 0
    if count + int(terminal) > limit:
        raise ValueError('槽位的必需作用与结果展示超出已有动作单元容量')
    return [f'interaction_{i + 1}' for i in range(count)], terminal, limit


def _without_fields(schema: dict, names: tuple[str, ...]) -> dict:
    result = deepcopy(schema)
    if 'oneOf' in result:
        result['oneOf'] = [_without_fields(item, names) for item in result['oneOf']]
    else:
        for name in names:
            result['properties'].pop(name, None)
        result['required'] = [name for name in result.get('required', []) if name not in names]
    return result


def scheduled_shot_schema(base_schema: dict, slot: dict, *, explicit_absence: bool = False) -> dict:
    """必填键来自计划；总展开容量仍由本地检查，避免组合枚举的巨大schema。"""
    keys, terminal, limit = _layout(slot)
    result = deepcopy(base_schema)
    definitions = result['$defs']
    entities = definitions['PassiveUnit']['properties']['target'].get('enum')
    if not entities:
        raise ValueError('新动作创作需要冻结的实体目录')
    resolution = entity_resolution_schema([entity for entity in entities if entity])
    definitions.update(resolution.pop('$defs'))
    definitions['EntityResolution'] = resolution
    for variant in definitions['InteractionUnit']['oneOf']:
        variant['properties']['resolution'] = {'$ref': '#/$defs/EntityResolution'}
    # 创作选项直接来自执行器能力；旧 CameraSpec 继续接纳历史别名。
    definitions['CameraSpec']['properties']['primary_movement']['enum'] = list(SUPPORTED_CAMERA_MOVEMENTS)
    framing = CANONICAL_FRAMING.get(slot.get('framing_family'))
    if framing:
        definitions['CameraSpec']['properties']['start_framing'] = {'type': 'string', 'const': framing}
    passive = _without_fields(definitions['PassiveUnit'], ('id',))
    following_interaction = _without_fields(definitions['InteractionUnit'], ('id',))
    optional_capacity = limit - len(keys) - int(terminal)
    definitions['ScheduledPassiveUnit'] = passive
    definitions['ScheduledFollowingUnit'] = {'oneOf': [
        following_interaction, {'$ref': '#/$defs/ScheduledPassiveUnit'}]}
    has_character = bool(definitions['InteractionUnit']['oneOf'][0]['properties']['actor']['enum'])
    if keys and not has_character:
        raise ValueError('既定交互任务缺少已登记的执行人物，无法创建有效的创作请求')
    if not has_character:
        definitions['ScheduledFollowingUnit'] = {'$ref': '#/$defs/ScheduledPassiveUnit'}
    following = {'type': 'array', 'maxItems': optional_capacity,
                 'items': {'$ref': '#/$defs/ScheduledFollowingUnit'}}
    interaction = _without_fields(definitions['InteractionUnit'], ('id', 'kind'))
    for variant in interaction['oneOf']:
        variant['properties']['following'] = deepcopy(following)
    definitions['ScheduledRequiredInteraction'] = interaction
    first_interaction = deepcopy(interaction)
    for variant in first_interaction['oneOf']:
        variant['properties']['overlap'] = {'type': 'boolean', 'const': False}
    definitions['ScheduledAfterOpeningInteraction'] = deepcopy(first_interaction)
    for variant in first_interaction['oneOf']:
        variant['properties']['link_previous'] = {'type': 'boolean', 'const': False}
    definitions['ScheduledFirstInteraction'] = first_interaction
    properties = {key: {'$ref': '#/$defs/ScheduledRequiredInteraction'} for key in keys}
    if keys:
        properties[keys[0]] = {'$ref': '#/$defs/ScheduledFirstInteraction'}
    required = list(keys)
    if terminal:
        properties['result_hold'] = _without_fields(definitions['PassiveUnit'], ('id', 'kind'))
        required.append('result_hold')
    without_opening = {'type': 'object', 'properties': properties,
                       'required': required, 'additionalProperties': False}
    variants = [without_opening]
    if keys and optional_capacity:
        # 非空入场用 first + following 表示；required 在提供方投影后仍然有效，
        # 不依赖会被移除的数组 minItems，也不让首项猜测前驱是否存在。
        with_opening = deepcopy(without_opening)
        with_opening['properties']['opening'] = {'type': 'object', 'properties': {
            'first': {'$ref': '#/$defs/ScheduledPassiveUnit'},
            'following': {'type': 'array', 'maxItems': optional_capacity - 1,
                          'items': {'$ref': '#/$defs/ScheduledPassiveUnit'}}},
            'required': ['first'], 'additionalProperties': False}
        with_opening['required'].append('opening')
        with_opening['properties'][keys[0]] = {'$ref': '#/$defs/ScheduledAfterOpeningInteraction'}
        variants.append(with_opening)
    if keys and explicit_absence:
        # 缺省与null都表示没有准备段，不占动作容量；对象一旦出现就必须完整。
        # 不用 action=""、weight=0 等无效单元表达“没有”。
        without_opening['properties']['opening'] = {'type': 'null'}
    result['properties']['shot']['properties']['action_plan'] = {'type': 'object', 'oneOf': variants}
    if not keys:
        # 无数量任务仍使用同一动作类型；必填首项避免空表演，不虚构交互配额。
        first = deepcopy(following_interaction)
        for variant in first['oneOf']:
            variant['properties']['overlap'] = {'type': 'boolean', 'const': False}
            variant['properties']['link_previous'] = {'type': 'boolean', 'const': False}
        definitions['ScheduledFirstUnit'] = {'oneOf': [first, {'$ref': '#/$defs/ScheduledPassiveUnit'}]}
        if not has_character:
            definitions['ScheduledFirstUnit'] = {'$ref': '#/$defs/ScheduledPassiveUnit'}
        properties.update(first={'$ref': '#/$defs/ScheduledFirstUnit'}, following={
            'type': 'array', 'maxItems': limit - 1 - int(terminal),
            'items': {'$ref': '#/$defs/ScheduledFollowingUnit'}})
        result['properties']['shot']['properties']['action_plan'] = {
            'type': 'object', 'properties': properties, 'required': ['first', *required],
            'additionalProperties': False}
    # 只保留新响应能够到达的定义，不向模型同时展示原数组的 id/kind 所有权。
    definitions = result.pop('$defs')
    reachable = {}

    def visit(node):
        if isinstance(node, dict):
            reference = node.get('$ref')
            if isinstance(reference, str) and reference.startswith('#/$defs/'):
                name = reference.removeprefix('#/$defs/')
                if name not in reachable:
                    reachable[name] = definitions[name]
                    visit(reachable[name])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    result['$defs'] = reachable
    return result


def _unit(raw: object, index: int, *, kind: str | None = None, passive_only: bool = False) -> dict:
    if not isinstance(raw, dict) or 'id' in raw or (kind is not None and 'kind' in raw):
        raise ValueError('动作内容必须是对象；编号及绑定的类型由程序提供，不能覆盖')
    unit = {'id': f'unit_{index + 1}', **deepcopy(raw)}
    if kind is not None:
        unit['kind'] = kind
    if unit.get('kind') in INDEPENDENT_ACTIVITIES:
        PassiveUnit.model_validate(unit)
    elif unit.get('kind') == 'interaction' and not passive_only:
        if ('carry' in unit or 'outcomes' in unit or 'response' not in unit
                or not isinstance(unit.get('resolution'), dict)
                or type(unit.get('link_previous')) is not bool):
            raise ValueError('新交互创作必须使用 resolution 与 link_previous，不能混用旧合同')
        parsed = InteractionUnit.model_validate(unit)
        CandidateResolution.model_validate(unit['resolution'])
        if parsed.effect.mode not in RESPONSE_MODES[parsed.response]:
            raise ValueError('交互的反应类别与作用方式不一致')
    else:
        raise ValueError('动作类型不适用于当前创作位置')
    return unit


def bind_authored_schedule(raw: dict, slot: dict, *, explicit_absence: bool = False) -> dict:
    """只展开已填写的内容；缺失、混用或超容量整体拒绝，不补造动作。"""
    if isinstance(raw, dict) and 'result_admission' in raw:
        raise ValueError('结果审查状态由程序管理，创作响应不能提供批准或待审状态')
    keys, terminal, limit = _layout(slot)
    schedule = raw.get('action_plan') if isinstance(raw, dict) else None
    required = set(keys) | ({'result_hold'} if terminal else set())
    if not keys:
        required.add('first')
    allowed = required | ({'opening'} if keys else {'following'})
    if (not isinstance(schedule, dict) or not required <= set(schedule)
            or not set(schedule) <= allowed):
        raise ValueError('动作创作必须逐项覆盖槽位绑定的必需作用，不能使用旧列表、缺项或新增键')
    camera_issues = camera_contract_issues(raw)
    if camera_issues:
        raise ValueError('；'.join(camera_issues))
    units = []

    def append(value, *, kind=None, passive_only=False):
        if len(units) >= limit:
            raise ValueError('展开后的动作超出已有单元容量；未截断或删除创作')
        units.append(_unit(value, len(units), kind=kind, passive_only=passive_only))

    if not keys:
        first = schedule['first']
        if isinstance(first, dict) and (first.get('overlap', False) is not False
                                       or first.get('link_previous', False) is not False):
            raise ValueError('本镜第一个动作没有前项，不能绑定或搭接不存在的运动来源')
        following = schedule.get('following', [])
        if not isinstance(following, list) or len(following) > limit:
            raise ValueError('following必须是有界的后续动作列表')
        append(first)
        for value in following:
            append(value)
    if 'opening' in schedule and not (explicit_absence and schedule['opening'] is None):
        opening = schedule['opening']
        if (not isinstance(opening, dict) or 'first' not in opening
                or not set(opening) <= {'first', 'following'}):
            raise ValueError('opening存在时必须包含首个入场动作；没有入场应省略该项')
        following = opening.get('following', [])
        if not isinstance(following, list) or len(following) > limit:
            raise ValueError('opening.following必须是有界的过渡或停顿列表')
        append(opening['first'], passive_only=True)
        for value in following:
            append(value, passive_only=True)
    for key in keys:
        value = schedule[key]
        if not isinstance(value, dict):
            raise ValueError('每个必需作用必须给出完整创作对象')
        if key == keys[0] and value.get('overlap', False) is not False:
            raise ValueError('首个必需作用之前没有交互，不能与被动入场或停顿搭接')
        if not units and value.get('link_previous') is True:
            raise ValueError('本镜第一个动作没有前项，不能绑定不存在的运动来源')
        following = value.get('following', [])
        if not isinstance(following, list) or len(following) > limit:
            raise ValueError('following必须是有界的后续动作列表')
        append({name: part for name, part in value.items() if name != 'following'}, kind='interaction')
        for item in following:
            append(item)
    if terminal:
        append(schedule['result_hold'], kind='hold')
    result = deepcopy(raw)
    framing = CANONICAL_FRAMING.get(slot.get('framing_family'))
    if framing:
        camera = result.get('camera')
        if not isinstance(camera, dict) or camera.get('start_framing', framing) != framing:
            raise ValueError('起始景别由生产槽位提供；丰富构图应在 composition 中表达')
        camera['start_framing'] = framing
    result['action_plan'] = units
    result['result_admission'] = 'review-required-v1'
    return result
