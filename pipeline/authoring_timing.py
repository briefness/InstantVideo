"""创作接口只暴露可执行的时间关系；历史运动合同由既有编译器解释。"""
from copy import deepcopy


LEGACY_VERSION = 'story-context-v8-action-semantics'
VERSION = 'story-context-v9-temporal-relations'
INSTRUCTION = (
    'timing 只决定时间：sequential 表示前项结束后紧接本项，不要求停顿、复位或相同人物；'
    'overlap_previous 表示前一动作尚未结束时开始本项，可以是不同人物或对象的并行表演，前项也可以是移动或停留。'
    '首项和最终结果展示使用 sequential；非交互省略 timing 时表示 sequential。'
    'interaction 的 motion_source 与时间独立：none（省略时默认）不声明运动来源；previous 才表示承接前一动作中共同参与者的运动。'
    '顺序动作也可以承接运动；同时发生本身不意味着承接。不要把独立的同时动作绑定为运动来源。'
    '看见、听见或理解前项而作出回应是剧情因果，不是时间重叠。'
    '具体借势、回应和连续表演保留在 action/result，列表保持动作开始的顺序。'
)


def execution_timing_schema(schema: dict, *, version: str = LEGACY_VERSION) -> dict:
    """按版本投影槽位契约；新版将时间关系与运动来源分开。"""
    result = deepcopy(schema)
    if version not in (LEGACY_VERSION, VERSION):
        raise ValueError('未知创作时序协议')
    independent = version == VERSION
    if independent:
        # opening 已提供前驱；历史 schema 的禁止搭接不适用于独立时间关系。
        for variant in result.get('$defs', {}).get('ScheduledAfterOpeningInteraction', {}).get('oneOf', []):
            variant['properties']['overlap'] = {'type': 'boolean'}

    def visit(node):
        if isinstance(node, dict):
            properties = node.get('properties')
            if independent and isinstance(properties, dict) and {'actor', 'action', 'result'} <= properties.keys():
                properties['timing'] = {'type': 'string', 'enum': ['sequential', 'overlap_previous'],
                                        'default': 'sequential'}
            if isinstance(properties, dict) and 'actor' in properties and 'link_previous' in properties:
                can_overlap = properties.get('overlap', {}).get('const') is not False
                can_link = properties['link_previous'].get('const') is not False
                properties.pop('link_previous')
                properties.pop('overlap', None)
                properties['timing'] = {'type': 'string',
                    'enum': ['sequential', 'overlap_previous'] if can_overlap else ['sequential'],
                    'description': ('顺序紧接，或在前项结束前开始；不声明运动来源。' if independent else
                                    '顺序紧接，或与前项交互的共同运动在时间上搭接；不表示剧情回应。')}
                if independent:
                    properties['motion_source'] = {'type': 'string', 'default': 'none',
                        'enum': ['none', 'previous'] if can_link else ['none']}
                node['required'] = [k for k in node.get('required', []) if k not in ('link_previous', 'overlap')]
                node['required'].append('timing')
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    if independent:
        definitions = result.get('$defs', {})
        if 'ScheduledPassiveUnit' in definitions:
            first = deepcopy(definitions['ScheduledPassiveUnit'])
            first['properties']['timing']['enum'] = ['sequential']
            definitions['ScheduledFirstPassiveUnit'] = first

        def restrict_boundaries(node):
            if isinstance(node, dict):
                for key, child in node.items():
                    if key in ('first', 'result_hold') and isinstance(child, dict):
                        if child.get('$ref') == '#/$defs/ScheduledPassiveUnit':
                            child['$ref'] = '#/$defs/ScheduledFirstPassiveUnit'
                        if 'timing' in child.get('properties', {}):
                            child['properties']['timing']['enum'] = ['sequential']
                    restrict_boundaries(child)
            elif isinstance(node, list):
                for child in node:
                    restrict_boundaries(child)
        restrict_boundaries(result)
        if definitions.get('ScheduledFirstUnit', {}).get('$ref') == '#/$defs/ScheduledPassiveUnit':
            definitions['ScheduledFirstUnit']['$ref'] = '#/$defs/ScheduledFirstPassiveUnit'
        for branch in definitions.get('ScheduledFirstUnit', {}).get('oneOf', []):
            if branch.get('$ref') == '#/$defs/ScheduledPassiveUnit':
                branch['$ref'] = '#/$defs/ScheduledFirstPassiveUnit'
    return result


def bind_execution_timing(raw: dict, *, version: str = LEGACY_VERSION) -> dict:
    """按请求版本绑定结构语义，不推测正文；默认保留历史调用的解释。"""
    if not isinstance(raw, dict) or not isinstance(raw.get('action_plan'), dict):
        raise ValueError('新时间编排必须使用槽位动作对象')
    result = deepcopy(raw)
    if version not in (LEGACY_VERSION, VERSION):
        raise ValueError('未知创作时序协议')
    independent = version == VERSION

    def unit(value, *, interaction=False):
        if not isinstance(value, dict):
            return
        is_interaction = interaction or value.get('kind') == 'interaction'
        if 'temporal_relation' in value:
            raise ValueError('temporal_relation 由编译边界管理，创作不能直接提供')
        if is_interaction or independent:
            if any(key in value for key in ('link_previous', 'overlap', 'carry')):
                raise ValueError('新时间协议不能混入旧承接字段 link_previous/overlap/carry')
            timing = value.get('timing', 'sequential' if not is_interaction else None)
            if not isinstance(timing, str) or timing not in ('sequential', 'overlap_previous'):
                raise ValueError('交互 timing 必须明确为 sequential 或 overlap_previous')
            value.pop('timing', None)
            if independent:
                value['temporal_relation'] = timing
                if is_interaction:
                    source = value.pop('motion_source', 'none')
                    if source not in ('none', 'previous'):
                        raise ValueError('motion_source 必须为 none 或 previous')
                    value['link_previous'] = source == 'previous'
                elif 'motion_source' in value:
                    raise ValueError('非交互不能声明交互运动来源')
            else:
                value.update(link_previous=timing == 'overlap_previous', overlap=timing == 'overlap_previous')
        following = value.get('following')
        if isinstance(following, list):
            for child in following:
                unit(child)

    schedule = result['action_plan']
    unit(schedule.get('first'))
    if isinstance(schedule.get('following'), list):
        for child in schedule['following']:
            unit(child)
    opening = schedule.get('opening')
    if isinstance(opening, dict):
        unit(opening.get('first'))
        if isinstance(opening.get('following'), list):
            for child in opening['following']:
                unit(child)
    for key, value in schedule.items():
        if key.startswith('interaction_'):
            unit(value, interaction=True)
    if independent and 'result_hold' in schedule:
        unit(schedule['result_hold'])
        if schedule['result_hold'].get('temporal_relation') != 'sequential':
            raise ValueError('结果展示必须在前项完成之后')
    return result
