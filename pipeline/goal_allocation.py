"""冻结目标到原稿证据的多对多关联；动作之间没有必要性授权边。

审查提供来源摘录和证据，程序绑定来源区间。摘录只划分执行范围，不创作
新义务；全片合并时必须无遗漏地承接目标。历史逐动作授权保持原协议。
"""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pipeline.planning_evidence import MAX_REVIEW_UNITS, selection_schema, validate_selection


POLICY = 'goal-allocation-v1'
LEGACY_VERSION = 'planning-semantics-v11-goal-allocation'
SCOPED_VERSION = 'planning-semantics-v12-scoped-goals'
VERSION = 'planning-semantics-v13-catalog-goals'
VERSIONS = (LEGACY_VERSION, SCOPED_VERSION, VERSION)


class GoalRequirement(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)
    reference: str = Field(min_length=1, max_length=200)
    start: int = Field(strict=True, ge=0, le=1600)
    end: int = Field(strict=True, gt=0, le=1600)


class GoalEvidenceUnit(BaseModel):
    """原稿位置与内容快照，用于确定执行范围，不授权表演细节。"""
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)
    shot_id: int = Field(strict=True, ge=1)
    beat_index: int = Field(strict=True, ge=0, le=11)
    actor: str = Field(min_length=1, max_length=1600)
    target: str = Field(max_length=1600)
    action: str = Field(min_length=1, max_length=1600)


class ScopedGoalRequirement(GoalRequirement):
    # 每个联合范围来自一次有界审查；实际成员资格在绑定和全片合并时核对。
    scope: list[GoalEvidenceUnit] = Field(min_length=2, max_length=MAX_REVIEW_UNITS)


_REQUIREMENT = TypeAdapter(GoalRequirement | ScopedGoalRequirement)


class AllocatedIntent(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)
    kind: Literal['allocated']
    requirements: list[GoalRequirement | ScopedGoalRequirement] = Field(max_length=150)


INSTRUCTION = '''goal_allocation以创作前目标为审查单位，固定键来自evidence_sources.goals，不能删除目标或改写目标。
每个目标的parts给出原稿如何实现它：quote逐字摘录目标中的完整功能语句（保留标点），units按全片顺序选择共同实现该功能的动作固定键，可以跨镜多选；relation为entailed/contradicted/undetermined，reason解释这些动作合起来如何完成摘录要求。
一个目标由一镜或多镜共同完成时只分配一次，程序保留各镜在该完整功能中的执行范围，不要求各镜重复完成全件事。可独立分配的功能也可分别摘录，所有部分合起来完整覆盖目标原文。持续展示或反复进行的目标可在实际承担它的多镜分别提供证据，reason解释持续必要性。每个摘录必须是有完整语义的必要功能，而非单字或姿态片段。
同一动作可实现多个目标，同一目标可由多个动作实现。units是完成证据而非表演相似性；不能用只转视线、准备、尝试等动作证明未完成的事件。明确的静止或等待目标照常提供证据。
不返回unit.completion_intent，不决定哪个动作是required/supporting，不创建动作之间的支持指针。未选作目标证据的表演仍保留原稿并接受因果、类型、连续性、身份和镜头审查，不独立增加完成义务。
缺失事件不能用改写quote或整体肯定来掩盖；同时核对目标本身忠于原需求，目标擅自扩写要求或原稿与目标矛盾时返回contradicted，证据不足返回undetermined。未实现目标的parts为空，不猜证据。
分批时只引用本批动作、只分配本批实际承担的目标部分，其他部分留空；全片合并后程序检查完整覆盖。不返回另一批的动作或虚构实现。
结果候选仍检查对象和因果，但不返回result_admission，不把作者状态升级为固定要求。
'''


def response_schema(base: dict, goals: dict, targets: dict, *, version: str = VERSION) -> dict:
    if version not in VERSIONS:
        raise ValueError('目标证据协议无效')
    schema = deepcopy(base)
    part = {'type': 'object', 'properties': {
        'quote': {'type': 'string', 'minLength': 1, 'maxLength': 1600},
        'units': selection_schema(targets, legacy_limit=None if version == VERSION else 12),
        'relation': {'type': 'string', 'enum': ['entailed', 'contradicted', 'undetermined']},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 1600}},
        'required': ['quote', 'units', 'relation', 'reason'], 'additionalProperties': False}
    goal = {'type': 'object', 'properties': {'parts': {
        'type': 'array', 'items': part, 'maxItems': 150}}, 'required': ['parts'], 'additionalProperties': False}
    schema['properties']['goal_allocation'] = {'type': 'object',
        'properties': {ref: deepcopy(goal) for ref in goals}, 'required': list(goals), 'additionalProperties': False}
    schema['required'].append('goal_allocation')
    return schema


def decode_allocation(raw: dict, goals: dict, targets: dict, *, storyboard: dict | None = None,
                      version: str = VERSION) -> dict:
    from pipeline.planning_review_protocol import _exact_keys
    if version not in VERSIONS:
        raise ValueError('目标证据协议无效')
    allocation = raw.get('goal_allocation')
    _exact_keys(allocation, goals, '目标覆盖')
    result = deepcopy(raw)
    result.pop('goal_allocation')
    units = result.get('units')
    if not isinstance(units, list) or any(not isinstance(u, dict) or
            type(u.get('shot_id')) is not int or type(u.get('beat_index')) is not int for u in units):
        raise ValueError('目标分配缺少有效的逐动作审查')
    by_key = {f"{u['shot_id']}:{u['beat_index']}": u for u in units}
    if len(by_key) != len(units) or set(by_key) != set(targets):
        raise ValueError('目标分配审查必须完整且无重复地覆盖当前动作目录')
    originals = ({f"{s['shot_id']}:{i}": evidence_unit(s['shot_id'], i, b)
                  for s in storyboard['shots'] for i, b in enumerate(s['action_beats'])}
                 if storyboard is not None else {})
    if storyboard is not None and (len(originals) != sum(len(s['action_beats']) for s in storyboard['shots'])
                                   or list(originals) != list(targets)):
        raise ValueError('目标证据目录与原稿来源或时间线不一致')
    if any(k != f"{t.get('shot_id')}:{t.get('beat_index')}" for k, t in targets.items()):
        raise ValueError('目标证据目录键与动作身份不一致')
    for unit in by_key.values():
        if unit.get('completion_intent') is not None:
            raise ValueError('目标分配不能混用逐动作必要性授权')
        unit['completion_intent'] = {'kind': 'allocated', 'requirements': []}
    for ref, judgment in allocation.items():
        _exact_keys(judgment, ['parts'], '目标分配')
        parts = judgment['parts']
        if not isinstance(parts, list) or len(parts) > 150:
            raise ValueError('目标分配必须是有界证据列表')
        text = goals[ref]['goal']
        for part_index, part in enumerate(parts):
            _exact_keys(part, ['quote', 'units', 'relation', 'reason'], '目标证据')
            quote, keys = part['quote'], part['units']
            if (not isinstance(quote, str) or not quote.strip() or quote not in text
                    or text.find(quote, text.find(quote) + 1) != -1):
                raise ValueError('执行范围必须是目标原文的唯一完整摘录，不能新增或改写要求')
            validate_selection(keys, targets, label=f'{ref}/parts/{part_index}',
                               legacy_limit=None if version == VERSION else 12)
            cross_shot = len({targets[k]['shot_id'] for k in keys}) > 1
            if cross_shot and (storyboard is None or version == LEGACY_VERSION):
                raise ValueError('历史目标证据必须引用同一镜的实际动作；不能隐式迁移执行范围')
            if (part['relation'] != 'entailed' or not isinstance(part['reason'], str)
                    or not part['reason'].strip() or len(part['reason']) > 1600):
                raise ValueError('目标实现证据不足或矛盾；未改稿、未重新审查')
            requirement = {'reference': ref, 'start': text.index(quote), 'end': text.index(quote) + len(quote)}
            if cross_shot:
                if any(k not in originals for k in keys):
                    raise ValueError('跨镜执行范围缺少原稿来源')
                requirement['scope'] = [originals[k] for k in keys]
            for key in keys:
                items = by_key[key]['completion_intent']['requirements']
                if requirement in items:
                    raise ValueError('目标执行范围不能重复分配')
                items.append(deepcopy(requirement))
    return result


def requirement_key(item: dict) -> str:
    key = f"{item['reference']}@{item['start']}:{item['end']}"
    if 'scope' in item:
        # 同一功能可以再次发生；来源区间不是一次联合执行的身份。
        key += '/units:' + ','.join(f"{u['shot_id']}:{u['beat_index']}" for u in item['scope'])
    return key


def requirement_text(item: dict, goals: dict) -> str:
    item = _REQUIREMENT.validate_python(item).model_dump()
    goal = goals.get(item['reference'])
    if not goal or not 0 <= item['start'] < item['end'] <= len(goal['goal']):
        raise ValueError('执行范围越出冻结目标来源')
    return goal['goal'][item['start']:item['end']]


def validate_intents(intents: list[dict], goals: dict) -> None:
    for intent in intents:
        intent = AllocatedIntent.model_validate(intent).model_dump()
        items = intent['requirements']
        if len({requirement_key(i) for i in items}) != len(items):
            raise ValueError('动作不能重复承接同一目标范围')
        for item in items:
            if not requirement_text(item, goals).strip():
                raise ValueError('执行范围不能只包含空白')
            if 'scope' in item:
                keys = [(u['shot_id'], u['beat_index']) for u in item['scope']]
                if len(set(keys)) != len(keys) or len({s for s, _ in keys}) < 2:
                    raise ValueError('跨镜执行范围必须包含不同镜的无重复来源')


def validate_coverage(storyboard: dict, goals: dict) -> None:
    """跨镜、跨批合并后核对来源守恒；不从关键词推断完成。"""
    validate_scopes(storyboard)
    spans = {ref: [] for ref in goals}
    for shot in storyboard['shots']:
        if shot.get('completion_policy') != POLICY:
            raise ValueError('全片目标分配协议不能混用')
        seen = set()
        for beat in shot['action_beats']:
            for item in beat['completion_intent']['requirements']:
                requirement_text(item, goals)
                key = requirement_key(item)
                if key not in seen:
                    spans[item['reference']].append((item['start'], item['end']))
                    seen.add(key)
    for ref, goal in goals.items():
        end = 0
        for start, stop in sorted(spans[ref]):
            if goal['goal'][end:start].strip():
                raise ValueError('全片未承接必要目标：' + ref)
            end = max(end, stop)
        if goal['goal'][end:].strip():
            raise ValueError('全片未承接必要目标：' + ref)


def completion_criteria(shot: dict, intents: list[dict]) -> dict:
    criteria = {}
    goals = shot['completion_goals']
    for index, (beat, intent) in enumerate(zip(shot['action_beats'], intents), 1):
        keys = []
        for item in intent['requirements']:
            key = requirement_key(item)
            keys.append(key)
            criterion = criteria.setdefault(key, {'kind': 'completion', 'beat_id': index,
                'beat_ids': [], 'actors': [], 'targets': [],
                'action': requirement_text(item, goals),
                'source': {k: v for k, v in item.items() if k != 'scope'}})
            if 'scope' in item:
                scope = {'shot_id': shot['shot_id'], 'units': deepcopy(item['scope'])}
                if criterion.get('execution_scope', scope) != scope:
                    raise ValueError('同一功能的执行范围自相矛盾')
                criterion['execution_scope'] = scope
            criterion['beat_ids'].append(index)
            for field, value in [('actors', beat['actor']), ('targets', beat.get('target', ''))]:
                if value and value not in criterion[field]:
                    criterion[field].append(value)
        criteria[f'beat:{index}/completion'] = {'kind': 'realization', 'beat_id': index,
            'actor': beat['actor'], 'supports': keys}
    for criterion in criteria.values():
        if 'execution_scope' in criterion:
            local = [u for u in criterion['execution_scope']['units'] if u['shot_id'] == shot['shot_id']]
            expected = [u['beat_index'] + 1 for u in local]
            if criterion['beat_ids'] != expected:
                raise ValueError('本镜没有完整承接审定的跨镜执行范围')
            for unit in local:
                if evidence_unit(shot['shot_id'], unit['beat_index'], shot['action_beats'][unit['beat_index']]) != unit:
                    raise ValueError('本镜执行范围与原稿来源不一致')
    return criteria


def evidence_unit(shot_id: int, index: int, beat: dict) -> dict:
    return GoalEvidenceUnit(shot_id=shot_id, beat_index=index, actor=beat['actor'],
                            target=beat.get('target', ''), action=beat['action']).model_dump()


def validate_scopes(storyboard: dict) -> None:
    """联合证据不能在分发、序列化或批次合并中丢掉一个承担阶段。"""
    actual = {f"{s['shot_id']}:{i}": (s, b, i)
              for s in storyboard['shots'] for i, b in enumerate(s['action_beats'])}
    order = {key: i for i, key in enumerate(actual)}
    groups = {}
    for key, (_, beat, _) in actual.items():
        for item in beat['completion_intent']['requirements']:
            ref = requirement_key(item)
            group = groups.setdefault(ref, {'scope': item.get('scope'), 'owners': set()})
            if group['scope'] != item.get('scope'):
                raise ValueError('同一联合执行范围包含不一致的来源快照')
            group['owners'].add(key)
    for group in groups.values():
        if group['scope'] is None:
            continue
        keys = [f"{u['shot_id']}:{u['beat_index']}" for u in group['scope']]
        if (set(keys) != group['owners'] or any(k not in actual for k in keys)
                or keys != sorted(keys, key=order.__getitem__)):
            raise ValueError('全片未完整承接跨镜执行范围，或范围来源无序')
        for key, unit in zip(keys, group['scope']):
            shot, beat, index = actual[key]
            if evidence_unit(shot['shot_id'], index, beat) != unit:
                raise ValueError('跨镜执行范围与原稿来源不一致')


SCOPED_INSTRUCTION = (
    'For execution_scope, action is a joint goal across the listed shots, not a whole-goal obligation '
    'in every shot. Evaluate only this shot\'s functional contribution, using its listed units to '
    'locate that contribution within the source goal. Other shots are context, never evidence of '
    'unseen completion and never actions to repeat here. A missing local contribution fails even '
    'if a later shot plans the final outcome. The units are authored realizations, not additional '
    'requirements: preserve the necessary event, participants, response and explicit user method, '
    'while allowing equivalent choreography and incidental poses. Unknown evidence stays unknown. '
    'Each participating shot must pass its own contribution; local success alone is not whole-goal success.'
)


def criterion_instruction(criterion: dict) -> str:
    if 'execution_scope' not in criterion:
        return criterion['action']
    # 生成不接收其他镜的动作文案，避免将上下文再次执行；审查保留联合原稿用于定位职责。
    scope = criterion['execution_scope']
    return (f"Joint goal context: {criterion['action']} Execute only this shot's contribution "
            f"in beats {criterion['beat_ids']}; other participating shots "
            f"{list(dict.fromkeys(u['shot_id'] for u in scope['units'] if u['shot_id'] != scope['shot_id']))} "
            'carry the remaining work. Do not repeat or anticipate their work. '
            'Preserve the local necessary event and explicit method; authored poses are realization choices.')
