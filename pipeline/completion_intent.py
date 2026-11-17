"""审定功能拥有验收权；作者的实现细节保留拍摄权，不反向成为要求。"""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from pipeline.planning_review_sources import review_evidence_sources
from pipeline.goal_allocation import AllocatedIntent
from pipeline import goal_allocation as allocation
from pipeline.intent_evidence import RequiredEvidence, evidence_grounded


POLICY = 'admitted-intent-v1'
SOURCE_POLICY = 'source-goals-v1'


class CompletionGoal(BaseModel):
    """在分镜创作前冻结；实现方案不能反向增加目标。"""
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True, str_strip_whitespace=True)
    goal: str = Field(min_length=1, max_length=1600)
    basis: Literal['user', 'story']
    evidence: RequiredEvidence


GOAL_INSTRUCTION = '''completion_goals在任何分镜产生之前定义影片的可观察目标，按叙事顺序列出。
每项包含goal、basis(user/story)、evidence(选择原需求目录的一个或多个来源地址)。user覆盖明确事件、结果、指定动作方法；story只补充实现主题所需的最少叙事功能。
目标描述观众必须看到什么发生、谁受到什么影响；不设计左右手、眼神路线、站姿、镜头内停留等表演实现，除非用户明确指定。
不要把一整段剧情塞进一个目标，也不要把一种表演拆成多个要求。场景、人物、禁止项、风格和播放速度仍由原需求及已有字段完整保留。
只有氛围、展示或抽象主题时，也描述它最少需要呈现的可观察内容，不强加冲突、交互或结局。不得按镜头数复制目标。
'''


def source_goals(storyboard: dict) -> dict:
    intent = (storyboard.get('production_plan') or {}).get('director_intent') or {}
    if intent.get('authoring_protocol') != 'scene-units-v1':
        return {}  # 精确编舞和参考复现继续使用其原始执行权威。
    values = intent.get('completion_goals')
    if values is None:
        return {}
    if not isinstance(values, list) or not 1 <= len(values) <= 150:
        raise ValueError('创作前目标目录必须非空且有界')
    result = {}
    for index, value in enumerate(values):
        goal = CompletionGoal.model_validate(value).model_dump()
        if not evidence_grounded(goal['evidence'], intent.get('source_request', '')):
            raise ValueError('创作前目标必须引用原始需求')
        result[f'/intent/completion_goals/{index}'] = goal
    return result


class RequiredIntent(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True, str_strip_whitespace=True)
    kind: Literal['required']
    goal: str = Field(min_length=1, max_length=1600)
    basis: Literal['user', 'continuity', 'story']
    reference: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=1600)


class SupportingIntent(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True, str_strip_whitespace=True)
    kind: Literal['supporting']
    supports: int = Field(strict=True, ge=1, le=12)
    reason: str = Field(min_length=1, max_length=1600)


CompletionIntent = Annotated[RequiredIntent | SupportingIntent | AllocatedIntent, Field(discriminator='kind')]
_INTENT = TypeAdapter(CompletionIntent)

INSTRUCTION = '''每个unit必须独立审定completion_intent，覆盖所有activity，不按题材或hold/transition标签判断必要性。
先回答“去掉或换掉这个具体表演，用户事件、后续因果或本镜叙事功能是否仍成立”。
required的goal写最小且完整、可观察的必要功能（包含发生过程与目标反应，不能仅写一个静态终点）；不重抄混有姿态/实现路径的action整句。
basis=user引用用户明确要求；continuity引用确实无法缺少的后续依赖；story引用本镜narrative_beat的实际叙事变化。
reference必须选择evidence_sources.allowed中该项对应basis的地址。reason解释该引用为何要求goal，引用存在不等于蕴含成立。
用户明确指定的方法、节奏、动作、凝视、等待或姿态仍是必要功能；不能因为动作小、静止或没有交互就降级。
没有独立必要功能的表演片段用supporting，supports指向本镜另一个required的动作序号（从1开始），reason解释替换该表演为何不损害该功能。
supporting不能自指、互相引用或指向另一supporting，不能把整镜工作降级，也不能掩盖明确要求、关键物体变化、真实作用反应和后续依赖。
原动作与结果不重写；功能审定与原始拍摄方案并存。固定user/continuity结果照常审查，supporting不豁免它们。
所有unit的功能合起来必须保留完整原需求、事件顺序和全片结局，不能只因局部合理就批准遗漏；无法完成这个判断时，现有结局证据、接缝证据或结果保全判断必须明确否定或证据不足。使用本次schema的实际字段，不返回已被证据协议替代的整体布尔值。
'''


def intent_evidence_sources(storyboard: dict) -> dict:
    sources = review_evidence_sources(storyboard)
    goals = source_goals(storyboard)
    if goals:
        sources['goals'] = goals
        for path, goal in goals.items():
            sources['fields'][path] = goal['goal']
        for allowed in sources['allowed'].values():
            # 仅功能引用受目录约束；旧结果归类目录保持原有含义。
            allowed['goal_references'] = list(goals)
        return sources
    for i, shot in enumerate(storyboard['shots']):
        narrative = shot.get('narrative_beat') or {}
        paths = []
        for field in ('state_change', 'state_after'):
            text = narrative.get(field)
            if isinstance(text, str) and text.strip():
                path = f'/shots/{i}/narrative_beat/{field}'
                paths.append(path)
                sources['fields'][path] = text
        for j in range(len(shot['action_beats'])):
            sources['allowed'][f"{shot['shot_id']}:{j}"]['story'] = paths
    return sources


def intent_schema(sources: dict, key: str, owners: list[int]) -> dict:
    variants = []
    if sources.get('goals'):
        variants.append({'type': 'object', 'properties': {
            'kind': {'type': 'string', 'const': 'required'},
            'reference': {'type': 'string', 'enum': list(sources['goals'])},
            'reason': {'type': 'string', 'minLength': 1, 'maxLength': 1600}},
            'required': ['kind', 'reference', 'reason'], 'additionalProperties': False})
    for basis in (() if sources.get('goals') else ('user', 'continuity', 'story')):
        references = sources['allowed'][key].get(basis, [])
        if not references:
            continue
        schema = RequiredIntent.model_json_schema()
        schema['properties']['basis'] = {'type': 'string', 'const': basis}
        schema['properties']['reference'] = {'type': 'string', 'enum': references}
        variants.append(schema)
    if owners:
        schema = SupportingIntent.model_json_schema()
        schema['properties']['supports'] = {'type': 'integer', 'enum': owners}
        variants.append(schema)
    if not variants:
        raise ValueError('功能审定缺少需求或叙事来源')
    return {'oneOf': variants}


def validate_intent(value: object, sources: dict, key: str) -> dict:
    if (sources.get('goals') and isinstance(value, dict)
            and set(value) == {'kind', 'reference', 'reason'} and value['kind'] == 'required'):
        goal = sources['goals'].get(value['reference']) if isinstance(value['reference'], str) else None
        if not goal:
            raise ValueError('分镜引用了不存在的创作前目标')
        value = {**value, 'goal': goal['goal'], 'basis': goal['basis']}
    intent = _INTENT.validate_python(value).model_dump()
    if intent['kind'] == 'allocated':
        raise ValueError('历史逐动作审查不能隐式升级为目标分配协议')
    if sources.get('goals'):
        if intent['kind'] == 'required':
            goal = sources['goals'].get(intent['reference'])
            if not goal or any(intent[k] != goal[k] for k in ('goal', 'basis')):
                raise ValueError('分镜只能绑定创作前目标，不能新增或改写必要功能')
        return intent
    if intent['kind'] == 'required' and intent['reference'] not in sources['allowed'][key].get(intent['basis'], []):
        raise ValueError('必要功能必须引用当前动作允许的需求、后续依赖或叙事来源')
    return intent


def validate_links(intents: list[dict]) -> None:
    if not intents or not any(i['kind'] == 'required' for i in intents):
        raise ValueError('整镜必须保留必要功能，不能全部降级为表演细节')
    for index, intent in enumerate(intents, 1):
        if intent['kind'] == 'supporting':
            owner = intent['supports']
            if owner == index or owner > len(intents) or intents[owner - 1]['kind'] != 'required':
                raise ValueError('表演细节必须直接支持本镜另一项必要功能，不能自指、循环或越界')


def validate_allocated_work(shot: dict, intents: list[dict]) -> None:
    """必要性投影不能消掉生产计划已经承诺的交互配额。"""
    slot = shot.get('production_slot') or {}
    required = sum(b.get('activity') == 'interaction' and (i['kind'] == 'required'
                   or (i['kind'] == 'allocated' and bool(i['requirements'])))
                   for b, i in zip(shot['action_beats'], intents))
    if (required < slot.get('min_interaction_events', 0)
            or (not required and (slot.get('min_interaction_seconds', 0) > 0
                                 or slot.get('result_hold_seconds', 0) > 0))):
        raise ValueError('必要功能不能取消已有的交互分配；需由独立审查保留对应工作')


def admitted_unit_intents(storyboard: dict, units: list[dict]) -> dict:
    if not any(u.get('completion_intent') is not None for u in units):
        if source_goals(storyboard):
            raise ValueError('创作前目标必须经过分镜绑定审查')
        return {}  # 历史审查不升级，不用旧结果推测新功能。
    sources = intent_evidence_sources(storyboard)
    by_key = {(u['shot_id'], u['beat_index']): u for u in units}
    result = {}
    for shot in storyboard['shots']:
        if shot.get('result_admission') != 'review-required-v1':
            continue
        raw = [by_key[(shot['shot_id'], j)].get('completion_intent') for j in range(len(shot['action_beats']))]
        if any(isinstance(i, dict) and i.get('kind') == 'allocated' for i in raw):
            intents = [AllocatedIntent.model_validate(i).model_dump() for i in raw]
            allocation.validate_intents(intents, sources.get('goals', {}))
        else:
            intents = [validate_intent(value, sources, f"{shot['shot_id']}:{j}") for j, value in enumerate(raw)]
            validate_links(intents)
        validate_allocated_work(shot, intents)
        result[shot['shot_id']] = intents
    return result


def admitted_shot_intents(shot: dict) -> list[dict]:
    policy = shot.get('completion_policy')
    raw = [b.get('completion_intent') for b in shot.get('action_beats', [])]
    if policy is None:
        if any(v is not None for v in raw) or shot.get('completion_goals') is not None:
            raise ValueError('功能审定缺少协议来源')
        return []
    if (policy not in (POLICY, SOURCE_POLICY, allocation.POLICY) or shot.get('result_policy') != 'functional-v1'
            or shot.get('result_admission') != 'reviewed-results-v1'):
        raise ValueError('功能审定协议与结果审查不一致')
    intents = [_INTENT.validate_python(value).model_dump() for value in raw]
    if policy != allocation.POLICY and any(i['kind'] == 'allocated' for i in intents):
        raise ValueError('历史逐动作合同不能混入目标分配')
    if policy in (SOURCE_POLICY, allocation.POLICY):
        goals = shot.get('completion_goals')
        if not isinstance(goals, dict) or not (0 if policy == allocation.POLICY else 1) <= len(goals) <= 150 or any(not isinstance(k, str) for k in goals):
            raise ValueError('执行镜头缺少已冻结目标来源')
        goals = {k: CompletionGoal.model_validate(v).model_dump() for k, v in goals.items()}
        if policy == allocation.POLICY:
            allocation.validate_intents(intents, goals)
        else:
            for intent in intents:
                validate_intent(intent, {'goals': goals}, '')
    if policy != allocation.POLICY:
        validate_links(intents)
    validate_allocated_work(shot, intents)
    return intents


def validate_goal_coverage(storyboard: dict) -> None:
    """全片合并后检查来源覆盖；不将未完成的分批计划误判为全片遗漏。"""
    goals = source_goals(storyboard)
    if not goals:
        return
    for shot in storyboard['shots']:
        if shot.get('completion_policy') not in (SOURCE_POLICY, allocation.POLICY):
            raise ValueError('新目标目录不能沿用旧的必要性授权')
        for path, value in (shot.get('completion_goals') or {}).items():
            if path not in goals or value != goals[path]:
                raise ValueError('执行目标与创作前冻结目录不一致')
    if any(s.get('completion_policy') == allocation.POLICY for s in storyboard['shots']):
        for shot in storyboard['shots']:
            admitted_shot_intents(shot)
        allocation.validate_coverage(storyboard, goals)
        return
    references = {intent['reference'] for shot in storyboard['shots']
                  for intent in admitted_shot_intents(shot) if intent['kind'] == 'required'}
    missing = [path for path, goal in goals.items() if goal['basis'] == 'user' and path not in references]
    if missing:
        raise ValueError('全片未承接必要目标：' + ', '.join(missing))


def intent_instruction(shot: dict) -> str:
    intents = admitted_shot_intents(shot)
    if not intents:
        return ''
    if shot.get('completion_policy') == allocation.POLICY:
        criteria = allocation.completion_criteria(shot, intents)
        goals = '; '.join(allocation.criterion_instruction(c) for c in criteria.values() if c['kind'] == 'completion')
        if any('execution_scope' in c for c in criteria.values()):
            return ('Required observable functions in their execution scopes: ' + goals
                    + '. Complete each local contribution once. Preserve causal responses, timing and explicit user methods; '
                    'performance details do not add completion obligations.')
        return ('Required observable functions: ' + goals + '. For this shot, complete each function once across '
                'its contributing actions. Preserve authored performance, causal responses, timing and explicit user methods. '
                'Performance details do not introduce additional completion obligations.')
    goals = '; '.join(f"beat {i}: {item['goal']}" for i, item in enumerate(intents, 1) if item['kind'] == 'required')
    return ('Required observable functions: ' + goals + '. Authored action/pose text is the proposed realization; '
            'preserve these functions, explicit user methods, fixed states, causal responses and timing. '
            'Supporting beats serve their linked function and do not mandate an additional pose or event.')
