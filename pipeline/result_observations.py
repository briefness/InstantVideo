"""一次观察同时用于动作完成判定与续拍交接，不生成第二份事实。

固定终态来自用户/后续依赖；新合同的功能来自独立审定，不等于作者的每个动作句。
历史合同不迁移。模型观察仍可能有语义误差，本模块只保证来源和派生一致。
"""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from pipeline.action_results import result_claims, functional_outcomes


DYNAMIC_STATE = ('action_phase', 'pose_and_gaze', 'prop_state', 'open_motion')
DERIVED_FIELDS = ('accepted', 'failure_reason', 'primary_action_completed', 'narrative_state_change_valid')


class ResultObservation(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True, str_strip_whitespace=True)
    status: Literal['satisfied', 'violated', 'unknown']
    observation: str = Field(min_length=1, max_length=1600)
    sample_indices: list[StrictInt] = Field(min_length=1, max_length=64)


class ObservationUncertaintyError(ValueError):
    """结构和来源有效，但必要事实尚无法从画面确认。"""
    def __init__(self, keys: list[str]):
        self.keys = keys
        label = '片尾快照观察不足' if 'terminal_observation' in keys else '动作或结果观察不足'
        super().__init__(label + '，语义不确定：' + ', '.join(keys)
                         + '；保留原片，不自动重试')


LINKED_STATE_INSTRUCTION = (
    'Include the actual ending situation in observations supported '
    'by the final current sample. end_state_sources maps each dynamic state field to observation keys '
    'supported by that final sample (one source per field); empty lists mean no observed information for that field, not absence. '
)

INSTRUCTION = (
    'Functional observation protocol: completion_contract is the sole catalogue of action completion '
    'and fixed-state obligations. For each key requested by the observations schema return exactly one entry: status '
    '(satisfied, violated, unknown), factual observation and 1-based current sample_indices. '
    'For kind=completion judge whether the intended action actually finishes, including its target '
    'response; preparation, touching, attempting or merely showing an effect is not completion. '
    'The action text describes the intended work, not a mandatory choreography or resting pose. '
    'Use only kind=state entries and explicit user requirements to mandate a particular endpoint. '
    'Judge transitions and holds as well as interactions. Alternative coherent execution is allowed. '
    'Never invent unseen completion. ' + LINKED_STATE_INSTRUCTION +
    'end_state_context describes only location, subject identity, camera, screen direction and lighting. '
    'Do not return accepted, failure_reason, primary_action_completed, narrative_state_change_valid, '
    'observed_end_state or event result_visible: these are derived from the same observations. '
    'Other schema judgments still use true/false/null; null and unknown pause review, not reject the take. '
    'All observations must come from the supplied video, never from the planned answer.'
    ' With admitted-intent-v1, kind=completion contains the independently admitted necessary function, '
    'not the original performance prose. kind=realization records actual performance for a linked '
    'function: its fidelity status does not independently veto completion. Still observe it truthfully '
    'for handoff. Fixed kind=state obligations, identity, causality and explicit user methods remain mandatory. '
    'Do not promote realization details through other aggregate judgments.'
    ' With observation_policy=requirements-only-v1, realization entries are only links between '
    'action units and required functions, not observation questions. Do not guess missing performance '
    'content from unit numbers. Judge the requested completion/state entries and describe the actual '
    'terminal state independently. Identity, causal order, explicit methods and camera checks still apply.'
)

SNAPSHOT_INSTRUCTION = (
    ' With source-goals-v1 or goal-allocation-v1, event observations describe what happened over their cited time intervals. '
    'A goal-allocation completion entry is one function shared by its beat_ids: judge the combined work once, '
    'not the entire function separately for each contributing beat. Realization supports lists reference these functions; '
    'an empty list adds no completion obligation. '
    'terminal_observation separately describes only the final current sample, whose index is fixed by the schema. '
    'Report the actual ending action_phase, pose_and_gaze, prop_state and open_motion there; do not copy '
    'an earlier completion description or force an early event to continue until the final frame. '
    'A completed action may be followed by a different pose or another action. The snapshot is factual handoff '
    'evidence, not another performance requirement. Use status=unknown when the final state cannot be seen; '
    'never infer it from the plan. There are no end_state_sources references in this protocol.'
)

# 同一范围投影同时解释生成、续拍和观察；不按题材或动作词推断必要性。
from pipeline.goal_allocation import SCOPED_INSTRUCTION
SNAPSHOT_INSTRUCTION += ' ' + SCOPED_INSTRUCTION


def completion_contract(shot: dict) -> dict:
    from pipeline.completion_intent import admitted_shot_intents
    intents = admitted_shot_intents(shot)
    version = shot.get('result_policy')
    if version is None:
        return {}
    if version != 'functional-v1':
        raise ValueError('未知的动作结果协议')
    if shot.get('result_admission') != 'reviewed-results-v1':
        return {}
    if ((shot.get('production_slot') or {}).get('choreography') != 'flexible'
            or shot.get('motion_reference') or shot.get('motion_plan')):
        raise ValueError('功能结果协议不能改写精确动作或参考复现合同')
    criteria = {}
    allocated = shot.get('completion_policy') == 'goal-allocation-v1'
    if allocated:
        from pipeline.goal_allocation import completion_criteria
        criteria = completion_criteria(shot, intents)
    for index, beat in enumerate(shot['action_beats'], 1):
        if allocated:
            for source, claim in result_claims(beat).items():
                if source not in [*(beat.get('result_directions') or []), *(beat.get('result_functions') or [])]:
                    criteria[f'beat:{index}/{source}'] = {'kind': 'state', 'beat_id': index, **claim}
            continue
        criteria[f'beat:{index}/completion'] = {
            'kind': 'completion', 'beat_id': index, 'activity': beat['activity'],
            'actor': beat['actor'], 'target': beat.get('target', ''), 'action': beat['action'],
            'functional_responses': functional_outcomes(beat)}
        if intents:
            intent = intents[index - 1]
            criteria[f'beat:{index}/completion'] = (
                {'kind': 'completion', 'beat_id': index, 'actor': beat['actor'],
                 'target': beat.get('target', ''), 'activity': beat['activity'], 'action': intent['goal']}
                if intent['kind'] == 'required' else
                {'kind': 'realization', 'beat_id': index, 'actor': beat['actor'],
                 'supports': f"beat:{intent['supports']}/completion"})
        for source, claim in result_claims(beat).items():
            if source not in [*(beat.get('result_directions') or []), *(beat.get('result_functions') or [])]:
                criteria[f'beat:{index}/{source}'] = {'kind': 'state', 'beat_id': index, **claim}
    if not criteria:
        raise ValueError('功能结果协议缺少动作来源')
    from pipeline.observed_boundary import assessment_criteria
    criteria.update(assessment_criteria(shot))
    return {'version': version, 'criteria': criteria,
            **({'execution_semantics': shot['execution_semantics']}
               if shot.get('execution_semantics') else {}),
            **({'evidence_policy': shot['evidence_policy']} if shot.get('evidence_policy') else {}),
            **({'observation_policy': 'requirements-only-v1'} if allocated else {}),
            **({'intent_policy': shot['completion_policy']} if intents else {})}


def observation_criteria(catalog: dict) -> dict:
    """动作关联是编译信息，不是另一组让模型猜测的完成问题。"""
    criteria = catalog['criteria']
    if catalog.get('observation_policy') not in (None, 'requirements-only-v1'):
        raise ValueError('未知的动作观察协议，不能静默按旧字段解释')
    if catalog.get('observation_policy') == 'requirements-only-v1':
        return {k: c for k, c in criteria.items() if c['kind'] != 'realization'}
    return criteria


def observation_properties(properties: dict, catalog: dict, sample_count: int) -> dict:
    result = deepcopy(properties)
    for name in DERIVED_FIELDS:
        result.pop(name, None)
    state = result.pop('observed_end_state')
    state['properties'] = {k: v for k, v in state['properties'].items() if k not in DYNAMIC_STATE}
    state['required'] = list(state['properties'])
    result['end_state_context'] = state
    witness = ResultObservation.model_json_schema()
    witness['properties']['sample_indices']['items'].update(minimum=1, maximum=sample_count)
    keys = list(observation_criteria(catalog))
    result['observations'] = {'type': 'object', 'properties': {k: deepcopy(witness) for k in keys},
                              'required': keys, 'additionalProperties': False}
    result['end_state_sources'] = {'type': 'object', 'properties': {
        k: {'type': 'array', 'items': {'type': 'string', 'enum': keys},
            'minItems': 1 if k == 'action_phase' else 0, 'maxItems': 1} for k in DYNAMIC_STATE},
        'required': list(DYNAMIC_STATE), 'additionalProperties': False}
    if catalog.get('intent_policy') in ('source-goals-v1', 'goal-allocation-v1'):
        result.pop('end_state_sources')
        result['terminal_observation'] = {'type': 'object', 'properties': {
            'sample_index': {'type': 'integer', 'const': sample_count},
            'status': {'type': 'string', 'enum': ['observed', 'unknown']},
            **{k: {'type': 'string', 'maxLength': 1600, 'minLength': 1 if k == 'action_phase' else 0}
               for k in DYNAMIC_STATE}},
            'required': ['sample_index', 'status', *DYNAMIC_STATE], 'additionalProperties': False}
    if 'action_event_evidence' in result:
        item = result['action_event_evidence']['items']
        item['properties'].pop('result_visible')
        item['required'].remove('result_visible')
    return result


def decode_observations(value: dict, catalog: dict, schema: dict, sample_count: int) -> dict:
    """只做来源校验与确定性派生，不从旧否决文本猜测新观察。"""
    if not isinstance(value, dict) or set(value) != set(schema['properties']):
        raise ValueError('观察字段与当前协议不符；不接受独立的总判定或重复终态')
    criteria = catalog['criteria']
    raw = value.get('observations')
    if not isinstance(raw, dict) or set(raw) != set(observation_criteria(catalog)):
        raise ValueError('动作与结果观察必须完整绑定合同来源')
    observations = {}
    unresolved = []
    for key, item in raw.items():
        witness = ResultObservation.model_validate(item)
        indices = witness.sample_indices
        if indices != sorted(set(indices)) or not all(1 <= i <= sample_count for i in indices):
            raise ValueError('观察采样来源越界、重复或无序')
        if witness.status == 'unknown' and criteria[key]['kind'] != 'realization':
            unresolved.append(key)
        observations[key] = witness
    state = value.get('end_state_context')
    fields = schema['properties']['end_state_context']['properties']
    if (not isinstance(state, dict) or set(state) != set(fields)
            or any(not isinstance(v, str) for v in state.values())):
        raise ValueError('终态场景信息结构无效')
    state = dict(state)
    if not state.get('location', '').strip() or not state.get('subject', '').strip():
        raise ValueError('缺少可交接的场景或主体观察')
    if catalog.get('intent_policy') in ('source-goals-v1', 'goal-allocation-v1'):
        terminal = value.get('terminal_observation')
        if (not isinstance(terminal, dict) or set(terminal) != {'sample_index', 'status', *DYNAMIC_STATE}
                or type(terminal['sample_index']) is not int or terminal['sample_index'] != sample_count
                or terminal['status'] not in ('observed', 'unknown')
                or any(not isinstance(terminal[k], str) or len(terminal[k]) > 1600 for k in DYNAMIC_STATE)
                or not terminal['action_phase'].strip()):
            raise ValueError('片尾快照必须来自实际末帧且足以交接，不能使用中途事件或猜测状态')
        if terminal['status'] == 'unknown':
            unresolved.append('terminal_observation')
        state.update({k: terminal[k] for k in DYNAMIC_STATE})
    else:
        links = value.get('end_state_sources')
        if not isinstance(links, dict) or set(links) != set(DYNAMIC_STATE):
            raise ValueError('动态终态缺少同源观察引用')
        for field, sources in links.items():
            if (not isinstance(sources, list) or any(not isinstance(k, str) for k in sources)
                    or len(sources) > 1 or (field == 'action_phase' and not sources)
                    or any(k not in observations or sample_count not in observations[k].sample_indices
                           or observations[k].status == 'unknown' for k in sources)):
                raise ValueError('交接状态必须引用包含末帧的同一观察，不得复制计划或早期状态')
            state[field] = '; '.join(observations[k].observation for k in sources)
    result = deepcopy(value)
    for key in ('observations', 'end_state_sources', 'end_state_context', 'terminal_observation'):
        result.pop(key, None)
    failed = [k for k, w in observations.items() if w.status == 'violated' and criteria[k]['kind'] != 'realization']
    result.update(accepted=not failed, primary_action_completed=not failed,
                  narrative_state_change_valid=not failed, observed_end_state=state,
                  failure_reason='; '.join(f'{k}: {observations[k].observation}' for k in failed))
    # 边界判断与动作完成分开，但所有必要问题都必须通过同源事实核对。
    domains = {'continuity': 'boundary_continuity_valid', 'camera': 'composition_change_valid'}
    for key, criterion in criteria.items():
        if criterion.get('domain') in domains:
            field = domains[criterion['domain']]
            result[field] = (False if observations[key].status == 'violated' else result.get(field, True))
    action_failed = [k for k in failed if criteria[k].get('domain') not in domains]
    result['primary_action_completed'] = not action_failed
    result['narrative_state_change_valid'] = not action_failed
    for event in result.get('action_event_evidence', []):
        if not isinstance(event, dict) or 'result_visible' in event or type(event.get('event_id')) is not int:
            raise ValueError('事件结果只能从已绑定观察推导')
        owned = []
        for key, criterion in criteria.items():
            if criterion['beat_id'] == event['event_id']:
                sources = criterion['supports'] if criterion['kind'] == 'realization' else key
                owned.extend(sources if isinstance(sources, list) else [sources])
        if not owned:
            raise ValueError('事件缺少动作观察来源')
        event['result_visible'] = all(observations[k].status == 'satisfied' for k in owned)
    if unresolved:
        # unknown 是协议内的正常回答；必须先排除其他字段、画面地址与终态的
        # 结构/来源错误，才可归因为语义未决，不能用它遮蔽损坏的响应。
        from jsonschema import Draft202012Validator
        if not Draft202012Validator(schema).is_valid(value):
            raise ValueError('观察响应结构与当前协议不符')
        raise ObservationUncertaintyError(unresolved)
    return result
