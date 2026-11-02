"""Compile semantic action units into the existing executable Action Contract.

The author owns causes, reactions and staging. This boundary owns derived timing,
phase and repeated event projections; it never interprets or rewrites action prose.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline.models import CameraSpec, CharacterBlockingSpec, ContinuityStateSpec, VisibleOutcomeSpec
from pipeline.action_results import AuthoredResolution, EntityResolution, CandidateResolution, lower_authored_resolution
from pipeline.action_timing import MIN_INTERACTION_MS, MIN_PASSIVE_MS, passive_floor_ms


# Creative descriptions share a prose budget. IDs, lists and numeric execution
# fields retain their own constraints; an aesthetic description is not an ID.
Description = Annotated[str, Field(max_length=1600)]
Text = Annotated[Description, Field(min_length=1)]
NonBlank = Annotated[str, Field(min_length=1, pattern=r'\S')]


def format_authoring_slots(plan: dict) -> str:
    fields = ('shot_id', 'duration', 'narrative_function', 'framing_family',
              'min_interaction_events', 'min_interaction_seconds', 'result_hold_seconds', 'required_outcome')
    return json.dumps([{k: slot[k] for k in fields if k in slot} for slot in plan['slots']], ensure_ascii=False)


class AuthoredModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True, hide_input_in_errors=True)


class Effect(AuthoredModel):
    mode: Literal['direct_contact', 'directed_path', 'area_effect', 'indirect_effect']
    motion: Literal['static', 'sweep', 'expand', 'propagate']
    scope: Literal['single', 'subset', 'all'] = 'single'
    source: Text
    region: Text
    unaffected: Text


class ActionUnit(AuthoredModel):
    id: Annotated[str, Field(min_length=1, max_length=80)]
    actor: Text
    action: Text
    result: Text
    weight: Annotated[float, Field(gt=0, le=100, strict=True, allow_inf_nan=False)] = 1.


class InteractionUnit(ActionUnit):
    kind: Literal['interaction']
    target: Text
    effect: Effect
    response: Literal['impact', 'evasion', 'deflection'] = 'impact'
    carry: str = Field(default='', max_length=800)
    link_previous: bool | None = Field(default=None, strict=True)
    overlap: bool = Field(default=False, strict=True)
    outcomes: list[VisibleOutcomeSpec] | None = Field(default=None, min_length=1, max_length=16)
    resolution: AuthoredResolution | EntityResolution | CandidateResolution | None = None

    @model_validator(mode='after')
    def one_result_authority(self):
        if self.link_previous is not None and self.carry:
            raise ValueError('动作承接只能使用前项绑定或旧 carry，不能同时声明两份原因')
        if self.resolution is not None:
            if self.outcomes is not None:
                raise ValueError('不能同时提供旧 outcomes 和角色绑定的 resolution')
        return self


class PassiveUnit(ActionUnit):
    kind: Literal['transition', 'hold']
    target: str = Field(default='', max_length=1600)


Unit = Annotated[InteractionUnit | PassiveUnit, Field(discriminator='kind')]


class AuthoredShot(AuthoredModel):
    shot_id: int = Field(gt=0, strict=True)
    scene_id: Text
    scene_description: Text
    visual_description: Text
    camera: CameraSpec
    blocking: dict[str, CharacterBlockingSpec]
    start_state: ContinuityStateSpec
    end_state: ContinuityStateSpec
    story_change: Text
    story_state_after: Text
    key_props: list[str] = Field(default_factory=list, max_length=40)
    continuity_props: list[str] = Field(default_factory=list, max_length=40)
    lighting: Description = ''
    mood: Description = 'cinematic'
    action_plan: list[Unit] = Field(min_length=1, max_length=12)
    result_admission: Literal['review-required-v1'] | None = None

    @model_validator(mode='after')
    def unique_causes(self):
        ids = [unit.id for unit in self.action_plan]
        if len(ids) != len(set(ids)):
            raise ValueError('同一动作单元不能重复计数；结果必须属于其原因')
        return self


class AuthoredContinuityState(ContinuityStateSpec):
    """新创作必须给出既有执行器依赖的事实；旧存档仍用宽松状态类型。"""
    location: NonBlank
    subject: NonBlank
    action_phase: NonBlank
    camera: NonBlank


class CreationShot(AuthoredShot):
    """创建请求与本地解析共用必填性，历史编译继续使用 AuthoredShot。"""
    start_state: AuthoredContinuityState
    end_state: AuthoredContinuityState


def authored_shot_response_schema(names: list[str], *, require_outcomes: bool = False,
                                 scene_entities: list[str] | None = None,
                                 bound_handoffs: bool = False) -> dict:
    """Project the same response/mode contract into constrained authoring."""
    from pipeline.action_timing import RESPONSE_MODES
    schema = CreationShot.model_json_schema()
    schema['properties'].pop('shot_id')
    schema['properties'].pop('result_admission')
    schema['required'].remove('shot_id')
    definitions = schema.pop('$defs')
    # 此函数仍描述历史数组接口；实体结果只由新槽位接口投影，避免改变精确编舞请求。
    definitions.pop('EntityResolution', None)
    definitions.pop('EntityState', None)
    definitions.pop('CandidateResolution', None)
    definitions.pop('CandidateState', None)
    interaction = definitions['InteractionUnit']
    if bound_handoffs:
        interaction['properties'].pop('carry')
        interaction['properties']['link_previous'] = {'type': 'boolean'}
        interaction['required'].append('link_previous')
    else:
        interaction['properties'].pop('link_previous')
    for name in ('InteractionUnit', 'PassiveUnit'):
        definitions[name]['properties']['actor']['enum'] = names
    if scene_entities is not None:
        entities = [*names, *scene_entities]
        definitions['InteractionUnit']['properties']['target']['enum'] = entities
        definitions['PassiveUnit']['properties']['target']['enum'] = ['', *entities]
        # 与执行端同一实体目录；动作描述属于 action/result，不是实体引用。
        definitions['CharacterBlockingSpec']['properties']['action_target']['enum'] = ['', *entities]
        definitions['SceneChangeSpec']['properties']['entity']['enum'] = entities
        for field in ('key_props', 'continuity_props'):
            schema['properties'][field]['items']['enum'] = entities
    if require_outcomes:
        definitions['InteractionUnit']['properties'].pop('outcomes')
        definitions['InteractionUnit']['properties']['resolution'] = {'$ref': '#/$defs/AuthoredResolution'}
        definitions['InteractionUnit']['required'].append('resolution')
    else:
        definitions['InteractionUnit']['properties'].pop('resolution')
    variants = []
    for response, modes in RESPONSE_MODES.items():
        variant = deepcopy(definitions['InteractionUnit'])
        variant['properties']['response'] = {'const': response, 'type': 'string'}
        variant['required'].append('response')
        effect = deepcopy(definitions['Effect'])
        effect['properties']['mode']['enum'] = list(modes)
        variant['properties']['effect'] = effect
        variants.append(variant)
    definitions['InteractionUnit'] = {'oneOf': variants}
    first_variants = deepcopy(variants)
    for variant in first_variants:
        variant['properties']['overlap'] = {'type': 'boolean', 'const': False}
        if bound_handoffs:
            variant['properties']['link_previous'] = {'type': 'boolean', 'const': False}
    # 第一单元可承接前镜运动，但本镜时间轴内不存在可搭接的前项。
    schema['properties']['action_plan']['prefixItems'] = [
        {'oneOf': [*first_variants, {'$ref': '#/$defs/PassiveUnit'}]},
    ]
    return {'type': 'object', 'properties': {'shot': schema}, 'required': ['shot'],
            'additionalProperties': False, '$defs': definitions}


def _apportion(total: int, weights: list[float], floors: list[int]) -> list[int]:
    """Weighted milliseconds with lower bounds and exact largest remainders."""
    if total < sum(floors) or not weights or len(weights) != len(floors):
        raise ValueError('不可执行的时间预算')
    values = [0] * len(weights)
    free = list(range(len(weights)))
    while free:
        scale = sum(weights[i] for i in free)
        raw = {i: total * weights[i] / scale for i in free}
        bounded = [i for i in free if raw[i] < floors[i]]
        if bounded:
            for i in bounded:
                values[i] = floors[i]
                total -= floors[i]
                free.remove(i)
            continue
        for i in free:
            values[i] = int(raw[i])
        remainder = total - sum(values[i] for i in free)
        for i in sorted(free, key=lambda i: (-(raw[i] - values[i]), i))[:remainder]:
            values[i] += 1
        break
    return values


def compile_action_units(shot: AuthoredShot, slot: dict) -> list[dict]:
    pending_results = shot.result_admission == 'review-required-v1'
    if pending_results and slot.get('choreography') != 'flexible':
        raise ValueError('候选结果审查仅用于自由编舞，不能改写精确执行合同')
    units = shot.action_plan
    events = [i for i, unit in enumerate(units) if isinstance(unit, InteractionUnit)]
    if len(events) < slot.get('min_interaction_events', 0):
        raise ValueError(f'Shot {shot.shot_id}: 独立作用不足，不能把反应复制成新的作用')
    total = int(slot['duration'] * 1000)
    hold = int(round(slot.get('result_hold_seconds', 0) * 1000))
    if hold and units[-1].kind != 'hold':
        raise ValueError(f'Shot {shot.shot_id}: 结局缺少持续可见的结果单元')
    widths = [MIN_INTERACTION_MS if i in events else MIN_PASSIVE_MS for i in range(len(units))]
    if hold:
        widths[-1] = max(widths[-1], hold)
    minimum = int(round(slot.get('min_interaction_seconds', 0) * 1000))
    passive = [i for i in range(len(units)) if i not in events]
    minimum = max(minimum, sum(widths[i] for i in events))
    available = total - sum(widths[i] for i in passive)
    if minimum > available or sum(widths) > total:
        raise ValueError(f'Shot {shot.shot_id}: 动作与结果在镜头时长内不可执行')
    weighted = round(total * sum(units[i].weight for i in events) / sum(u.weight for u in units))
    event_budget = min(available, max(minimum, weighted))
    for group, budget in ((events, event_budget), (passive, total - event_budget)):
        if group:
            allocated = _apportion(budget, [units[i].weight for i in group], [widths[i] for i in group])
            for i, value in zip(group, allocated):
                widths[i] = value
    beats, cursor = [], 0
    for i, (unit, width) in enumerate(zip(units, widths)):
        beat = {'actor': unit.actor, 'target': unit.target, 'action': unit.action,
                'visible_result': unit.result, 'activity': unit.kind,
                'phase': 'peak' if unit.kind == 'interaction' else ('trigger' if unit.kind == 'transition' else 'aftermath')}
        start, end = cursor, cursor + width
        if isinstance(unit, InteractionUnit):
            if unit.resolution is not None:
                lowered = lower_authored_resolution(unit.actor, unit.target, unit.resolution,
                                                   pending_review=pending_results)
                beat.update({key: lowered[key] for key in ('outcomes', 'scene_changes')})
            if unit.outcomes is not None:
                entities = [outcome.entity for outcome in unit.outcomes]
                if (len(entities) != len(set(entities)) or unit.target not in entities
                        or not set(entities) <= {unit.actor, unit.target}):
                    raise ValueError(f'Shot {shot.shot_id}: 可见结果必须包含目标且只能归属于本次作用的参与者')
                beat['outcomes'] = [outcome.model_dump() for outcome in unit.outcomes]
            previous = units[i - 1] if i else None
            linked = bool(previous and {previous.actor, previous.target} & {unit.actor, unit.target})
            carry = unit.carry
            if unit.link_previous is not None:
                if unit.link_previous and not linked:
                    raise ValueError(f'Shot {shot.shot_id}: 前项绑定需要本镜中有共同参与者的前一动作')
                # 顺序已唯一确定来源；运动细节只由原 action 表达，不再生成第二份因果文案。
                carry = (f'Beat {i} -> Beat {i + 1}: continue the preceding movement into this action '
                         'without an unexplained pause or reset.' if unit.link_previous else '')
            if unit.overlap:
                if not isinstance(previous, InteractionUnit) or not carry or not linked:
                    raise ValueError(f'Shot {shot.shot_id}: 搭接必须来自相邻共同参与者的明确运动承接')
                start -= min(widths[i - 1], width) // 5
            if carry and i:
                if not linked:
                    raise ValueError(f'Shot {shot.shot_id}: 运动承接缺少共同参与者')
                beat['flow_from_previous'] = carry
            effect = unit.effect
            geometry = {'actor': unit.actor, 'target': unit.target, 'effect_phase': 'active',
                        'interaction_mode': effect.mode, 'outcome_scope': effect.scope,
                        'effect_motion': effect.motion, 'source': effect.source,
                        'effect_region': effect.region, 'reaction_scope': unit.result,
                        'unaffected_behavior': effect.unaffected,
                        'must_share_frame': effect.mode == 'direct_contact',
                        'line_of_action_visible': effect.mode == 'directed_path',
                        'occlusion_policy': 'none'}
            for role, entity in [('actor', unit.actor), ('target', unit.target)]:
                staging = shot.blocking.get(entity)
                # Objects may have camera placement without human body blocking.
                # Missing placement stays unspecified; effect.region is not a
                # screen coordinate and cannot supply an invented position.
                geometry[f'{role}_screen_position'] = (
                    staging.frame_position if staging and staging.frame_position
                    else shot.camera.screen_positions.get(entity, ''))
            beat.update(interaction_geometry=geometry, response_kind=unit.response,
                        effect_at_seconds=(start + (end - start) * 2 // 5) / 1000,
                        reaction_at_seconds=(start + (end - start) * 7 // 10) / 1000)
        beat.update(start_seconds=start / 1000, end_seconds=end / 1000)
        beats.append(beat)
        cursor = end
    return beats


def compile_authored_storyboard(raw: dict, plan: dict) -> dict:
    """A pure compilation; no repair, phrase matching, model call or mutation."""
    if not isinstance(raw, dict) or not isinstance(raw.get('shots'), list):
        raise ValueError('创作稿必须包含 shots 列表')
    slots = plan['slots']
    if len(raw['shots']) != len(slots):
        raise ValueError('创作稿镜头数与生产计划不一致')
    result = {key: deepcopy(value) for key, value in raw.items() if key != 'shots'}
    result['shots'] = []
    known = {c['name'] for c in raw.get('characters', [])}
    prior_state = (raw.get('story_arc') or {}).get('goal', '')
    authored_shots = []
    for raw_shot, slot in zip(raw['shots'], slots):
        shot = AuthoredShot.model_validate(raw_shot)
        pending_results = shot.result_admission == 'review-required-v1'
        if pending_results and slot.get('choreography') != 'flexible':
            raise ValueError('候选结果审查仅用于自由编舞，不能改写精确执行合同')
        authored_shots.append(shot)
        if shot.shot_id != slot['shot_id']:
            raise ValueError('创作稿镜头 ID 与生产计划不一致')
        registry = known | set(shot.key_props) | set(shot.continuity_props)
        participants = []
        visible_entities = []
        for unit in shot.action_plan:
            if unit.actor not in known or (unit.target and unit.target not in registry):
                raise ValueError(f'Shot {shot.shot_id}: 未注册的动作参与者')
            participants.extend(e for e in (unit.actor, unit.target) if e in known)
            # Identity references apply to characters; action visibility applies
            # to every interaction target, including registered scene objects.
            # Merely registering a prop does not make it a required subject.
            visible_entities.append(unit.actor)
            if isinstance(unit, InteractionUnit):
                visible_entities.append(unit.target)
            if isinstance(unit, InteractionUnit) and unit.resolution is not None:
                claims = lower_authored_resolution(unit.actor, unit.target, unit.resolution,
                                                  pending_review=pending_results)['source_map']
                for change in claims:
                    if change['entity'] not in registry:
                        raise ValueError(f'Shot {shot.shot_id}: 场景结果引用了未注册的对象')
                    if change['entity'] in known:
                        participants.append(change['entity'])
        beats = compile_action_units(shot, slot)
        events = [b for b in beats if b['activity'] == 'interaction']
        if not events:
            raise ValueError('事件生产槽位缺少真实作用')
        # Cause/reaction text is projected once; no independent action-bearing prompt to drift.
        prompt = shot.visual_description + ' ' + ' '.join(f'{u.action}. {u.result}.' for u in shot.action_plan)
        compiled = shot.model_dump(exclude={'action_plan', 'story_change', 'story_state_after', 'result_admission'})
        if pending_results:
            compiled['result_admission'] = shot.result_admission
        if slot.get('choreography') == 'flexible':
            compiled['generation_direction'] = 'authored-staging-v1'
            compiled['media_selection'] = 'planned-prefix-v1'
        compiled.update(duration=slot['duration'], prompt_en=prompt, characters=list(dict.fromkeys(participants)),
                        primary_action='; '.join(u.action for u in shot.action_plan), action_beats=beats,
                        interaction_geometry=deepcopy(events[-1]['interaction_geometry']),
                        required_visible_entities=list(dict.fromkeys([*participants, *visible_entities])), coverage_role='interaction',
                        continuity_from_previous='none' if not result['shots'] else 'intentional_cut',
                        transition_to_next='cut', extract_character_ref=False,
                        narrative_beat={'function': slot['narrative_function'], 'state_before': prior_state,
                                        'state_change': shot.story_change, 'state_after': shot.story_state_after})
        first = shot.action_plan[0]
        if isinstance(first, InteractionUnit) and first.carry:
            compiled['start_state']['open_motion'] = first.carry
        result['shots'].append(compiled)
        prior_state = shot.story_state_after
    # Per-slot validation is provisional. Only the complete authored film can
    # allocate the film-wide floor, without turning weights into model math duties.
    if (plan.get('min_interaction_seconds') is not None
            and sum(s['duration'] for s in slots) == plan.get('planned_duration')):
        _allocate_film_timing(result['shots'], authored_shots, plan)
        for compiled, slot in zip(result['shots'], slots):
            if slot.get('choreography') == 'flexible':
                compiled['generation_rhythm'] = 'allocated-v1'
    return result


def _allocate_film_timing(compiled: list[dict], authored: list[AuthoredShot], plan: dict) -> None:
    from pipeline.film_rhythm import planned_interaction_seconds

    preferred, capacity = [], []
    for shot, source, slot in zip(compiled, authored, plan['slots']):
        preferred.append(round(planned_interaction_seconds([{**shot, 'production_slot': slot}]) * 1000))
        capacity.append(round(slot['duration'] * 1000) - passive_floor_ms(
            [unit.kind for unit in source.action_plan], slot.get('result_hold_seconds', 0)))
    deficit = max(0, round(plan['min_interaction_seconds'] * 1000) - sum(preferred))
    slack = [max(0, limit - desired) for limit, desired in zip(capacity, preferred)]
    if deficit > sum(slack):
        raise ValueError('全片交互预算与动作及收尾最低时长不可同时满足；未删除或改写动作')
    if not deficit:
        return
    eligible = [i for i, available in enumerate(slack) if available]
    extras = _apportion(deficit, [slack[i] for i in eligible], [0] * len(eligible))
    for i, extra in zip(eligible, extras):
        if extra:
            slot = plan['slots'][i]
            compiled[i]['action_beats'] = compile_action_units(authored[i], {
                **slot, 'min_interaction_seconds': (preferred[i] + extra) / 1000})
