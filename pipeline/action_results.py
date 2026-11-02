"""Result data and pure projections used by the shared Action Contract.

The author may describe several facts about one entity. Pending claims remain
lossless review inputs; historical obligations keep their original partitioning.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


StateText = Annotated[str, Field(min_length=1, max_length=1600)]


class SceneChangeSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True, hide_input_in_errors=True)
    entity: StateText
    state: StateText
    importance: Literal['required', 'incidental']


class AuthoredResolution(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True, hide_input_in_errors=True)
    target_state: StateText
    actor_state: StateText | None
    scene_changes: list[SceneChangeSpec] = Field(max_length=14)


class EntityState(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True, hide_input_in_errors=True)
    state: StateText
    importance: Literal['required', 'incidental']


class EntityResolution(BaseModel):
    """结果以固定实体为主键，归属不随本次动作的主客体角色切换。"""
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)
    states: dict[StateText, Annotated[list[EntityState], Field(min_length=1, max_length=16)]] = Field(
        min_length=1, max_length=16)

    @model_validator(mode='after')
    def existing_claim_budget(self):
        if sum(len(claims) for claims in self.states.values()) > 16:
            raise ValueError('结果陈述总数不能超过已有的 16 项容量')
        return self


class CandidateState(EntityState):
    # 兼容旧响应的作者注释；新创作无需决定验收义务。
    importance: Literal['required', 'incidental'] | None = Field(
        default=None, description='旧响应的作者注释，可省略；必要性由独立审查决定')


class CandidateResolution(EntityResolution):
    states: dict[StateText, Annotated[list[CandidateState], Field(min_length=1, max_length=16)]] = Field(
        min_length=1, max_length=16)


def entity_resolution_schema(entities: list[str]) -> dict:
    schema = CandidateResolution.model_json_schema()
    item = schema['properties']['states']['additionalProperties']
    schema['properties']['states'] = {
        'type': 'object', 'properties': {entity: item for entity in entities},
        'additionalProperties': False, 'minProperties': 1, 'maxProperties': 16}
    return schema


def lower_authored_resolution(actor: str, target: str, resolution: AuthoredResolution | EntityResolution,
                              *, pending_review: bool = False) -> dict:
    """One source claim becomes one execution claim, including repeated owners.

    Pending claims all reach independent review, including author-labelled details.
    Their container is not approval. Historical approved inputs retain partitioning.
    Both input formats retain the existing sixteen-claim total and fourteen-item
    scene projection capacity. Legacy relative-role fields keep their meaning.
    """
    if isinstance(resolution, CandidateResolution) and not pending_review:
        raise ValueError('创作候选结果必须进入独立必要性审查，不能直接作为执行义务')
    if isinstance(resolution, EntityResolution):
        claims = [(f'resolution/states/{entity.replace("~", "~0").replace("/", "~1")}/{i}',
                   entity, claim.state, claim.importance)
                  for entity, states in resolution.states.items() for i, claim in enumerate(states)]
        if not any(entity == target and (pending_review or importance == 'required')
                   for _, entity, _, importance in claims):
            raise ValueError('结果候选缺少本次作用目标的状态；未补造结果' if pending_review
                             else '结果必须包含本次作用目标的必需状态；未补造结果')
    else:
        # 历史相对角色合同按原义回放；不能用正文中的名字猜测并修复归属。
        claims = [('resolution/target_state', target, resolution.target_state, 'required')]
        if resolution.actor_state is not None:
            claims.append(('resolution/actor_state', actor, resolution.actor_state, 'required'))
        claims.extend((f'resolution/scene_changes/{i}', c.entity, c.state, c.importance)
                      for i, c in enumerate(resolution.scene_changes))
    result = {'outcomes': [], 'scene_changes': [], 'source_map': []}
    for source, entity, state, importance in claims:
        field = 'outcomes' if (pending_review or importance == 'required') and entity in {actor, target} else 'scene_changes'
        result['source_map'].append({'source': source, 'destination': f'{field}/{len(result[field])}',
                                     'entity': entity, 'importance': importance})
        result[field].append({'entity': entity, 'state': state,
                              **({'importance': 'required' if pending_review else importance}
                                 if field == 'scene_changes' else {})})
    if len(result['scene_changes']) > 14:
        raise ValueError('场景变化超出既有执行合同的 14 项容量；未截断结果')
    return result


def result_claims(beat: dict) -> dict[str, dict]:
    return {**{f'outcomes/{i}': dict(c) for i, c in enumerate(beat.get('outcomes') or [])},
            **{f'scene_changes/{i}': {'entity': c['entity'], 'state': c['state']}
               for i, c in enumerate(beat.get('scene_changes') or []) if c['importance'] == 'required'}}


def required_outcomes(beat: dict) -> list[dict]:
    directions = set(beat.get('result_directions') or [])
    return [claim for path, claim in result_claims(beat).items() if path not in directions]


def directed_outcomes(beat: dict) -> list[dict]:
    directions = set(beat.get('result_directions') or [])
    return [claim for path, claim in result_claims(beat).items() if path in directions]


def incidental_changes(beat: dict) -> list[dict]:
    return [*({'entity': change['entity'], 'state': change['state']}
              for change in beat.get('scene_changes') or [] if change['importance'] == 'incidental'),
            *(claim for claim in directed_outcomes(beat)
              if claim['entity'] not in {beat.get('actor'), beat.get('target')})]


def scene_change_issues(shot: dict) -> list[str]:
    """Validate new result references at both persistence and direct compilation."""
    beats = [b for b in shot.get('action_beats', []) if isinstance(b, dict)
             and (b.get('scene_changes') is not None or b.get('result_directions') is not None)]
    if not beats:
        return []
    slot = shot.get('production_slot') or {}
    if slot.get('choreography') != 'flexible' or shot.get('motion_reference') or shot.get('motion_plan'):
        return ['场景结果分类仅用于自由编舞；精确动作或骨架参考不能忽略这些约束']
    registry = set(shot.get('characters', [])) | set(shot.get('key_props', [])) | set(shot.get('continuity_props', []))
    issues = []
    for beat in beats:
        directions = beat.get('result_directions')
        if directions is not None and (
                shot.get('result_admission') != 'reviewed-results-v1'
                or not isinstance(directions, list)
                or any(not isinstance(p, str) for p in directions)
                or len(directions) != len(set(directions))
                or not set(directions) <= set(result_claims(beat))):
            issues.append('表演建议必须来自完整审查后的原始结果陈述')
        changes = beat.get('scene_changes')
        if beat.get('activity') != 'interaction' or not beat.get('outcomes'):
            issues.append('场景结果必须属于具有明确目标结果的交互')
        if not isinstance(changes, list) or len(changes) > 14:
            issues.append('场景结果必须是有界列表')
            continue
        for raw in changes:
            try:
                change = SceneChangeSpec.model_validate(raw)
            except ValueError:
                issues.append('场景结果字段无效')
                continue
            if change.importance == 'required' and change.entity in {beat.get('actor'), beat.get('target')}:
                issues.append('动作双方的必需结果必须由编译器归入 outcomes')
            if change.entity not in registry:
                issues.append('场景结果引用了未注册的对象')
    return issues
