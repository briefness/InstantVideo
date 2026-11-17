"""Author a frozen story, then fill each program-owned shot slot exactly once."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline.action_authoring import CreationShot, Description, NonBlank, InteractionUnit, authored_shot_response_schema, compile_authored_storyboard, format_authoring_slots
from pipeline.action_results import lower_authored_resolution
from pipeline.participants import iter_structured_entity_references
from pipeline.models import CharacterSpec, StoryArcSpec
from pipeline.authoring_contract import AuthoringContract, AuthoringContractError
from pipeline.planning_checkpoint import PlanningCheckpointError
from pipeline.operation_failure import add_secondary_failure
from pipeline.film_rhythm import planning_rhythm_context
from pipeline.requested_outcome import requested_ending
from pipeline.authoring_context import SceneLocation, bind_authored_context, contextual_shot_schema
from pipeline.authoring_schedule import scheduled_shot_schema, bind_authored_schedule
from pipeline import authoring_timing
from pipeline.planning_boundaries import authoring_progress
from pipeline.action_vocabulary import AUTHORING_SEMANTICS
from pipeline.completion_intent import source_goals


class AuthoredStoryArc(StoryArcSpec):
    """新故事事实没有可自动补出的默认值；历史 StoryArcSpec 保持兼容。"""
    goal: NonBlank
    stakes: NonBlank
    turning_point: NonBlank
    resolution: NonBlank


class StoryIdentity(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)
    title: str = Field(min_length=1, max_length=200)
    mood: Description = 'cinematic'
    music_style: Description = 'cinematic orchestral'
    theme_elements: list[str] = Field(min_length=1, max_length=30)
    story_arc: AuthoredStoryArc
    characters: list[CharacterSpec] = Field(min_length=1, max_length=16)
    scene_entities: list[Annotated[str, Field(min_length=1, max_length=128)]] | None = Field(default=None, max_length=40)
    locations: list[SceneLocation] | None = Field(default=None, min_length=1, max_length=150)
    shot_durations: list[Annotated[int, Field(strict=True, ge=4, le=15)]] | None = None

    @model_validator(mode='after')
    def distinct_scene_entities(self):
        if self.scene_entities is not None:
            if (len(self.scene_entities) != len(set(self.scene_entities))
                    or any(not e.strip() or e != e.strip() for e in self.scene_entities)
                    or set(self.scene_entities) & {c.name for c in self.characters}):
                raise ValueError('场景对象 ID 必须非空、唯一，且不能与角色 ID 重复')
        return self


class SceneStoryIdentity(StoryIdentity):
    """物体和环境可独立成为画面主体，不为无人镜头制造人物。"""
    characters: list[CharacterSpec] = Field(min_length=0, max_length=16)


def _bind_scene_entities(shot: dict, names: list[str], catalog: list[str], *, scene_units: bool = False) -> None:
    """Derive per-shot registrations from frozen IDs, never infer aliases."""
    known = set(names) | set(catalog)
    declared = [*shot.get('key_props', []), *shot.get('continuity_props', [])]
    references = []
    # 创作列表与执行列表仅容器名不同；复用执行端的实体字段遍历，避免局部清单漏掉空间引用。
    references_shot = {**shot, 'action_beats': [
        {**unit, 'activity': unit['kind']} for unit in shot.get('action_plan', [])]}
    if scene_units:
        references_shot['production_slot'] = {'authoring_protocol': 'scene-units-v1'}
    for reference in iter_structured_entity_references(references_shot):
        entity = reference.value
        # 执行遍历器也列出尚未编译的 geometry 槽位；必需创作字段由类型边界检查。
        if entity in ('', None):
            continue
        allowed = names if reference.kind == 'character' else known
        if not isinstance(entity, str) or entity not in allowed:
            raise ValueError(f'{reference.field} 引用了冻结故事中未注册的对象；不能猜测别名或临时新增对象')
        references.append(entity)
    for unit in shot.get('action_plan', []):
        if unit.get('resolution') is not None:
            parsed = InteractionUnit.model_validate(unit)
            references.extend(claim['entity'] for claim in lower_authored_resolution(
                parsed.actor, parsed.target, parsed.resolution,
                pending_review=shot.get('result_admission') == 'review-required-v1')['source_map'])
    if any(not isinstance(e, str) or e not in known for e in [*declared, *references]):
        raise ValueError('分镜引用了冻结故事中未注册的对象；不能猜测别名或临时新增对象')
    shot['key_props'] = list(dict.fromkeys([*shot.get('key_props', []),
        *(e for e in references if e in catalog)]))


def generate_event_storyboard(client, request: str, plan: dict, *, aspect_ratio: str,
                              resolution: str, style: str, save_diagnostic=None, planning_checkpoint=None) -> dict:
    from pipeline.storyboard import _call_llm_for_storyboard, _compile_storyboard_contract, _validate_storyboard_richness
    from pipeline.provider_capabilities import planning_execution_contract
    execution_contract = planning_execution_contract(resolution, aspect_ratio, plan)
    plan = deepcopy(plan)
    def author_call(key, system, user, **kwargs):
        contract = AuthoringContract(kwargs.pop('local_response_schema'), generation_constraints=True)
        kwargs['response_schema'] = contract.wire_schema
        def request_model():
            if save_diagnostic:
                from config import LLM_MODEL
                save_diagnostic(f'planning_{key}_contract.json', contract.record())
                save_diagnostic(f'planning_{key}_request.json', {
                    'model': LLM_MODEL, 'system': system, 'user': user,
                    **{k: deepcopy(kwargs[k]) for k in ('temperature', 'reasoning', 'response_schema') if k in kwargs},
                })
            response = _call_llm_for_storyboard(client, system, user, **kwargs)
            if save_diagnostic:
                save_diagnostic(f'planning_{key}.json', deepcopy(response))
            try:
                contract.validate(response)
            except AuthoringContractError as exc:
                if save_diagnostic:
                    try:
                        save_diagnostic(f'planning_{key}_contract_error.json', {
                            'version': contract.record()['version'], 'issues': exc.issues})
                    except Exception as secondary:
                        add_secondary_failure(exc, secondary, 'save_diagnostic')
                raise
            return response
        if planning_checkpoint is None:
            return request_model()
        return planning_checkpoint.step(key, {'system': system, 'user': user,
            'contract': contract.record(),
            **{k: v for k, v in kwargs.items() if k != 'save_response'}}, request_model)
    try:
        require_outcomes = any(s.get('choreography') == 'flexible' for s in plan['slots'])
        scene_units = any(s.get('authoring_protocol') == 'scene-units-v1' for s in plan['slots'])
        goals = source_goals({'production_plan': plan})
        goal_context = {'completion_goals': goals} if goals else {}
        identity_type = SceneStoryIdentity if scene_units else StoryIdentity
        identity_schema = identity_type.model_json_schema()
        flexible_time = scene_units and plan['director_intent'].get('duration_mode') in ('exact', 'approximate')
        time_context = {}
        if flexible_time:
            from pipeline.production_shape import duration_bounds
            lower, upper = duration_bounds(plan['target_duration'], plan['director_intent'])
            lower = max(lower, len(plan['slots']) * execution_contract['hard_limits']['min_duration'])
            upper = min(upper, len(plan['slots']) * execution_contract['hard_limits']['max_duration'])
            time_context = {'duration_budget': {'target_seconds':plan['target_duration'],
                'minimum_seconds':lower, 'maximum_seconds':upper, 'shot_count':len(plan['slots'])}}
            identity_schema['properties']['shot_durations'] = {'type':'array',
                'minItems':len(plan['slots']), 'maxItems':len(plan['slots']),
                'items':{'type':'integer', 'minimum':execution_contract['hard_limits']['min_duration'],
                         'maximum':execution_contract['hard_limits']['max_duration']}}
            identity_schema['required'].append('shot_durations')
        else:
            identity_schema['properties'].pop('shot_durations')
        ending = requested_ending(plan)
        if isinstance(ending, dict):
            # 故事摘要是冻结的意图文字；证据保留结构，不能塞进字符串 const，
            # 也不能重新拼成伪连续引文。正式结局审查仍读取完整来源段。
            goal_context['ending_evidence'] = ending
            ending = plan['director_intent']['ending_requirement']
        if ending:
            arc_type = identity_schema['properties']['story_arc']['$ref'].rsplit('/', 1)[-1]
            identity_schema['$defs'][arc_type]['properties']['resolution'] = {'type': 'string', 'const': ending}
        if require_outcomes:
            identity_schema['properties']['scene_entities'] = {
                'type': 'array', 'maxItems': 40, 'items': {'type': 'string', 'minLength': 1, 'maxLength': 128}}
            identity_schema['required'].append('scene_entities')
            identity_schema['properties']['locations'] = {
                'type': 'array', 'minItems': 1, 'maxItems': 150, 'items': {'$ref': '#/$defs/SceneLocation'}}
            identity_schema['required'].append('locations')
        else:
            identity_schema['properties'].pop('scene_entities')
            identity_schema['properties'].pop('locations')
            identity_schema['$defs'].pop('SceneLocation')
        identity = author_call(
            'identity', '你是电影故事设计师。只确定全片故事、角色与视觉身份，不写分镜。只输出符合下列schema的JSON。'
            '角色name使用稳定英文ID，description写清外观、服装与场景实际需要的随身物品；结局取自用户要求。\n'
            + ('story_arc.resolution已由用户原文固定；只设计满足该结果的目标、冲突与转折，实现方式留给分镜，不替换结果。\n' if ending else '')
            + ('按故事各段的内容负荷输出shot_durations；段数固定、各段时长可不同，总和在duration_budget内。优先贴近目标，不为凑秒数重复动作。当前slots的时长是初始预算，选定后程序重建并冻结，之后不再拉伸或截短。\n' if flexible_time else '')
            + ('scene_entities登记全片动作涉及的道具与环境对象，使用唯一、明确、跨镜稳定的ID；不重复角色，没有则空列表。\n'
               'locations登记实际物理地点及description：同一地点的不同机位、局部位置或镜头共用一个ID，只有真正换地点才新增。\n'
               if require_outcomes else '')
            + ('' if require_outcomes else json.dumps(identity_schema, ensure_ascii=False)),
            json.dumps({'request': request, 'style': style, 'slots': json.loads(format_authoring_slots(plan)),
                        'execution_capabilities':execution_contract, **time_context, **goal_context}, ensure_ascii=False),
            temperature=.3, repair_json=False, reasoning=True,
            save_response=(lambda value: save_diagnostic('planning_identity_response.json', value)) if save_diagnostic else None,
            local_response_schema=identity_schema)
        identity_model = identity_type.model_validate(identity)
        if flexible_time:
            from pipeline.production_plan import build_production_plan
            plan = build_production_plan(plan['content_focus'], plan['target_duration'],
                intent=plan['director_intent'], contract_version=plan['version'],
                duration_allocation=identity_model.shot_durations,
                duration_limits=plan.get('duration_limits'))
            execution_contract = planning_execution_contract(resolution, aspect_ratio, plan)
        if ending and identity_model.story_arc.resolution != ending:
            raise ValueError('故事设定改写了用户结局；未请求后续分镜，不自动修稿')
        catalog = identity_model.scene_entities
        if require_outcomes and catalog is None:
            raise ValueError('自由编舞的故事设定缺少 scene_entities 对象登记')
        locations = identity_model.locations
        if require_outcomes and (not locations or len({s.id for s in locations}) != len(locations)
                                 or any(not s.description.strip() for s in locations)):
            raise ValueError('自由编舞的故事设定必须登记唯一、非空的物理地点 locations')
        frozen = identity_model.model_dump(exclude={'scene_entities', 'locations', 'shot_durations'})
        names = [c['name'] for c in frozen['characters']]
        if scene_units and not names and not catalog:
            raise ValueError('故事必须登记人物或场景主体，不能编写无归属的表演')
        if len(names) != len(set(names)):
            raise ValueError('角色 ID 必须唯一')
        raw = {**frozen, 'shots': []}
        compiled_shots = []
        lowering_trace = []
        response_schema = authored_shot_response_schema(names, require_outcomes=require_outcomes,
                                                       scene_entities=catalog if require_outcomes else None,
                                                       bound_handoffs=require_outcomes,
                                                       allow_scene_subjects=scene_units)
        if require_outcomes:
            response_schema = contextual_shot_schema(response_schema, locations, catalog)
        handoff_instruction = (
            'link_previous表示本动作是否连续承接本镜紧邻的上一动作。程序绑定前后项，不另写carry文案；'
            '具体如何借用前项的运动写在当前action中。第一单元必须为false，跨镜入场由start_state.open_motion表达；'
            'overlap仅用于link_previous=true的相邻交互搭接。'
            if require_outcomes else
            'carry写上一动作留下的运动怎样成为下一动作的起因，而非仅说“承接上一招”：第一单元用于跨镜入场，其余承接前一单元；'
            'overlap仅用于相邻交互搭接。')
        if scene_units:
            handoff_instruction = authoring_timing.INSTRUCTION
        system = '''你是影视动作编导。输入是冻结故事、一个固定镜头槽位、前镜内容。
只输出 {"shot": {...}}，内容符合''' + ('响应' if require_outcomes else '下方') + '''schema。程序负责镜头ID、数量和精确时间，不输出shot_id或其他镜头。
''' + AUTHORING_SEMANTICS + '''action写实际行为，result写由它产生的完整可见变化。
先编排符合题材的完整连续表演，再用动作单元表达独立作用、移动或持续状态；不要为了镜头槽位或事件计数让人物逐次停住复位。具体运动应服务角色意图，同一人可以连续行动，不要求人物轮流表演。''' + handoff_instruction + '''镜头可以切在运动中，下一镜从同一运动接入；只有叙事确实需要时才停顿或收势。weight表达相对时间需要，程序编译阶段与时间窗。
visual_description描述场景、质感与构图，动作只在action_plan中表达一次。运镜服务于动作可读性，camera.primary_movement用fixed/tracking/dolly in/dolly out/pan left/pan right/tilt up/tilt down/truck left/truck right/crane up/crane down/orbit/handheld。blocking描述角色起始位置和朝向。
action写具体可执行的编舞，result写所选执行的完整可见效果；有outcomes时，必要状态由outcomes单独承担，不把result的细节升级为验收义务。原始需求里的指定动作仍必须保留；不要额外发明必须逐帧复现的姿态。一次连续借势变招可以包含流动身法，不把每次摆手都拆成独立交互。
''' + ('''自由编舞的resolution.states按固定实体ID记录结果候选。每个键是人物或物体ID，每条陈述只需state，无需决定importance或验收义务；必要性由独立审查结合完整需求和后续剧情判断。旧格式的importance仅作为作者注释保留，不能批准或排除任何结果。目标必须有实际可观察的结果陈述，其他实体只写相关变化；同一实体无论发起动作还是作出反应都使用原ID。保留完整结果，不为同一次作用的多个后果另造动作；具体路径、招式和时刻放action/result。程序不会替作者补造缺失状态。
''' if require_outcomes else '''自由编舞使用resolution表达独立验收状态：target_state是本动作目标必须达到的状态，actor_state为执行者必要状态（没有则null）；对象由程序绑定，不再重复填写名字。state写动作之后可观察的状态或能力，不写攻击路线、招式或时刻。scene_changes描述已注册对象的额外结果陈述，entity必须使用原ID；同一对象可以有多条陈述，程序保留每条陈述并整理归属，不会消除矛盾。importance=required表示用户要求或后续剧情依赖的必要结果，incidental仅表示可有可无的画面细节。不能将用户要求或下一镜依赖的状态降为incidental；没有额外变化时给空列表。不把同一次作用的多个后果编造成多次交互。动作如何实现放action/result，是否成功由response与必要结果共同判断。精确编舞仍使用原action/result合同。
''') + '''延续previous_shot的状态、身份与空间；is_final=false时不提前完成全片结局，is_final=true时完整呈现规定结局。
film_rhythm若存在，交互时长要求属于全片而非每镜同一比例。依据本镜的准备、发展、转折、结果任务设定action_plan的weight；前镜的内容不改写，权重是节奏偏好，程序在全稿完成后分配满足全片预算的时间。入场与停顿不能标成interaction，收尾需留足结果确认，不为凑时长复制动作。
''' + ('''scene_id只能选择locations中的已登记物理地点；机位变化不能新建场景。
prop_roles只从scene_entities对本镜涉及的对象声明一次用途，角色身份由characters管理：visible表示可见道具或布景；persistent表示跨镜必须跟踪、保留的叙事道具；visible_and_persistent兼具两者。不要把全部场景装饰当作持久道具；也不要遗漏用户要求或后续动作依赖的关键物件。程序投影key_props与continuity_props，不输出这两份列表。
''' if require_outcomes else json.dumps(response_schema, ensure_ascii=False))
        if any(slot.get('motion_timing') for slot in plan['slots']):
            from pipeline.motion_timing import AUTHORING_TIMING
            system += AUTHORING_TIMING
        if scene_units:
            system += ('\n人物不必出现在每镜。transition/hold的actor可为登记的场景对象；'
                       '纯产品、环境展示也按对象状态组织，不虚构人物或交互。'
                       'story_progress是已完成进度与剩余全片任务，current_boundary是前镜实际终态。'
                       '本镜从该进度继续，按剩余镜头预算安排未发生事件和持续表演；'
                       '换机位不会重置故事时间，各镜将完整顺序拼接，不会自动剪去重复过程。')
        if goals:
            system += ('\ncompletion_goals是创作前已冻结、与后续审查共用的全片目标目录；'
                       '结合原需求与story_progress把目标安排到完整影片中，不逐镜重复全部目标。'
                       '表演和构图是实现目标的方案，不能新增必要事件，也不能替代目录中的明确要求。')
        if any(slot.get('creative_direction') for slot in plan['slots']):
            from pipeline.creative_direction import AUTHORING_DIRECTION
            system += AUTHORING_DIRECTION
        previous_creation = None
        all_slots = json.loads(format_authoring_slots(plan)) if scene_units else None
        for index, slot in enumerate(plan['slots']):
            slot_schema = scheduled_shot_schema(response_schema, slot, explicit_absence=scene_units) if require_outcomes else response_schema
            if scene_units:
                slot_schema = authoring_timing.execution_timing_schema(slot_schema, version=authoring_timing.VERSION)
            slot_system = system
            if require_outcomes and not slot.get('min_interaction_events'):
                slot_system += (
                    '\naction_plan.first填写首个动作单元，following按实际顺序填写其余单元，可以为空。'
                    '每个单元填写kind，不填id；无最低交互数时可以使用performance/transition/hold，也可以包含实际interaction。'
                    'transition表达没有独立物理作用的移动或状态转变，hold表达持续展示；不要把普通表演改造成攻击。'
                    '有result_hold时在末尾填写持续结果，不填id/kind。总动作单元最多12个，不补写或删减动作。')
            elif require_outcomes:
                slot_system += (
                    '\naction_plan由本槽位的既定工作绑定：按opening、interaction_1至最后一项、result_hold顺序表演。'
                    + ('没有准备段时opening为null（也可省略）；' if scene_units else '没有准备段时省略opening；') +
                    '有准备段时opening.first填写首个动作，opening.following按顺序填写其余准备动作。'
                    '各interaction键填写独立作用、完整反应及resolution，不填写id/kind。'
                    '每项following可为空，也可填写额外交互、过渡或停留（填写kind，不填id），按其数组顺序紧接该项。'
                    '有result_hold时填写结局已建立后的展示内容，不填写id/kind。程序绑定身份和顺序，不补写表演。'
                    + ('timing针对展开后紧邻的动作；同一人可连续作用，不要求双方交替。'
                       if scene_units else 'link_previous/overlap针对展开后紧邻的动作；同一人可连续作用，不要求双方交替。') +
                    'setup/turn/payoff是叙事作用，并不替代本槽位已有的交互任务。所有动作的精确时间仍由全片分配。')
            context = {'request': request, 'format': {'aspect_ratio': aspect_ratio, 'resolution': resolution, 'style': style},
                       'execution_capabilities':execution_contract,
                       'story': frozen, 'slot': json.loads(format_authoring_slots({'slots': [slot]}))[0],
                       'is_final': index == len(plan['slots']) - 1,
                       'film_rhythm': planning_rhythm_context(plan, compiled_shots),
                       'previous_shot': (previous_creation if scene_units else
                                         raw['shots'][-1] if raw['shots'] else None)}
            if require_outcomes:
                context['scene_entities'] = catalog
                context['locations'] = [s.model_dump() for s in locations]
                context['authoring_protocol'] = authoring_timing.VERSION if scene_units else 'story-context-v6-candidate-results'
            if scene_units:
                context['story_progress'] = authoring_progress(plan, raw['shots'], all_slots)
                context.update(deepcopy(goal_context))
            response = author_call(f'shot_{slot["shot_id"]:03d}', slot_system, json.dumps(context, ensure_ascii=False),
                                                temperature=.5, repair_json=False, reasoning=True,
                                                save_response=(lambda value: save_diagnostic(f'planning_shot_{slot["shot_id"]:03d}_response.json', value)) if save_diagnostic else None,
                                                local_response_schema=slot_schema)
            if set(response) != {'shot'} or not isinstance(response['shot'], dict) or 'shot_id' in response['shot']:
                raise ValueError(f'Shot {slot["shot_id"]}: 必须只返回本槽位创作内容，镜头ID由程序绑定')
            shot = {'shot_id': slot['shot_id'], **response['shot']}
            if require_outcomes:
                if scene_units:
                    shot = authoring_timing.bind_execution_timing(shot, version=authoring_timing.VERSION)
                shot = bind_authored_schedule(shot, slot, explicit_absence=scene_units)
                shot = bind_authored_context(shot, locations, catalog)
            # 创建要求与响应 schema 来自同一类型；不依赖提供方遵守约束。
            CreationShot.model_validate(shot)
            if require_outcomes:
                _bind_scene_entities(shot, names, catalog, scene_units=scene_units)
            raw['shots'].append(shot)
            if save_diagnostic:
                save_diagnostic('planning_authoring.json', deepcopy(raw))
            # Validate each slot before requesting the next; no failed slot is regenerated.
            compiled = compile_authored_storyboard({**frozen, 'shots': [shot]}, {**plan, 'slots': [slot]})
            compiled_shots.extend(compiled['shots'])
            previous_creation = {'shot_id': slot['shot_id'], **deepcopy(response['shot'])}
            if save_diagnostic:
                for i, unit in enumerate(shot['action_plan']):
                    if unit.get('resolution') is not None:
                        lowered = lower_authored_resolution(unit['actor'], unit['target'],
                                                            InteractionUnit.model_validate(unit).resolution,
                                                            pending_review=shot.get('result_admission') == 'review-required-v1')
                        lowering_trace.append({'shot_id': slot['shot_id'], 'beat_index': i,
                                               'unit_id': unit['id'], 'claims': lowered['source_map']})
                save_diagnostic('planning_lowering.json', {'version': 'authoring-lowering-v2-candidates',
                                                          'units': deepcopy(lowering_trace)})
        authored = compile_authored_storyboard(raw, plan)
        storyboard = _compile_storyboard_contract(authored, aspect_ratio, resolution, style, production_plan=plan)
        if require_outcomes:
            for shot in storyboard['shots']:
                shot['result_admission'] = 'review-required-v1'
        issues, critical = _validate_storyboard_richness(storyboard, user_request=request, require_narrative_contract=True)
    except PlanningCheckpointError:
        # 恢复没有重新审查响应，不能用准入拒绝覆盖首次内容失败现场。
        raise
    except AuthoringContractError as exc:
        if save_diagnostic:
            try:
                save_diagnostic('planning_check.json', {'critical': True, 'stage': 'response_contract', 'issues': [str(exc)]})
            except Exception as secondary:
                add_secondary_failure(exc, secondary, 'save_diagnostic')
        raise
    except (ValueError, TypeError, KeyError) as exc:
        if save_diagnostic:
            save_diagnostic('planning_check.json', {'critical': True, 'issues': [str(exc)]})
        raise ValueError(f'动作创作稿未通过编译；未重写原稿、未调用视频接口：{exc}') from exc
    if save_diagnostic:
        save_diagnostic('planning_compiled.json', storyboard)
        save_diagnostic('planning_check.json', {'critical': critical, 'issues': issues})
    if critical:
        raise ValueError('动作创作稿未通过就绪校验；未重写原稿、未调用视频接口：' + '；'.join(issues))
    for issue in issues:
        print(f'   {issue}')
    return storyboard
