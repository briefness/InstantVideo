"""Evidence-bound semantic admission before paid execution; never repairs a plan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

import config
from pipeline.llm_options import ark_output_schema
from pipeline.result_admission import (
    ResultAdmission, CLASSIFICATION_INSTRUCTION, INSTRUCTION, admission_issues, apply_result_admission,
)
from pipeline import planning_review_protocol as protocol
from pipeline.planning_review_sources import continuity_sources
from pipeline import requested_outcome as outcome


class UnitVerdict(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)
    shot_id: int = Field(strict=True, gt=0)
    beat_index: int = Field(strict=True, ge=0)
    cause_and_participants_valid: bool = Field(strict=True)
    classification_matches_description: bool = Field(strict=True)
    outcome_contract_valid: bool | None = Field(default=None, strict=True)
    result_admission: ResultAdmission | None = None
    evidence_fields: list[Literal['actor', 'target', 'action', 'visible_result', 'activity',
                                  'response_kind', 'interaction_geometry', 'outcomes', 'scene_changes']] = Field(min_length=1, max_length=9)
    reason: str = Field(min_length=1, max_length=1600)


class PlanVerdict(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)
    units: list[UnitVerdict] = Field(min_length=1, max_length=120)
    continuity_valid: bool = Field(strict=True)
    requested_outcome_valid: bool = Field(strict=True)
    reason: str = Field(min_length=1, max_length=1600)


def validate_planning_verdict(raw: dict, storyboard: dict) -> list[str]:
    verdict = PlanVerdict.model_validate(raw)
    expected = {(s['shot_id'], i): beat for s in storyboard['shots'] for i, beat in enumerate(s['action_beats'])}
    actual = [(unit.shot_id, unit.beat_index) for unit in verdict.units]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError('分镜语义检查未逐项覆盖全部动作，不能放行')
    issues = []
    for unit in verdict.units:
        beat = expected[(unit.shot_id, unit.beat_index)]
        if any(field not in beat for field in unit.evidence_fields):
            raise ValueError('分镜语义检查引用了原稿中不存在的字段')
        if beat.get('outcomes'):
            if unit.outcome_contract_valid is None or 'outcomes' not in unit.evidence_fields:
                raise ValueError('分镜语义检查缺少独立结果的来源与判断')
            if beat.get('scene_changes') and 'scene_changes' not in unit.evidence_fields:
                raise ValueError('分镜语义检查缺少场景结果分类的来源与判断')
            if not unit.outcome_contract_valid:
                issues.append(f'Shot {unit.shot_id} 动作 {unit.beat_index + 1}: 结果合同无效：{unit.reason}')
        if not unit.cause_and_participants_valid or not unit.classification_matches_description:
            issues.append(f'Shot {unit.shot_id} 动作 {unit.beat_index + 1}: {unit.reason}')
    if not verdict.continuity_valid or not verdict.requested_outcome_valid:
        issues.append(verdict.reason)
    issues.extend(admission_issues(storyboard, raw['units']))
    return issues


def ensure_planning_semantics(storyboard: dict, workspace: Path, save_diagnostic) -> dict:
    plan = storyboard.get('production_plan') or {}
    if plan.get('version') not in {'production-plan-v5-payoff-events', 'production-plan-v6-performance', 'production-plan-v7-film-rhythm', 'production-plan-v8-request-shape'} or not any(s.get('min_interaction_events') for s in plan['slots']):
        return storyboard
    if (plan.get('version') == 'production-plan-v8-request-shape'
            and sum(len(s['action_beats']) for s in storyboard['shots']) > 24):
        from pipeline.planning_review_batches import ensure_batched_planning_review
        return ensure_batched_planning_review(storyboard, workspace, save_diagnostic)
    has_outcomes = any(b.get('outcomes') for s in storyboard['shots'] for b in s['action_beats'])
    has_scene_changes = any(b.get('scene_changes') for s in storyboard['shots'] for b in s['action_beats'])
    version = ('planning-semantics-v4-scene-results' if has_scene_changes else
               'planning-semantics-v3-outcomes' if has_outcomes else 'planning-semantics-v2-field-evidence')
    reviewed_results = any(s.get('result_admission') == 'review-required-v1' for s in storyboard['shots'])
    if reviewed_results:
        version = 'planning-semantics-v5-result-necessity'
    cache_path = workspace / 'planning_semantic_review.json'
    saved = json.loads(cache_path.read_text()) if cache_path.is_file() else None
    if saved is not None and (not isinstance(saved, dict)
                             or saved.get('protocol_version') not in (None, *protocol.SUPPORTED_VERSIONS)
                             or saved.get('outcome_version') not in (None, outcome.VERSION)):
        raise ValueError('分镜审查缓存协议无效，不能隐式重新审查')
    bound = reviewed_results and (saved is None or saved.get('protocol_version') in protocol.SUPPORTED_VERSIONS)
    targets = protocol.review_targets(storyboard) if bound else None
    if bound:
        version = saved['protocol_version'] if saved is not None else protocol.VERSION
    sources = continuity_sources(storyboard) if bound and version == protocol.CITATION_VERSION else None
    from pipeline.planning_review_sources import review_evidence_sources
    evidence_sources = review_evidence_sources(storyboard) if bound and version == protocol.VERSION else None
    # 历史缓存按原请求验证；只有新请求或已标记的新缓存使用结局证据协议。
    outcome_context = (outcome.outcome_review_context(storyboard)
                       if saved is None or saved.get('outcome_version') == outcome.VERSION else None)
    if saved is not None and saved.get('outcome_version') and outcome_context is None:
        raise ValueError('结局审查缓存与当前需求或批次不一致')
    payload = {'version': version, 'model': config.LLM_MODEL, 'storyboard': storyboard}
    if bound:
        payload['review_targets'] = targets
    if sources is not None:
        payload['continuity_sources'] = sources
    if evidence_sources is not None:
        payload['evidence_sources'] = evidence_sources
    if outcome_context is not None:
        payload['outcome_review'] = outcome_context
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if saved is not None:
        if saved.get('fingerprint') != fingerprint:
            raise ValueError('分镜语义检查与当前稿件或模型不一致，已停止，未调用视频接口')
        raw = saved['verdict']
    else:
        from pipeline.storyboard import _call_llm_for_storyboard
        schema = (protocol.compile_review_response_schema(PlanVerdict.model_json_schema(), targets, sources=sources,
                                                          evidence_sources=evidence_sources)
                  if bound else PlanVerdict.model_json_schema())
        if outcome_context is not None:
            schema = outcome.outcome_response_schema(schema, outcome_context)
        client = OpenAI(api_key=config.ARK_API_KEY, base_url=config.ARK_BASE_URL, max_retries=0)
        raw = _call_llm_for_storyboard(client,
            '你是独立动作因果审稿人。检查给定分镜，而不是相信它的类型标签。只输出JSON，不改稿、不补动作。'
            '逐一覆盖每镜action_beats，beat_index从0开始。cause_and_participants_valid判断谁发起作用、谁受影响与文字原因结果是否一致；'
            'classification_matches_description比较既定activity类型与描述是否一致：transition可以是移动入场，hold可以是结果展示；'
            '这两种类型不需要外部交互，类型与描述吻合即为true。interaction则需要真实外部作用及对应response_kind的反应，'
            '被打后的倒下不能反写成受击者对攻击者的新作用。evidence_fields仅选择当前beat中支持判断的字段名，由程序取回原文，reason解释判断。'
            '有outcomes时，evidence_fields必须引用outcomes，outcome_contract_valid必须给出布尔判断：'
            '结果是否为对应参与者的完整可见状态、与response_kind和前后叙事一致，且没有把导演自选的路径、招式或时刻充当必要结果。'
            '有scene_changes时也必须引用该字段；outcome_contract_valid同时检查每项变化是否由本次作用引起、对象是否正确、与unaffected等作用范围是否矛盾。'
            'required是必须可见的对象结果，incidental只是可选细节；用户指定的结果、下一镜依赖的状态不能被标成incidental。'
            '检查后续镜头的依赖；如果后镜必须依赖前镜某个可选细节才能成立，continuity_valid=false，不准靠后镜文字补出未保证的状态。'
            '只有刀光或威胁没有目标反应不能通过；不能借抽象结果删除用户原需求中的指定动作或击杀等结局。'
            '检查跨镜场景身份、起止状态与动作衔接是否一致，以及最终结局是否满足原需求。结构合法不代表语义正确。'
            '若存在review_scope，只为本批shots输出units；previous_shot是上一批已审查的原稿，仅用于检查接缝，不能重复输出其units。'
            'is_final_batch=false时，requested_outcome_valid判断本批是否保持结局成立的前提且未提前完结；'
            '仅最后一批必须完整实现最终结局。顶层原需求、角色和全片计划始终有效，不得因分批改写或遗漏。'
            + ((CLASSIFICATION_INSTRUCTION if evidence_sources is not None else INSTRUCTION)
               if reviewed_results else '')
            + ((protocol.TARGET_INSTRUCTION + protocol.SOURCED_INSTRUCTION
                if evidence_sources is not None else protocol.INSTRUCTION) if bound else '')
            + (outcome.INSTRUCTION if outcome_context is not None else ''),
            json.dumps(payload, ensure_ascii=False), temperature=0, repair_json=False,
            response_schema=ark_output_schema(schema))
        save_diagnostic('planning_semantic_review.json', {'fingerprint': fingerprint, 'verdict': raw,
            **({'protocol_version': protocol.VERSION} if bound else {}),
            **({'outcome_version': outcome.VERSION} if outcome_context is not None else {})})
    outcome_evidence = None
    if outcome_context is not None:
        raw, outcome_evidence = outcome.decode_outcome_review(raw, outcome_context)
    if bound:
        raw = protocol.decode_review_response(raw, targets, sources=sources, evidence_sources=evidence_sources)
    issues = validate_planning_verdict(raw, storyboard)
    beats = {(s['shot_id'], i): b for s in storyboard['shots'] for i, b in enumerate(s['action_beats'])}
    save_diagnostic('planning_semantic_evidence.json', {'fingerprint': fingerprint, 'issues': issues,
        **({'requested_outcome': outcome_evidence} if outcome_evidence is not None else {}),
        'units': [{'shot_id': unit['shot_id'], 'beat_index': unit['beat_index'],
                   'source_fields': {field: beats[(unit['shot_id'], unit['beat_index'])][field]
                                     for field in unit['evidence_fields']}} for unit in raw['units']]})
    if issues:
        raise ValueError('分镜语义验收未通过；未重写分镜、未调用视频接口：' + '；'.join(issues))
    admitted = apply_result_admission(storyboard, raw['units']) if reviewed_results else storyboard
    if reviewed_results:
        save_diagnostic('result_admission.json', {'version': version, 'source_fingerprint': fingerprint,
            'units': raw['units'], 'execution_storyboard': admitted})
    return admitted
