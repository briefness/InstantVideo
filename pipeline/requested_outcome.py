"""用户拥有结果要求；创作提供实现，审查以原稿证据判断是否兑现。"""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


VERSION = 'requested-outcome-v1'
INSTRUCTION = '''存在outcome_review时，不返回requested_outcome_valid，而返回requested_outcome。
requirement是用户原文，不能用故事概要、作者意图、胜负标签或相近结果替换它。
先阅读sources中的实际变化与最终状态，evidence只选择这些原文路径；再判断它们是否足以证明完整要求。
relation=entailed表示证据足以证明；contradicted表示证据与要求矛盾；undetermined表示可能成立但没有证明。
“没有明显矛盾”“看起来像”“已按分镜执行”不能当作entailed。证据也允许以间接表演建立结果，不要求重复用户关键词。
reason解释证据与完整要求的关系，以及缺失或相矛盾之处。不要补写原稿未发生的事实。未规定的细节不构成失败。
'''


def requested_ending(plan: dict) -> str:
    intent = plan.get('director_intent') or {}
    if not intent.get('ending_requirement'):
        return ''
    quote = intent.get('ending_evidence')
    request = intent.get('source_request', '')
    if not isinstance(quote, str) or not quote.strip() or quote not in request:
        raise ValueError('用户结局缺少原始需求依据，不能用创作概要替代')
    return quote


def outcome_review_context(storyboard: dict) -> dict | None:
    if (storyboard.get('review_scope') or {}).get('is_final_batch') is False:
        return None
    requirement = requested_ending(storyboard.get('production_plan') or {})
    if not requirement:
        return None
    shots = storyboard['shots']
    shot, prefix = shots[-1], f'/shots/{len(shots) - 1}'
    sources = {}
    def add(path, value):
        if isinstance(value, str) and value.strip():
            sources[path] = value
    # 用户要求和作者概要属于待证明目标，不能成为自身成立的证据。
    for index, beat in enumerate(shot['action_beats']):
        base = f'{prefix}/action_beats/{index}'
        add(f'{base}/visible_result', beat.get('visible_result'))
        for name in ('outcomes', 'scene_changes'):
            for j, outcome in enumerate(beat.get(name, [])):
                add(f'{base}/{name}/{j}/state', outcome.get('state'))
    for name, value in (shot.get('end_state') or {}).items():
        add(f'{prefix}/end_state/{name}', value)
    add(f'{prefix}/narrative_beat/state_after', (shot.get('narrative_beat') or {}).get('state_after'))
    return {'version': VERSION, 'requirement': requirement, 'sources': sources}


class OutcomeJudgment(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True)
    relation: Literal['entailed', 'contradicted', 'undetermined']
    evidence: list[str] = Field(max_length=24)
    reason: str = Field(min_length=1, max_length=1600)


def outcome_response_schema(schema: dict, context: dict) -> dict:
    result = deepcopy(schema)
    judgment = OutcomeJudgment.model_json_schema()
    paths = list(context['sources'])
    # 空证据集只允许不确定结论；不向提供方发送空 enum。
    if paths:
        judgment['properties']['evidence']['items'] = {'type': 'string', 'enum': paths}
    else:
        judgment['properties']['evidence'] = {'type': 'array', 'const': []}
    if not paths:
        judgment['properties']['relation'] = {'type': 'string', 'const': 'undetermined'}
    result['properties'].pop('requested_outcome_valid')
    result['required'].remove('requested_outcome_valid')
    result['properties']['requested_outcome'] = judgment
    result['required'].append('requested_outcome')
    return result


def decode_outcome_review(raw: dict, context: dict) -> tuple[dict, dict]:
    if 'requested_outcome_valid' in raw or 'requested_outcome' not in raw:
        raise ValueError('用户结局必须提供原稿证据与判定，不能只返回整体批准')
    judgment = OutcomeJudgment.model_validate(raw['requested_outcome'])
    if (len(set(judgment.evidence)) != len(judgment.evidence)
            or any(path not in context['sources'] for path in judgment.evidence)
            or judgment.relation != 'undetermined' and not judgment.evidence):
        raise ValueError('用户结局证据必须引用实际结果原文，确定判断不能缺少证据')
    result = deepcopy(raw)
    result.pop('requested_outcome')
    result['requested_outcome_valid'] = judgment.relation == 'entailed'
    if not result['requested_outcome_valid']:
        result['reason'] = judgment.reason
    evidence = {**judgment.model_dump(), 'requirement': context['requirement'],
                'source_fields': {p: context['sources'][p] for p in judgment.evidence}}
    return result, evidence
