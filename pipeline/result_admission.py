"""Independent classification of authored states before execution is frozen.

The source claims are never rewritten. Directions remain generation material;
only independently justified requirements become state-matching obligations.
"""
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline.action_results import result_claims


POLICY = 'reviewed-results-v1'


class ResultDecision(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True, str_strip_whitespace=True)
    source: str = Field(min_length=1, max_length=80)
    basis: Literal['user', 'continuity', 'response', 'direction']
    evidence: str = Field(min_length=1, max_length=1600)
    minimal: bool = Field(strict=True)
    reason: str | None = Field(default=None, min_length=1, max_length=1600)


class ResultAdmission(BaseModel):
    model_config = ConfigDict(extra='forbid', hide_input_in_errors=True, str_strip_whitespace=True)
    decisions: list[ResultDecision] = Field(max_length=30)
    requirements_preserved: bool = Field(strict=True)
    continuous_performance: bool = Field(strict=True)
    reason: str = Field(min_length=1, max_length=1600)


CLASSIFICATION_INSTRUCTION = '''对result_admission=review-required-v1的镜头，只对有待归类结果的unit返回result_admission，不得只给整体outcome_contract_valid。
此模式下outcomes是待审定的作者候选，不是已生效的义务；outcome_contract_valid判断按以下分类后是否成立，合理的direction不应仅因原来写在outcomes就被否决。
decisions逐条覆盖当前动作的outcomes/N及importance=required的scene_changes/N；没有候选结果时此项不适用，可省略或为null，不能虚构批准。
所有unit仍须判断原因、参与者与activity分类；continuity_valid检查完整表演及跨镜衔接，requested_outcome_valid检查原需求和最终结局，包括过渡与收势。
basis=user：原需求明确指定这个状态，不能用作者自己的分镜证明用户要求。
basis=continuity：后续动作确实离不开这个状态。仅仅重复同一姿态不构成必要依赖。
basis=response：这是使本次作用及目标反应可辨认的最小功能结果，例如被迫让开、物体交到对方手中；具体膝盖角度、剑尖方向、站姿不是功能结果。
basis=direction：可替换的实现路径、姿势、招式收势或装饰。原文仍交给生成，不删除创作。
requirements_preserved检查此分类完整保留原需求、每次真实作用和目标反应、关键物体变化、最终结局与后续依赖。
continuous_performance检查动作作为完整表演能否连续执行：运动承接、双方配合、转折与停顿有叙事理由，不能为每个事件或镜头人为复位；有意义的停顿允许，不要求每镜打满。原稿无法成立则false，不自动重写。
字段归类是独立判断，不相信作者已将某个姿态称为“结果”。用户要求与后续依赖不确定时requirements_preserved=false。
'''

INSTRUCTION = CLASSIFICATION_INSTRUCTION + '''basis=user时，evidence必须逐字引用原需求。
basis=continuity时，evidence填写后续原稿的JSON Pointer，如/shots/1/start_state/prop_state；跨批的紧邻下一镜使用/review_scope/next_shot/start_state/prop_state。
basis=direction时，evidence说明替换它为何仍满足用户目标与后续因果。
minimal表示该陈述在它所属类别中没有混入另一类必要性；一句话混合必要结局和可选姿态时必须false，不能把整句话降级来绕过要求。
'''


def review_claims(shot: dict, beat: dict) -> dict:
    """适用性取自原稿中的待审结果，不按题材、姿态或缺失响应猜测。"""
    return result_claims(beat) if shot.get('result_admission') == 'review-required-v1' else {}


def admission_issues(storyboard: dict, units: list[dict]) -> list[str]:
    shots = {s['shot_id']: s for s in storyboard['shots']}
    expected = {(s['shot_id'], i) for s in storyboard['shots'] for i in range(len(s['action_beats']))}
    actual = [(u.get('shot_id'), u.get('beat_index')) for u in units]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('结果必要性审查必须完整覆盖全部动作单元')
    request = (storyboard.get('production_plan', {}).get('director_intent') or {}).get('source_request', '')
    issues = []
    for unit in units:
        shot = shots[unit['shot_id']]
        if shot.get('result_admission') != 'review-required-v1':
            continue
        beat = shot['action_beats'][unit['beat_index']]
        claims = review_claims(shot, beat)
        if unit.get('result_admission') is None:
            if claims:
                raise ValueError(f"Shot {shot['shot_id']} 动作 {unit['beat_index'] + 1}: 缺少待审结果的必要性判断")
            continue
        audit = ResultAdmission.model_validate(unit['result_admission'])
        paths = [d.source for d in audit.decisions]
        if len(paths) != len(set(paths)) or set(paths) != set(claims):
            raise ValueError('结果必要性审查必须逐条覆盖原始陈述，不能重复、省略或添加')
        for decision in audit.decisions:
            if not decision.minimal:
                issues.append(f"Shot {shot['shot_id']}: 必要结果与表演细节尚未分清：{decision.source}")
            if decision.basis == 'user' and decision.evidence not in request:
                raise ValueError('用户结果依据必须逐字来自原始需求')
            if decision.basis == 'continuity':
                # Only forward, readable source fields; arbitrary filesystem/URL
                # pointers and references to the claim itself are not evidence.
                tokens = decision.evidence.split('/')
                try:
                    external_next = tokens[:3] == ['', 'review_scope', 'next_shot']
                    if len(tokens) < 5 or (tokens[:2] != ['', 'shots'] and not external_next):
                        raise ValueError
                    current_index = next(i for i, s in enumerate(storyboard['shots']) if s['shot_id'] == shot['shot_id'])
                    if external_next:
                        next_shot = storyboard['review_scope']['next_shot']
                        index = len(storyboard['shots'])
                        if current_index != index - 1 or not next_shot:
                            raise ValueError
                    else:
                        index = int(tokens[2])
                        if str(index) != tokens[2] or not 0 <= index < len(storyboard['shots']):
                            raise ValueError
                        next_shot = storyboard['shots'][index]
                    if index < current_index:
                        raise ValueError
                    if index == current_index:
                        if tokens[3] != 'action_beats' or int(tokens[4]) <= unit['beat_index']:
                            raise ValueError
                    elif tokens[3] not in {'start_state', 'action_beats', 'narrative_beat'}:
                        raise ValueError
                    value = next_shot
                    for token in tokens[3:]:
                        if isinstance(value, list):
                            if not token.isdecimal() or str(int(token)) != token:
                                raise ValueError
                            value = value[int(token)]
                        else:
                            value = value[token]
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError
                except (KeyError, IndexError, TypeError, ValueError, StopIteration):
                    raise ValueError('结果衔接依据必须指向后续动作或镜头的实际文字') from None
        if not audit.requirements_preserved or not audit.continuous_performance:
            issues.append(f"Shot {shot['shot_id']}: {audit.reason}")
    return issues


def apply_result_admission(storyboard: dict, units: list[dict]) -> dict:
    issues = admission_issues(storyboard, units)
    if issues:
        raise ValueError('结果必要性审查未通过：' + '；'.join(issues))
    result = deepcopy(storyboard)
    shots = {s['shot_id']: s for s in result['shots']}
    pending = {s['shot_id'] for s in storyboard['shots'] if s.get('result_admission') == 'review-required-v1'}
    for unit in units:
        shot = shots[unit['shot_id']]
        if shot['shot_id'] not in pending:
            continue
        beat = shot['action_beats'][unit['beat_index']]
        if beat.get('activity') == 'interaction':
            # admission_issues 已确认：有候选结果时不能省略判断。
            beat['result_directions'] = [d['source'] for d in (unit.get('result_admission') or {}).get('decisions', [])
                                         if d['basis'] == 'direction']
        shot['result_admission'] = POLICY
    return result
