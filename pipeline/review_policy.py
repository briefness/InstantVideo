"""审查事实与交付决策分离；容错只能由已分类的证据授权。"""
from dataclasses import dataclass, fields


WARNING_POLICY = 'reviewed_with_warnings'
QUALITY_FIELDS = {
    'blocking_valid': '人物站位或表演调度未完全达到设计',
    'composition_change_valid': '镜头构图变化未完全达到设计',
    'screen_direction_valid': '镜头间屏幕方向或空间轴存在偏差',
    'creative_brief_valid': '风格或表演节奏未完全达到创作要求',
}


@dataclass(frozen=True)
class ReviewDecision:
    allowed: bool
    warnings: tuple[str, ...] = ()


def timing_is_quality_only(assessment: dict) -> bool:
    """核心工作必须已核对；坏数据、内容反证、确认逆序都不能降级。"""
    timing = assessment.get('timing') or {}
    return (assessment.get('status') in {'rejected', 'unavailable'}
            and assessment.get('content_issues') == []
            and timing.get('status') in {'failed', 'uncertain'}
            and timing.get('causal_order', {}).get('status') in {'passed', 'uncertain'}
            and (assessment.get('status') != 'unavailable'
                 or assessment.get('failure_kind') == 'semantic_uncertainty'))


def decide_review(review, mode: str) -> ReviewDecision:
    if mode not in {'strict', 'normal'}:
        raise ValueError('审查模式必须是严格或正常')
    # 不读取 failure_reason 来猜严重性；未知字段/类型不能变成质量警告。
    flags = {f.name: getattr(review, f.name) for f in fields(review)
             if f.type in (bool, 'bool') and f.name != 'accepted'}
    if (any(type(v) is not bool for v in flags.values())
            or type(review.accepted) is not bool
            or any(v is not True for v in review.required_entities_visible.values())):
        return ReviewDecision(False)
    failed = {k for k, v in flags.items() if not v}
    if review.accepted and failed:
        return ReviewDecision(False)
    if mode == 'strict':
        return ReviewDecision(review.accepted and not failed)
    if not all(review.observed_end_state.get(k) for k in ('location', 'subject', 'action_phase')):
        return ReviewDecision(False)
    quality = set(QUALITY_FIELDS)
    temporal = timing_is_quality_only(review.event_assessment)
    if temporal:
        quality.update(('action_timing_valid', 'action_sequence_valid'))
    if failed - quality or (not review.accepted and not failed):
        return ReviewDecision(False)
    warnings = [text for name, text in QUALITY_FIELDS.items() if name in failed]
    if failed & {'action_timing_valid', 'action_sequence_valid'}:
        warnings.append('动作时间或采样精度未完全满足严格验收，核心内容检查未发现反证')
    return ReviewDecision(True, tuple(warnings))
