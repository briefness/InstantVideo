"""分别保留内容观察与测时可靠性；不产生新观察或重拍许可。"""
from pipeline.performance import measured_performance, requires_measured_timing


def assess_event_evidence(contract, evidence, *, timestamps=None, media_duration=None) -> dict:
    policy = contract.review_projection.get('performance_policy') or {}
    # 只把归属明确的布尔反例作为已观察偏差；无效时间不抹掉这些信息。
    expected = contract.review_projection.get('action_events') or []
    content_issues = []
    if isinstance(evidence, list) and len(evidence) == len(expected):
        for event, observed in zip(expected, evidence):
            if (isinstance(observed, dict) and type(observed.get('event_id')) is int
                    and observed['event_id'] == event['event_id']):
                content_issues.extend(f"事件{event['event_id']}未满足 {key}"
                    for key in contract.event_evidence_fields() if observed.get(key) is False)
    issues = contract.event_evidence_issues(evidence)
    report = {'version': 'event-assessment-v1', 'content_issues': content_issues,
              'issues': issues, 'status': 'rejected' if issues else 'passed',
              'timing': {'status': 'contract_checked' if policy.get('version') == 'performance-v1'
                         else 'not_required'}}
    if policy.get('version') != 'performance-v2-outcomes':
        return report
    shape = contract.event_evidence_shape_issues(evidence)
    if shape:
        return {**report, 'status': 'unavailable', 'reason': '；'.join(shape),
                'timing': {'status': 'not_evaluated', 'reason': '事件证据结构无效'}}
    if not requires_measured_timing(policy):
        return report
    try:
        measurement = measured_performance(evidence,
            {**policy, 'duration': media_duration} if media_duration is not None else policy, timestamps)
    except (ValueError, TypeError, KeyError) as exc:
        return {**report, 'status': 'unavailable', 'reason': str(exc),
                'timing': {'status': 'invalid', 'reason': str(exc)}}
    report['timing'] = measurement
    if (measurement['status'] == 'uncertain'
            and (not issues or policy.get('time_basis') != 'sample-bounds-v1')):
        report.update(status='unavailable', reason=('采样精度不足以确认因果先后'
            if measurement.get('causal_order', {}).get('status') == 'uncertain'
            else '时间证据的采样范围跨越验收阈值'))
        if policy.get('time_basis') == 'sample-bounds-v1':
            report['failure_kind'] = 'semantic_uncertainty'
    elif measurement['status'] == 'failed':
        report['issues'] = [*issues, ('采样范围已确认因果顺序违反合同'
            if measurement.get('causal_order', {}).get('status') == 'failed'
            else '采样范围已确认实际交互时长或结果保持不足')]
        report['status'] = 'rejected'
    return report
