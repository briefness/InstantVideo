"""Whole-film rhythm budgeting and evidence admission; never submits or repairs takes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def planned_interaction_seconds(shots: list[dict]) -> float:
    from pipeline.action_timing import action_timing_issues
    total = 0.
    for shot in shots:
        if action_timing_issues(shot):
            raise ValueError('全片节奏需要有效的逐镜动作时间轴')
        end = 0.
        for beat in shot.get('action_beats', []):
            if beat.get('activity') == 'interaction':
                total += max(0., beat['end_seconds'] - max(end, beat['start_seconds']))
                end = max(end, beat['end_seconds'])
    return total


def planning_rhythm_context(plan: dict, completed_shots: list[dict]) -> dict | None:
    from pipeline.action_timing import passive_floor_ms

    minimum = plan.get('min_interaction_seconds')
    if minimum is None:
        return None
    completed = planned_interaction_seconds(completed_shots)
    remaining = plan['slots'][len(completed_shots):]
    capacity = sum(s['duration'] - s.get('result_hold_seconds', 0) for s in remaining)
    required = max(0., minimum - completed)
    prior_capacity = sum(slot['duration'] - passive_floor_ms(
        [b['activity'] for b in shot['action_beats']], slot.get('result_hold_seconds', 0)) / 1000
        for shot, slot in zip(completed_shots, plan['slots']))
    if minimum > prior_capacity + capacity + 1e-6:
        raise ValueError('剩余镜头无法满足全片交互预算；停止创作，未提交视频')
    return {'minimum_total_interaction_seconds': minimum,
            'completed_planned_interaction_seconds': round(completed, 3),
            'remaining_required_interaction_seconds': round(required, 3),
            'remaining_available_seconds': round(capacity, 3),
            'prior_reallocatable_seconds': round(max(0., prior_capacity - completed), 3)}


def planned_film_timing_issues(storyboard: dict) -> list[str]:
    from pipeline.performance import performance_projection
    plan = storyboard.get('production_plan') or {}
    if plan.get('min_interaction_seconds') is None:
        return []
    try:
        if any(performance_projection(shot).get('version') != 'performance-v2-outcomes'
               for shot in storyboard.get('shots', [])):
            return ['全片节奏验收需要每个镜头完整的独立结果合同；不能等生成后才发现证据不兼容']
        planned = planned_interaction_seconds(storyboard.get('shots', []))
    except (KeyError, TypeError, ValueError):
        return ['全片节奏缺少有效动作时间轴']
    if planned + 1e-6 < plan['min_interaction_seconds']:
        return [f"全片交互时长 {planned:g} 秒不足 {plan['min_interaction_seconds']:g} 秒；不能以运镜或停顿补足"]
    return []


def ensure_observed_film_rhythm(storyboard: dict, results: list, workspace: Path) -> None:
    """Recompute from original accepted observations, not saved aggregate scores.

    Evidence is bound to media bytes and the actual accepted contract (which can
    include an observed-state continuation). No duplicate ledger or fresh review.
    """
    from pipeline.causality import CompiledActionContract
    from pipeline.performance import measured_performance
    from pipeline.run_state import RunWorkspace
    from pipeline.semantic_review import SemanticReviewUnavailableError

    minimum = (storyboard.get('production_plan') or {}).get('min_interaction_seconds')
    if minimum is None:
        return
    workspace = Path(workspace).resolve()
    report = {'version': 'film-rhythm-v1', 'minimum_interaction_seconds': minimum,
              'status': 'unavailable', 'clips': []}
    try:
        ids = [r.shot_id for r in results]
        if ids != [s['shot_id'] for s in storyboard['shots']]:
            raise ValueError('全片节奏缺少完整有序镜头')
        low, high = 0., 0.
        seen_media = set()
        for shot, result in zip(storyboard['shots'], results):
            if result.status != 'success' or result.semantic_accepted is not True or not result.local_path:
                raise ValueError('全片节奏只能使用已接受原片')
            video = Path(result.local_path).resolve()
            if not video.is_relative_to(workspace):
                raise ValueError('原片不在当前工作区')
            with video.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if digest in seen_media:
                raise ValueError('同一原片不能重复充当多个已生成镜头')
            seen_media.add(digest)
            cache = workspace / 'semantic_reviews' / f'{digest}.json'
            if cache.is_symlink() or cache.stat().st_size > 4 * 1024 * 1024:
                raise ValueError('节奏证据文件无效')
            saved = json.loads(cache.read_text())
            contract = CompiledActionContract(**saved['action_contract'])
            if (saved['video_hash'] != digest or saved['review']['accepted'] is not True
                    or not result.accepted_contract_fingerprint
                    or contract.fingerprint() != result.accepted_contract_fingerprint):
                raise ValueError('节奏证据与已接受的原片或合同不一致')
            events = saved['action_event_evidence']
            if contract.event_evidence_issues(events):
                raise ValueError('节奏证据中的动作未通过逐事件检查')
            policy = contract.review_projection['performance_policy']
            if policy['version'] != 'performance-v2-outcomes':
                raise ValueError('缺少可比较的连续交互时间证据')
            measurement = measured_performance(events, {**policy, 'duration': saved['media_duration']},
                                               saved['sample_timestamps'], edit_duration=shot['duration'])
            if measurement['status'] != 'passed':
                raise ValueError('剪辑范围内的动作或结果停留时间不能确认通过')
            a, b = measurement['interaction_seconds']
            low += a
            high += b
            report['clips'].append({'shot_id': result.shot_id, 'video_sha256': digest,
                                    'accepted_contract_fingerprint': result.accepted_contract_fingerprint,
                                    'edit_duration': shot['duration'],
                                    'interaction_seconds': [a, b]})
        report.update(interaction_seconds=[low, high],
                      status='passed' if low >= minimum else 'failed' if high < minimum else 'uncertain')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        report['reason'] = '已接受素材缺少可核对的全片时间证据'
        RunWorkspace._atomic_write_json(workspace / 'film_rhythm_review.json', report)
        raise SemanticReviewUnavailableError('全片时间证据缺失或不一致；保留已生成素材，不自动重拍') from exc
    RunWorkspace._atomic_write_json(workspace / 'film_rhythm_review.json', report)
    if report['status'] == 'uncertain':
        raise SemanticReviewUnavailableError('全片时间证据跨越节奏阈值；保留原片，停止导出，不自动重拍')
    if report['status'] == 'failed':
        raise RuntimeError('各镜头动作可读，但全片交互时长不足；保留原片，停止导出，不自动重拍')
