"""Bound long-plan review responses without dropping units or boundary context."""
import hashlib
import json
from pathlib import Path

import config
from pipeline.run_state import RunWorkspace
from pipeline.planning_evidence import BATCH_REVIEW_UNITS


def _source_units(shots: list[dict]) -> list[dict]:
    from pipeline.goal_allocation import evidence_unit
    ids = [s['shot_id'] for s in shots]
    if len(set(ids)) != len(ids):
        raise ValueError('分批审查镜头身份重复，不能覆盖原稿来源')
    return [evidence_unit(s['shot_id'], i, b) for s in shots for i, b in enumerate(s['action_beats'])]


def merge_review_batches(storyboard: dict, admitted_shots: list[dict]) -> dict:
    """合并只搬运已审定证据，不按目标文字相同而合并不同发生次数。"""
    from pipeline.completion_intent import validate_goal_coverage
    if _source_units(storyboard['shots']) != _source_units(admitted_shots):
        raise ValueError('分批合并与原稿动作目录不一致：存在遗漏、换序或来源改写')
    result = {**storyboard, 'shots': admitted_shots}
    validate_goal_coverage(result)
    return result


def ensure_batched_planning_review(storyboard: dict, workspace: Path, save_diagnostic) -> dict:
    from pipeline.planning_review import ensure_planning_semantics

    shots = storyboard['shots']
    if not 1 <= len(shots) <= 150 or any(not 1 <= len(s['action_beats']) <= 12 for s in shots):
        raise ValueError('分批审查需要有界镜头和每镜 1–12 个动作单元')
    _source_units(shots)
    source = {'version': 'planning-batches-v1', 'model': config.LLM_MODEL, 'storyboard': storyboard}
    fingerprint = hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    root = Path(workspace)
    summary_path = root / 'planning_semantic_review.json'
    if summary_path.is_file() and json.loads(summary_path.read_text()).get('fingerprint') != fingerprint:
        raise ValueError('分批语义检查与当前稿件或模型不一致，已停止')
    batches, current, count = [], [], 0
    for shot in shots:
        size = len(shot['action_beats'])
        if current and count + size > BATCH_REVIEW_UNITS:
            batches.append(current)
            current, count = [], 0
        current.append(shot)
        count += size
    if current:
        batches.append(current)
    records, previous, admitted_shots = [], None, []
    for index, batch in enumerate(batches, 1):
        relative = f'planning_review_batches/{index:03d}'
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        scoped = {**storyboard, 'shots': batch, 'review_scope': {
            'source_fingerprint': fingerprint, 'previous_shot': previous,
            'is_final_batch': index == len(batches),
        }}
        if any(s.get('result_admission') == 'review-required-v1' for s in batch):
            scoped['review_scope']['next_shot'] = batches[index][0] if index < len(batches) else None
        # Reuse the same admission/cache rules; a batch is at most 24 units, so
        # this call cannot split recursively. Failed batches remain failed.
        admitted = ensure_planning_semantics(scoped, directory,
            lambda name, data, directory=directory: RunWorkspace._atomic_write_json(directory / name, data))
        if _source_units(batch) != _source_units(admitted['shots']):
            raise ValueError('分批审查结果与本批原稿动作目录不一致，停止后续批次')
        admitted_shots.extend(admitted['shots'])
        records.append({'shot_ids': [s['shot_id'] for s in batch],
                        'review': f'{relative}/planning_semantic_review.json',
                        'evidence': f'{relative}/planning_semantic_evidence.json'})
        previous = batch[-1]
    result = merge_review_batches(storyboard, admitted_shots)
    save_diagnostic('planning_semantic_review.json', {
        'version': 'planning-batches-v1', 'fingerprint': fingerprint, 'status': 'passed', 'batches': records})
    return result
