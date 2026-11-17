"""从冻结目标导出单镜观察问题；全片目标与本镜贡献不能共用完成语义。"""
from copy import deepcopy

from pipeline.goal_allocation import GoalEvidenceUnit


INSTRUCTION = (
    'For kind=contribution, evaluate only the necessary functional contribution of the current '
    'shot located by execution_scope.units. goal_context is the joint source goal, not a '
    'whole-goal completion question for this clip. Its participating_shots carry the remaining work. '
    'Local units locate the contribution; they are authored realizations, not obligations to '
    'reproduce incidental poses, gaze durations or resting hand positions. Necessary participants, '
    'causal order, target response, explicit methods and kind=state constraints still apply. '
    'Do not infer unseen local work from a future result. A local contribution verdict never '
    'certifies the joint goal or any other shot.'
)


def local_assessment(catalog: dict) -> dict:
    """保留固定键和来源；移除当前问题中的外镜动作，不改编译合同或原稿。"""
    result = deepcopy(catalog)
    for criterion in result['criteria'].values():
        scope = criterion.get('execution_scope')
        if 'execution_scope' not in criterion:
            continue
        if criterion.get('kind') != 'completion' or not isinstance(scope, dict):
            raise ValueError('单镜核对缺少有效的冻结完成范围')
        shot_id = scope.get('shot_id')
        if type(shot_id) is not int or shot_id < 1 or not isinstance(scope.get('units'), list):
            raise ValueError('单镜核对缺少有效镜头身份')
        units = [GoalEvidenceUnit.model_validate(u).model_dump() for u in scope['units']]
        ids = [(u['shot_id'], u['beat_index']) for u in units]
        local = [u for u in units if u['shot_id'] == shot_id]
        if (len(set(ids)) != len(ids) or len({s for s, _ in ids}) < 2 or not local
                or [u['beat_index'] + 1 for u in local] != criterion.get('beat_ids')):
            raise ValueError('单镜核对范围与冻结动作归属不一致')
        criterion['kind'] = 'contribution'
        criterion['goal_context'] = {
            'action': criterion.pop('action'),
            'participating_shots': list(dict.fromkeys(u['shot_id'] for u in units)),
        }
        # 原稿快照留在编译合同中用于来源绑定，不给验收端二次解释为义务。
        # 目标原文保留用户明确的方法；局部身份、参与者和目标仍定位本镜贡献。
        if catalog.get('execution_semantics') == 'intent-bounded-v1':
            local = [{k: u[k] for k in ('shot_id', 'beat_index', 'actor', 'target')} for u in local]
        criterion['execution_scope'] = {'shot_id': shot_id, 'units': local}
    return result


def source_requirement(criterion: dict) -> str:
    """引用绑定仍指向完整来源文字，不能把局部观察冒充全片完成。"""
    text = (criterion['goal_context']['action'] if criterion['kind'] == 'contribution'
            else criterion.get('action') or criterion.get('state'))
    if not isinstance(text, str) or not text.strip():
        raise ValueError('核对缺少既定要求')
    return text
