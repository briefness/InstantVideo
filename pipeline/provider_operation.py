"""素材是操作数，不是操作。保持时间承接与摄影切换两个独立维度。

Fast 2.0 没有 2.5 的任务类型参数；官方延长操作通过文本分派。
这里记录真实能力，不把时间定位、视图窗口误报成服务端裁剪或强控制。
"""
from __future__ import annotations

from fractions import Fraction
import math

from pipeline.readiness import GenerationReadinessError


VERSION = 'provider-operation-v1'


def execution_intent(shot: dict) -> dict:
    return {'boundary': shot.get('continuity_from_previous') or 'intentional_cut',
            **({'intent': 'successor-v1'} if shot.get('observed_origin') else {})}


def compile_operation(bindings: list[dict], execution: dict) -> dict:
    if (not isinstance(execution, dict) or set(execution) not in ({'boundary'}, {'boundary', 'intent'})
            or ('intent' in execution and execution['intent'] != 'successor-v1')
            or execution['boundary'] not in ('seamless', 'intentional_cut')):
        raise GenerationReadinessError('素材执行的镜头边界无效；未提交生成')
    states = [b for b in bindings if b.get('usage') in ('selected_state', 'official_tail')]
    if len(states) > 1:
        raise GenerationReadinessError('存在多个动作承接来源；不能猜测先后关系或删除参考')
    operation = {'version': VERSION, 'task': 'reference_generation', 'source': None,
        'camera_boundary': ('cut_to_current_shot' if execution['boundary'] == 'intentional_cut'
                            else 'continuous_camera'),
        'output_scope': 'new_segment', 'transport_control': 'semantic_reference'}
    if not states:
        return operation
    state = states[0]
    operation['source'] = {key: state[key] for key in (
        'logical_index', 'media_label', 'source_task_id', 'source_sha256', 'timestamp', 'source_duration')}
    if state['media_type'] == 'image_url':
        operation.update(task='continue_from_frame', transport_control='image_conditioning')
        return operation
    fps = float(Fraction(state['source_fps']))
    timestamp, duration = state['timestamp'], state['source_duration']
    if (not all(type(v) in (int, float) and math.isfinite(v) for v in (timestamp, duration))
            or not 0 <= timestamp <= duration or not 24 <= fps <= 60):
        raise GenerationReadinessError('动作承接来源的时间范围无效；未提交生成')
    # 观察帧是结束前的采样，不是剪辑终点。使用剪辑选择器保存的终点，
    # 仅容纳一个视频帧的输出量化。历史记录缺终点时，不反推或伪造延长能力。
    end = state.get('selected_end_seconds')
    if end is None and state['usage'] == 'official_tail':
        end = duration
    if end is not None and (type(end) not in (int, float) or not math.isfinite(end)
                           or not timestamp <= end <= duration):
        raise GenerationReadinessError('动作承接的选定范围无效；未提交生成')
    operation['source']['selected_end_seconds'] = end
    terminal = end is not None and duration - end <= 1 / fps + 1e-6
    operation.update(task='extend_after' if terminal else 'continue_from_observation',
                     transport_control='prompt_dispatch' if terminal else 'semantic_reference')
    if execution.get('intent') == 'successor-v1' and execution['boundary'] == 'intentional_cut':
        operation.update(version='provider-operation-v2', task='generate_successor',
                         transport_control='semantic_reference')
    return operation


def operation_prefix(operation: dict) -> str:
    task, source = operation['task'], operation['source']
    if task == 'generate_successor':
        prefix = (f"参考素材生成下一镜的新视频，时间起点承接{source['media_label']}在"
                  f"{source['timestamp']:.3f}秒的状态，执行本镜分配的事件。")
    elif task == 'extend_after':
        prefix = f"向后延长{source['media_label']}，生成它结束之后的新片段。"
    elif task == 'continue_from_observation':
        prefix = (f"从{source['media_label']}在{source['timestamp']:.3f}秒处的已观察状态继续，"
                  '生成新片段；该观察点位于原视频内部，不是原视频末端。')
    elif task == 'continue_from_frame':
        prefix = f"从{source['media_label']}的已观察状态开始生成新片段。"
    else:
        prefix = '使用素材中的指定主体与场景元素，生成独立的新片段。'
    if source:
        prefix += ('在新片段起点切换至当前分镜的构图，并执行当前分镜的运镜。'
                   if operation['camera_boundary'] == 'cut_to_current_shot'
                   else '摄影连续承接，在此基础上执行当前分镜的运镜。')
    return prefix + '\n'
