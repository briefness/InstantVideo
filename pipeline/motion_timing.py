"""动作速度与影像时间处理独立于剧情密度；不推算真实画面的速度。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pipeline.intent_evidence import RequiredEvidence, evidence_grounded, has_evidence


class MotionTimingSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    performer_tempo: Literal['fast', 'natural', 'deliberate', 'unspecified']
    playback: Literal['real_time', 'slow_motion', 'unspecified']
    evidence: RequiredEvidence

    @model_validator(mode='after')
    def require_direction(self):
        if not has_evidence(self.evidence):
            raise ValueError('动作速度要求缺少原文依据')
        if self.performer_tempo == self.playback == 'unspecified':
            raise ValueError('未要求动作速度或时间处理时应省略指导')
        return self


def motion_timing_projection(shot: dict) -> dict:
    slot = shot.get('production_slot') or {}
    raw = slot.get('motion_timing')
    if raw is None:
        return {}
    direction = MotionTimingSpec.model_validate(raw).model_dump()
    brief = slot.get('creative_brief')
    if not isinstance(brief, str) or not evidence_grounded(direction['evidence'], brief):
        raise ValueError('动作速度指导必须来自冻结需求原文')
    return direction


def motion_timing_instruction(direction: dict) -> str:
    if not direction:
        return ''
    direction = MotionTimingSpec.model_validate(direction).model_dump()
    tempo, playback = direction['performer_tempo'], direction['playback']
    tempo_text = {
        'fast': 'Use brief accelerations, quick redirections and immediate responses with natural speed variation. '
                'A long interaction window is not permission to slow one gesture across it. ',
        'natural': 'Perform movements at their ordinary physical tempo, with natural acceleration and recovery. ',
        'deliberate': 'Perform deliberately paced movements; this describes the subject, not slowed footage. ',
        'unspecified': '',
    }[tempo]
    playback_text = {
        'real_time': 'Present real-time motion, without slow motion, suspended impacts, replay or speed ramps. ',
        'slow_motion': 'Present the requested slow-motion treatment while preserving the underlying performer tempo. ',
        'unspecified': 'No playback-time treatment was specified; cinematic styling alone does not request slow motion. ',
    }[playback]
    return (f'Motion timing v1: performer_tempo={tempo}; playback={playback}. '
            + tempo_text + playback_text
            + 'Interaction occupancy, event count and camera speed are not evidence of performer speed. ')


PHRASE_SCOPE = (
    'Preserve the ordered plot outcomes and their causal participants. Within authored maneuvers, '
    'connected footwork, recovery, redirection and reactive adjustments may flow through those outcomes; '
    'an event boundary is not a pose, stop or a limit of one motor gesture. '
    'Do not add new consequential strikes, injuries, targets or plot outcomes; those must be authored. '
    'Preserve explicit choreography and reference-motion constraints.'
)

AUTHORING_TIMING = '''
motion_timing来自用户原文，performer_tempo是主体运动速度，playback是影像时间处理；两者与pacing、交互时长、camera.speed分别负责不同问题。fast主体也可用slow_motion拍摄；deliberate动作也可按real_time播放，不互相替代。
先按完整镜头时长设计能以该速度完成的连续表演，再表达其中的因果作用。min_interaction_events是剧情作用的最低数量，不是肢体动作的数量上限；一个作用内可有连续步法、卸力、调整与变向，不要求每个结果停住展示。若连续表演需要新增有后果的作用，使用已有following交互表达原因和结果，不能交给生成器临时发明伤害或结局。
weight只分配内容时间，不能把短促动作拉慢来填满镜头。结局发生前写足必要发展，结果确认按叙事需要简洁呈现；不以反复摆姿势或提前收势填满剩余时间。精确编舞与参考动作保持原路径和顺序，不以自由表演改写。
'''
