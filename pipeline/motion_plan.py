"""Offline skeletal blocking bound to, but never replacing, an Action Contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipeline.action_timing import action_timing_issues, sequence_required, timing_required
from pipeline.causality import ACTION_CONTRACT_VERSION, compile_action_contract, compile_interaction_blocking
from pipeline.models import ShotSpec
from pipeline.visual_invariants import safe_body_orientations


MOTION_PLAN_VERSION = "motion-previs-v1"
MOTION_PREVIS_GUIDANCE = (
    "v1 动作参考只支持单镜头、固定机位、独立双足双人、无道具的一次直接接触，"
    "角色采用明确二维侧面朝向。不得删减用户动作、运镜、人物或结局来迎合能力范围；"
    "超出范围的要求应保留，由执行校验拒绝，不能假称已经实现。"
)


def motion_previs_plan_issues(plan: dict) -> list[str]:
    """只检查创意前已知的槽位约束；具体动作仍由 supported_motion_shot 验证。"""
    slots = plan.get("slots", [])
    if not isinstance(slots, list) or len(slots) != 1 or not isinstance(slots[0], dict):
        return ["v1 动作预演仅支持单镜头独立工作区；不将连续镜头拆成互不相干的参考"]
    minimum = slots[0].get("min_interaction_events", 0)
    if type(minimum) is not int or minimum < 0 or minimum > 1:
        return ["v1 动作预演只有一次接触，无法满足生产计划的多个交互事件"]
    return []


# 拓扑顺序保证父关节先求值；长度以画布短边为单位，角度在屏幕平面内。
BONES = {
    "neck": ("pelvis", 0.23, -90.0),
    "head": ("neck", 0.09, -90.0),
    "lead_shoulder": ("neck", 0.06, 0.0),
    "lead_elbow": ("lead_shoulder", 0.14, -45.0),
    "lead_hand": ("lead_elbow", 0.14, -70.0),
    "rear_shoulder": ("neck", 0.06, 180.0),
    "rear_elbow": ("rear_shoulder", 0.14, 120.0),
    "rear_hand": ("rear_elbow", 0.14, 70.0),
    "lead_knee": ("pelvis", 0.17, 70.0),
    "lead_foot": ("lead_knee", 0.17, 100.0),
    "rear_knee": ("pelvis", 0.17, 110.0),
    "rear_foot": ("rear_knee", 0.17, 80.0),
}
CONTACT_TOLERANCE = 0.015
MOTION_EPSILON = 0.001
Angle = Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]
Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Length = Annotated[float, Field(ge=0.02, le=0.4, allow_inf_nan=False)]


class MotionPose(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    frame: int = Field(ge=0, le=449)
    root: list[Coordinate] = Field(min_length=2, max_length=2)
    angles: dict[str, Angle] = Field(default_factory=dict)

    @field_validator("angles")
    @classmethod
    def known_bones(cls, value: dict) -> dict:
        if set(value) - BONES.keys():
            raise ValueError("angles 包含未知骨骼")
        return value


class MotionRig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    entity: str = Field(min_length=1, max_length=128)
    facing: Literal["left", "right"]
    lengths: dict[str, Length] = Field(default_factory=dict)
    keyframes: list[MotionPose] = Field(min_length=2, max_length=64)

    @field_validator("lengths")
    @classmethod
    def known_bones(cls, value: dict) -> dict:
        if set(value) - BONES.keys():
            raise ValueError("lengths 包含未知骨骼")
        return value

    @model_validator(mode="after")
    def ordered_keyframes(self) -> MotionRig:
        frames = [pose.frame for pose in self.keyframes]
        if frames != sorted(set(frames)) or frames[0] != 0:
            raise ValueError("关键帧须从 0 开始、严格递增且不能重复")
        return self


class MotionContact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    frame: int = Field(gt=0, lt=449)
    reaction_frame: int = Field(gt=0, le=449)
    actor_joint: str
    target_joint: str

    @model_validator(mode="after")
    def ordered_contact(self) -> MotionContact:
        if self.reaction_frame <= self.frame:
            raise ValueError("反应帧必须晚于接触帧")
        if any(joint not in BONES for joint in (self.actor_joint, self.target_joint)):
            raise ValueError("接触关节必须是已知骨骼的末端")
        return self


class MotionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal["motion-previs-v1"]
    shot_id: int = Field(gt=0)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    width: int = Field(ge=256, le=1280)
    height: int = Field(ge=256, le=1280)
    fps: Literal[12, 24, 30] = 24
    rigs: list[MotionRig] = Field(min_length=2, max_length=2)
    contact: MotionContact

    @model_validator(mode="after")
    def bounded_canvas(self) -> MotionPlan:
        if self.width % 2 or self.height % 2 or self.width * self.height > 921600:
            raise ValueError("画布须为偶数尺寸，且不超过 921600 像素")
        if len({rig.entity for rig in self.rigs}) != 2:
            raise ValueError("两个骨架必须绑定不同人物")
        return self


def motion_source_binding(shot: dict, ratio: str) -> str:
    """Bind a numeric plan to canonical semantics and the planned camera/state."""
    shot = ShotSpec.model_validate(shot).model_dump(mode="json")
    compile_interaction_blocking(shot)
    contract = compile_action_contract(shot)
    payload = {
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "contract": asdict(contract),
        "shot_id": shot.get("shot_id"),
        "duration": shot.get("duration"),
        "ratio": ratio,
        "camera": shot.get("camera", {}),
        "blocking": shot.get("blocking", {}),
        "characters": shot.get("characters", []),
        "required_visible_entities": shot.get("required_visible_entities", []),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode()).hexdigest()


def supported_motion_shot(shot: dict, ratio: str) -> tuple[str, str]:
    """Fail closed instead of inventing choreography for unsupported contracts."""
    ShotSpec.model_validate(shot)
    # v1 的数值计划只有一组接触/反应帧及裸骨架。约束必须由所有
    # 编译、参考绑定与旧 CLI 共用，不能只在自动规划入口检查。
    if sequence_required(shot):
        raise ValueError("v1 动作预演只有一次接触，不能表达事件序列或轮流交互")
    beats = shot.get("action_beats", [])
    if (sum(beat.get("activity") == "interaction" for beat in beats) > 1
            or sum(beat.get("phase") == "peak" for beat in beats) > 1):
        raise ValueError("v1 动作预演不能把多个交互节拍压缩为一次接触")
    if shot.get("key_props") or shot.get("continuity_props"):
        raise ValueError("v1 骨架没有道具附着与碰撞模型，不能据此控制道具交互")
    contract = compile_action_contract(shot)
    raw = shot.get("interaction_geometry", {})
    if (raw.get("effect_phase"), raw.get("interaction_mode"), raw.get("outcome_scope")) != (
        "active", "direct_contact", "single"
    ):
        raise ValueError("预演目前仅支持 active / direct_contact / single 镜头")
    actor, target = (str(contract.canonical_geometry.get(key, "")).strip()
                     for key in ("actor", "target"))
    if not actor or not target or actor == target:
        raise ValueError("预演需要两个不同且明确的人物")
    entities = set(shot.get("characters", [])) | set(shot.get("required_visible_entities", []))
    if entities != {actor, target}:
        raise ValueError("镜头可见人物必须恰好是 actor 和 target；不支持群体或第三方")
    if not contract.contracted_visible_result:
        raise ValueError("Action Contract 缺少可见动作结果")
    if shot.get("camera", {}).get("primary_movement") not in {"fixed", "static", "locked_off"}:
        raise ValueError("预演目前仅支持明确声明的固定机位")
    if shot.get("continuity_from_previous", "none") != "none":
        raise ValueError("预演目前只支持独立镜头，不能替代已接受的上一镜尾帧")
    try:
        width, height = (int(part) for part in ratio.split(":"))
        if width <= 0 or height <= 0:
            raise ValueError
    except (AttributeError, TypeError, ValueError):
        raise ValueError("分镜 ratio 必须是有效的宽:高") from None
    duration = shot.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, int) or not 4 <= duration <= 15:
        raise ValueError("预演镜头时长须为 4–15 秒的整数")
    return actor, target


def sample_motion_rig(rig: MotionRig, frame: int, width: int, height: int) -> dict:
    """Interpolate absolute bone angles, preserving lengths between keyframes."""
    if not 0 <= frame <= rig.keyframes[-1].frame:
        raise ValueError("采样帧不在关键帧范围内")
    first, last = rig.keyframes[-2:]
    for start, end in zip(rig.keyframes, rig.keyframes[1:]):
        if start.frame <= frame <= end.frame:
            first, last = start, end
            break
    weight = (frame - first.frame) / (last.frame - first.frame)
    short_edge = min(width, height)
    extent = (width / short_edge, height / short_edge)
    points = {"pelvis": [
        (first.root[i] + (last.root[i] - first.root[i]) * weight) * extent[i]
        for i in range(2)
    ]}
    direction = 1 if rig.facing == "right" else -1
    for name, (parent, default_length, default_angle) in BONES.items():
        angle = first.angles.get(name, default_angle)
        delta = (last.angles.get(name, default_angle) - angle + 180) % 360 - 180
        radians = math.radians(angle + delta * weight)
        length = rig.lengths.get(name, default_length)
        points[name] = [
            points[parent][0] + direction * length * math.cos(radians),
            points[parent][1] + length * math.sin(radians),
        ]
    return points


def _motion_timing_issues(shot: dict, plan: MotionPlan, first_contact: int | None,
                          first_reaction: int | None) -> list[str]:
    if not timing_required(shot):
        return []
    issues = action_timing_issues(shot)
    if issues:
        return issues
    windows = [beat for beat in shot["action_beats"] if beat.get("activity") == "interaction"]
    if len(windows) != 1:
        return ["v1 数值接触需要唯一的合同交互时窗"]
    start, end = windows[0]["start_seconds"], windows[0]["end_seconds"]
    contact, reaction = plan.contact.frame / plan.fps, plan.contact.reaction_frame / plan.fps
    if not start <= contact < reaction < end:
        issues.append("数值接触和反应没有同时落在合同的交互时窗内")
    # 几何允许声明时间与采样事件相差一帧，但这不能令实际事件越出合同。
    if first_contact is not None and first_reaction is not None and not (
        start <= first_contact / plan.fps < first_reaction / plan.fps < end
    ):
        issues.append("骨架实际接触和反应没有同时落在合同的交互时窗内")
    return issues


def compile_motion_previs(shot: dict, ratio: str, plan: MotionPlan) -> dict:
    """Return planned geometry and diagnostics, never a semantic acceptance."""
    actor, target = supported_motion_shot(shot, ratio)
    if plan.shot_id != shot["shot_id"] or plan.source_sha256 != motion_source_binding(shot, ratio):
        raise ValueError("预演与分镜绑定不匹配；请核对修改后的合约、机位与时长后重新建模")
    if {rig.entity for rig in plan.rigs} != {actor, target}:
        raise ValueError("骨架人物与 Action Contract 的 actor / target 不匹配")
    ratio_width, ratio_height = map(int, ratio.split(":"))
    if plan.width * ratio_height != plan.height * ratio_width:
        raise ValueError("预演画幅必须与分镜 ratio 一致")
    frame_count = shot["duration"] * plan.fps
    if any(rig.keyframes[-1].frame != frame_count - 1 for rig in plan.rigs):
        raise ValueError("每个骨架的最后关键帧必须覆盖 duration × fps − 1")
    if plan.contact.reaction_frame >= frame_count - 1:
        raise ValueError("反应帧之后必须保留可观察的收势时间")
    frames = [{
        "frame": frame,
        "time_seconds": frame / plan.fps,
        "roles": {"actor": actor, "target": target},
        "entities": {rig.entity: sample_motion_rig(rig, frame, plan.width, plan.height)
                     for rig in plan.rigs},
    } for frame in range(frame_count)]
    issues = []
    orientations = safe_body_orientations(shot.get("blocking"))
    for rig in plan.rigs:
        if orientations.get(rig.entity) != f"profile toward screen-{rig.facing}":
            issues.append(f"{rig.entity} 骨架朝向与分镜不一致，或超出当前二维侧面参考支持范围")
        counterpart = target if rig.entity == actor else actor
        sign = 1 if rig.facing == "right" else -1
        if any(sign * (sample["entities"][counterpart]["pelvis"][0]
                       - sample["entities"][rig.entity]["pelvis"][0]) <= 0 for sample in frames):
            issues.append(f"{rig.entity} 朝向背离交互对象")
    short_edge = min(plan.width, plan.height)
    extent = (plan.width / short_edge, plan.height / short_edge)
    out_of_frame = [sample["frame"] for sample in frames if any(
        not 0.025 <= point[i] <= extent[i] - 0.025
        for joints in sample["entities"].values() for point in joints.values() for i in range(2)
    )]
    if out_of_frame:
        issues.append("骨架出画或过于接近边缘")
    if any(not 0.04 <= sample["entities"][rig.entity]["head"][0] <= extent[0] - 0.04
           for sample in frames for rig in plan.rigs):
        issues.append("头部朝向轮廓出画或过于接近边缘")
    distances = [math.dist(
        sample["entities"][actor][plan.contact.actor_joint],
        sample["entities"][target][plan.contact.target_joint],
    ) for sample in frames]
    contact_frames = [i for i, distance in enumerate(distances) if distance <= CONTACT_TOLERANCE]
    first_contact = contact_frames[0] if contact_frames else None
    if distances[plan.contact.frame] > CONTACT_TOLERANCE:
        issues.append("声明的接触帧上两个接触关节没有相遇")
    if first_contact is not None and abs(first_contact - plan.contact.frame) > 1:
        issues.append("首次接触与声明时间相差超过一帧")
    target_start = frames[0]["entities"][target]
    target_motion = [max(math.dist(point, target_start[name]) for name, point in
                         sample["entities"][target].items()) for sample in frames]
    motion_frames = [i for i, distance in enumerate(target_motion) if distance > MOTION_EPSILON]
    first_reaction = motion_frames[0] if motion_frames else None
    issues.extend(_motion_timing_issues(shot, plan, first_contact, first_reaction))
    # 本切片限定接触前 target 保持防守姿态；不能把主动迎击误认作受击反应。
    if first_reaction is None or max(target_motion) < 0.02:
        issues.append("目标缺少可观察的接触后运动")
    elif first_reaction < plan.contact.reaction_frame or first_reaction <= plan.contact.frame:
        issues.append("目标在约定反应帧或接触之前已经运动")
    elif first_reaction > plan.contact.reaction_frame + 1:
        issues.append("目标实际运动晚于约定反应帧超过一帧")
    actor_start = frames[0]["entities"][actor][plan.contact.actor_joint]
    if max(math.dist(sample["entities"][actor][plan.contact.actor_joint], actor_start)
           for sample in frames[:plan.contact.frame + 1]) < 0.02:
        issues.append("作用方缺少可观察的接近动作")
    ordering = [sample["entities"][target]["pelvis"][0] - sample["entities"][actor]["pelvis"][0]
                for sample in frames]
    if any(value * ordering[0] <= 0 for value in ordering):
        issues.append("人物左右关系发生交换或重叠")
    return {
        "version": MOTION_PLAN_VERSION,
        "source_sha256": plan.source_sha256,
        "coordinate_system": "screen_xy_in_short_edge_units; y_down; fixed_camera",
        "frame_count": frame_count,
        "frames": frames,
        "diagnostics": {
            "status": "invalid" if issues else "valid_geometry",
            "issues": issues,
            "first_contact_frame": first_contact,
            "contact_distance": distances[plan.contact.frame],
            "contact_tolerance": CONTACT_TOLERANCE,
            "first_target_motion_frame": first_reaction,
            "out_of_frame_count": len(out_of_frame),
            "generation_quality": "not_evaluated",
            "semantic_acceptance": "not_evaluated",
        },
    }
