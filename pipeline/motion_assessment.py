"""Versioned human observations; geometry and temporal order alone imply no causality."""

from __future__ import annotations

import hashlib
import json
import re
from pipeline.causality import compile_action_contract
from pipeline.visual_invariants import compile_visual_invariants

REVIEW_VERSION = "motion-review-v3"
FRAME_TOLERANCE = 2
FLAGS = ("identity_ok", "scope_ok", "endpoint_ok", "causal_link_ok", "visual_invariants_ok")
REVIEW_INSTRUCTIONS = (
    "使用匿名 reviewer_id。每条设置 assessment 为 unreviewed（未评）、unjudgeable（无法判断）或 reviewed（已评）。"
    "逐帧填写从 0 开始的首次接触和目标首次反应帧号；事件未发生填 null。"
    "identity_ok 检查角色身份和左右方向；scope_ok 检查作用对象与范围；endpoint_ok 检查合约可见终态；"
    "causal_link_ok 要求可见接触与同一目标的后续反应相连，且目标没有提前反应，不能只根据两个帧号或同步运动推断。"
    "visual_invariants_ok 检查合约 visual_invariants 中的身体朝向、固定机位、身份、左右关系等要求；朝向镜头不能代替彼此相向。"
    "已评必须填写五项 true/false；遮挡、身份或因果关系无法判断时选 unjudgeable 并在 notes 说明。"
    "只分享 reviewer 目录，画面自身仍可能暴露分组。"
)


def motion_review_contract(shot: dict) -> dict:
    """人工评审与生成、自动语义验收消费同一份视觉约束。"""
    return {**compile_action_contract(shot).review_projection,
            "visual_invariants": compile_visual_invariants(shot).review_projection}


def contract_digest(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def blank_rating(review_id: str, sha256: str, frames: int) -> dict:
    return {"review_id": review_id, "sha256": sha256, "frames": frames,
            "assessment": "unreviewed", "contact_frame": None, "reaction_frame": None,
            **dict.fromkeys(FLAGS), "notes": ""}


def assess_rating(rating: dict, frames: int, *, legacy: bool = False,
                  version: str = REVIEW_VERSION) -> str:
    """Return pass/fail/unreviewed/unjudgeable without inferring missing evidence."""
    if not isinstance(rating, dict) or type(frames) is not int or not 1 <= frames <= 900:
        raise ValueError("评审结构或视频帧数无效")
    notes = rating.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 4000:
        raise ValueError("评审备注无效或过长")
    for key in ("contact_frame", "reaction_frame"):
        value = rating.get(key)
        if value is not None and (type(value) is not int or not 0 <= value < frames):
            raise ValueError("评审帧号越界或类型错误")
    if version not in ("motion-review-v2", REVIEW_VERSION):
        raise ValueError("不支持的评审版本")
    keys = FLAGS[:3] if legacy else FLAGS[:4] if version == "motion-review-v2" else FLAGS
    flags = [rating.get(key) for key in keys]
    if any(value is not None and type(value) is not bool for value in flags):
        raise ValueError("评审判断必须为 true / false / null")
    if legacy:
        # v1 保留历史结果，调用方明确标为旧版，不能视作通过新版校准。
        state = "unreviewed" if all(value is None for value in flags) else "reviewed"
    else:
        state = rating.get("assessment")
    if state not in ("unreviewed", "unjudgeable", "reviewed"):
        raise ValueError("评审状态无效")
    if state == "unreviewed":
        if any(value is not None for value in flags) or any(rating.get(k) is not None for k in ("contact_frame", "reaction_frame")):
            raise ValueError("已填写证据时须明确设置评审状态")
        return state
    if state == "unjudgeable":
        if not notes.strip():
            raise ValueError("无法判断时必须说明原因")
        return state
    if any(value is None for value in flags):
        raise ValueError("已评审必须完整填写所有判断")
    contact, reaction = rating.get("contact_frame"), rating.get("reaction_frame")
    return "pass" if all(flags) and contact is not None and reaction is not None and contact < reaction else "fail"


def assess_document(document: dict, entries: dict[str, dict], contract: dict) -> dict:
    version = document.get("version")
    legacy = version is None
    if not legacy and version not in ("motion-review-v2", REVIEW_VERSION):
        raise ValueError("不支持的评审版本")
    if version == "motion-review-v2":
        # 历史四项标准按历史合约读取，不补填新证据，也不能参加新版双人共识。
        contract = {key: value for key, value in contract.items() if key != "visual_invariants"}
    if not legacy and document.get("contract_sha256") != contract_digest(contract):
        raise ValueError("评审合约指纹不一致")
    if not legacy and document.get("contract") != contract:
        raise ValueError("评审合约内容改变")
    ratings = document.get("ratings")
    if not isinstance(ratings, list) or len(ratings) > len(entries):
        raise ValueError("评分列表无效")
    verdicts, by_id = {}, {}
    for rating in ratings:
        if not isinstance(rating, dict) or not isinstance(rating.get("review_id"), str):
            raise ValueError("评分结构无效")
        identifier = rating["review_id"]
        if identifier not in entries or identifier in verdicts:
            raise ValueError("评审样本未知或重复")
        entry = entries[identifier]
        if rating.get("sha256") != entry["sha256"] or type(rating.get("frames")) is not int or rating["frames"] != entry["frames"]:
            raise ValueError("评审视频指纹或帧数变化")
        verdicts[identifier] = assess_rating(rating, entry["frames"], legacy=legacy,
                                           version=version or REVIEW_VERSION)
        by_id[identifier] = rating
    reviewer = document.get("reviewer_id", "")
    if not isinstance(reviewer, str):
        raise ValueError("reviewer_id 必须是匿名字符串")
    if not legacy and (reviewer or any(v != "unreviewed" for v in verdicts.values())):
        if not isinstance(reviewer, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", reviewer):
            raise ValueError("请填写 1–64 位字母、数字、下划线组成的匿名 reviewer_id")
    return {"version": "legacy-v1" if legacy else version, "reviewer_id": reviewer,
            "verdicts": verdicts, "ratings": by_id}


def compare_assessments(first: dict, second: dict, identifiers: list[str]) -> dict:
    if (first["version"] != REVIEW_VERSION or second["version"] != REVIEW_VERSION
            or not first["reviewer_id"] or not second["reviewer_id"]
            or first["reviewer_id"] == second["reviewer_id"]):
        raise ValueError("双人复核需要两个不同 reviewer_id 的新版评审")
    verdicts, disagreements, frame_deltas = {}, [], []
    comparable = agreed = 0
    for identifier in identifiers:
        a, b = (doc["verdicts"].get(identifier, "unreviewed") for doc in (first, second))
        if "unreviewed" in (a, b):
            verdicts[identifier] = "unreviewed"
            continue
        comparable += 1
        agreed += a == b
        timing_disagrees = False
        for key in ("contact_frame", "reaction_frame"):
            x, y = (doc["ratings"][identifier].get(key) for doc in (first, second))
            delta = abs(x - y) if x is not None and y is not None else None
            if delta is not None:
                frame_deltas.append({"review_id": identifier, "event": key, "delta_frames": delta})
            timing_disagrees |= (x is None) != (y is None) or (delta is not None and delta > FRAME_TOLERANCE)
        if a != b or (a in {"pass", "fail"} and timing_disagrees):
            verdicts[identifier] = "disputed"
            disagreements.append(identifier)
        else:
            verdicts[identifier] = a
    return {"verdicts": verdicts, "disagreements": disagreements, "frame_deltas": frame_deltas,
            "comparable": comparable, "verdict_agreement": agreed / comparable if comparable else None,
            "frame_tolerance": FRAME_TOLERANCE,
            "note": "不同编号不能证明评审者相互独立；分歧须人工复核，不自动择优或多数表决"}
