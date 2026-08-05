"""Bounded semantic review for generated takes."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Any

from openai import OpenAI

import config
from pipeline.causality import (
    CompiledActionContract,
    compile_action_contract,
    requires_causal_review,
)
from pipeline.narrative import (
    compile_narrative_carriers,
    narrative_review_instruction,
    requires_narrative_review,
)
from tools.ffmpeg_ops import get_video_duration
from tools.frame_extractor import (
    composition_change_is_readable,
    extract_frame,
    frame_structure_similarity,
)


SEMANTIC_REVIEW_VERSION = "continuity-v18-temporal-causality"
_STANDARD_SAMPLING_POLICY = "five-point-v1"
_CAUSAL_SAMPLING_POLICY = "adaptive-nine-point-v1"
_STANDARD_COMPLETION_TOKENS = 4096
_CAUSAL_COMPLETION_TOKENS = 8192
_MAX_RESPONSE_ATTEMPTS = 2
_BOUNDARY_REVIEW_FIELDS = (
    "environment_continuity_valid",
    "action_handoff_valid",
    "screen_direction_valid",
    "prop_continuity_valid",
)
_OBSERVED_STATE_KEYS = (
    "location",
    "subject",
    "action_phase",
    "camera",
    "screen_direction",
    "pose_and_gaze",
    "prop_state",
    "open_motion",
    "lighting",
)


def _strict_bool(value: Any, field: str, *, default: bool | None = None) -> bool:
    if value is None and default is not None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _clean_optional_text(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.casefold() in {"none", "null", "n/a", "无"} else text


def _apply_causal_evidence_verdict(
    parsed: dict[str, Any],
    action_contract: CompiledActionContract,
    causal_evidence: object,
) -> None:
    evidence_issues = action_contract.evidence_issues(causal_evidence)
    if evidence_issues:
        parsed["effect_path_valid"] = False
        parsed["reaction_causality_valid"] = False
        parsed["failure_reason"] = "；".join(evidence_issues)
        return
    parsed["effect_path_valid"] = True
    parsed["reaction_causality_valid"] = True


@dataclass(frozen=True)
class SemanticReview:
    accepted: bool
    required_entities_visible: dict[str, bool]
    action_geometry_valid: bool
    primary_action_completed: bool
    observed_end_state: dict[str, str]
    effect_path_valid: bool = True
    reaction_causality_valid: bool = True
    narrative_state_change_valid: bool = True
    blocking_valid: bool = True
    composition_change_valid: bool = True
    boundary_continuity_valid: bool = True
    identity_continuity_valid: bool = True
    environment_continuity_valid: bool = True
    action_handoff_valid: bool = True
    screen_direction_valid: bool = True
    prop_continuity_valid: bool = True
    identity_crop_boxes: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    failure_reason: str = ""

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        require_causality: bool = False,
        require_narrative: bool = False,
        require_blocking: bool = False,
        require_composition: bool = False,
    ) -> "SemanticReview":
        required = value.get("required_entities_visible", {})
        observed = value.get("observed_end_state", {})
        review = cls(
            accepted=_strict_bool(value.get("accepted"), "accepted"),
            required_entities_visible={
                str(name): _strict_bool(
                    visible, f"required_entities_visible.{name}"
                )
                for name, visible in required.items()
            } if isinstance(required, dict) else {},
            action_geometry_valid=_strict_bool(
                value.get("action_geometry_valid"), "action_geometry_valid"
            ),
            effect_path_valid=_strict_bool(
                value.get("effect_path_valid"),
                "effect_path_valid",
                default=not require_causality,
            ),
            reaction_causality_valid=_strict_bool(
                value.get("reaction_causality_valid"),
                "reaction_causality_valid",
                default=not require_causality,
            ),
            narrative_state_change_valid=_strict_bool(
                value.get("narrative_state_change_valid"),
                "narrative_state_change_valid",
                default=not require_narrative,
            ),
            blocking_valid=_strict_bool(
                value.get("blocking_valid"),
                "blocking_valid",
                default=not require_blocking,
            ),
            composition_change_valid=_strict_bool(
                value.get("composition_change_valid"),
                "composition_change_valid",
                default=not require_composition,
            ),
            primary_action_completed=_strict_bool(
                value.get("primary_action_completed"), "primary_action_completed"
            ),
            observed_end_state={
                key: str(observed.get(key, "")).strip()
                for key in _OBSERVED_STATE_KEYS
            } if isinstance(observed, dict) else {},
            boundary_continuity_valid=_strict_bool(
                value.get("boundary_continuity_valid"),
                "boundary_continuity_valid",
                default=True,
            ),
            identity_continuity_valid=_strict_bool(
                value.get("identity_continuity_valid"),
                "identity_continuity_valid",
                default=True,
            ),
            environment_continuity_valid=_strict_bool(
                value.get("environment_continuity_valid"),
                "environment_continuity_valid",
                default=True,
            ),
            action_handoff_valid=_strict_bool(
                value.get("action_handoff_valid"),
                "action_handoff_valid",
                default=True,
            ),
            screen_direction_valid=_strict_bool(
                value.get("screen_direction_valid"),
                "screen_direction_valid",
                default=True,
            ),
            prop_continuity_valid=_strict_bool(
                value.get("prop_continuity_valid"),
                "prop_continuity_valid",
                default=True,
            ),
            identity_crop_boxes=_parse_crop_boxes(
                value.get("identity_crop_boxes")
            ),
            failure_reason=_clean_optional_text(value.get("failure_reason")),
        )
        objectively_valid = (
            all(review.required_entities_visible.values())
            and review.action_geometry_valid
            and review.effect_path_valid
            and review.reaction_causality_valid
            and review.narrative_state_change_valid
            and review.blocking_valid
            and review.composition_change_valid
            and review.primary_action_completed
            and review.boundary_continuity_valid
            and review.identity_continuity_valid
            and review.environment_continuity_valid
            and review.action_handoff_valid
            and review.screen_direction_valid
            and review.prop_continuity_valid
        )
        reason = "" if objectively_valid else review.failure_reason
        if not objectively_valid and not reason:
            reason = _objective_failure_reason(review)
        return replace(review, accepted=objectively_valid, failure_reason=reason)


def _objective_failure_reason(review: SemanticReview) -> str:
    reasons = []
    missing = [
        name for name, visible in review.required_entities_visible.items()
        if not visible
    ]
    if missing:
        reasons.append("必需主体不可见: " + ", ".join(missing))
    if not review.action_geometry_valid:
        reasons.append("动作空间关系不成立")
    if not review.effect_path_valid:
        reasons.append("作用路径或区域与结果不一致")
    if not review.reaction_causality_valid:
        reasons.append("反应范围不符合因果契约")
    if not review.narrative_state_change_valid:
        reasons.append("镜头未产生约定的可见故事状态变化")
    if not review.blocking_valid:
        reasons.append("角色调度、朝向、视线或动作目标不符合契约")
    if not review.composition_change_valid:
        reasons.append("有意切镜未实现约定的构图变化")
    if not review.primary_action_completed:
        reasons.append("主动作未完成")
    if not review.boundary_continuity_valid:
        reasons.append("与上一镜头边界不连续")
    if not review.identity_continuity_valid:
        reasons.append("角色身份不连续或身份参考帧被污染")
    if not review.environment_continuity_valid:
        reasons.append("同场景环境或地标不连续")
    if not review.action_handoff_valid:
        reasons.append("上一镜开放动作与当前镜头动作交接断裂")
    if not review.screen_direction_valid:
        reasons.append("屏幕运动方向或空间轴不连续")
    if not review.prop_continuity_valid:
        reasons.append("持续道具或武器形态不连续")
    return "；".join(reasons) or "语义验收未通过"


def _parse_crop_boxes(value: Any) -> dict[str, tuple[float, float, float, float]]:
    if not isinstance(value, dict):
        return {}
    boxes = {}
    for name, raw_box in value.items():
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(point) for point in raw_box)
        except (TypeError, ValueError):
            continue
        if (
            0.0 <= x1 < x2 <= 1.0
            and 0.0 <= y1 < y2 <= 1.0
            and x2 - x1 >= 0.1
            and y2 - y1 >= 0.1
        ):
            boxes[str(name)] = (x1, y1, x2, y2)
    return boxes


class SemanticReviewUnavailableError(RuntimeError):
    """The take exists, but semantic acceptance could not be determined."""


class SemanticTakeReviewer:
    """Review a take against its contract and accepted continuity references."""

    def __init__(self, output_dir: str | Path, *, client=None, model: str | None = None):
        self.output_dir = Path(output_dir)
        self.cache_dir = self.output_dir / "semantic_reviews"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.model = model or config.LLM_MODEL

    def accepted_identity_crop_boxes(
        self,
        video_path: str | Path,
    ) -> dict[str, tuple[float, float, float, float]]:
        """Return reviewer-approved identity boxes for this exact local video."""
        path = Path(video_path)
        if not path.is_file():
            return {}
        video_hash = _sha256_file(path)
        cached = _read_cache(self.cache_dir / f"{video_hash}.json")
        if (
            not cached
            or cached.get("video_hash") != video_hash
            or cached["review"].get("accepted") is not True
        ):
            return {}
        return _parse_crop_boxes(cached["review"].get("identity_crop_boxes"))

    def review(
        self,
        video_path: str,
        shot: dict,
        *,
        previous_frame_path: str | None = None,
        identity_reference_paths: list[str] | None = None,
        boundary_context: dict[str, Any] | None = None,
        identity_crop_entities: list[str] | None = None,
    ) -> SemanticReview:
        video_hash = _sha256_file(Path(video_path))
        contract = _review_contract(shot)
        causal_review = requires_causal_review(shot)
        boundary_review = bool(
            previous_frame_path and Path(previous_frame_path).is_file()
        )
        state_handoff_review = bool(
            boundary_review
            and isinstance(boundary_context, dict)
            and boundary_context.get("same_scene") is True
        )
        first_frame_handoff = bool(
            state_handoff_review
            and boundary_context.get("state_reference_role") == "first_frame"
        )
        completion_token_budget = _completion_token_budget(
            causal_review or state_handoff_review
        )
        action_contract = compile_action_contract(shot)
        evidence_review = action_contract.requires_evidence
        narrative_review = requires_narrative_review(shot)
        blocking_review = bool(shot.get("blocking"))
        production_slot = shot.get("production_slot")
        plan_requires_result = bool(
            isinstance(production_slot, dict)
            and production_slot.get("requires_visible_result")
        )
        composition_review = bool(
            previous_frame_path
            and Path(previous_frame_path).is_file()
            and shot.get("continuity_from_previous") == "intentional_cut"
            and shot.get("composition_change") in {"medium", "large"}
        )
        contract["boundary_context"] = boundary_context or {}
        crop_entities = list(dict.fromkeys(identity_crop_entities or []))
        contract["identity_crop_entities"] = crop_entities
        contract_hash = hashlib.sha256(
            json.dumps(contract, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        reference_paths = [
            path
            for path in [previous_frame_path, *(identity_reference_paths or [])]
            if path and Path(path).is_file()
        ]
        evaluator = {
            "model": self.model,
            "reviewer_version": SEMANTIC_REVIEW_VERSION,
            "sampling_policy": (
                _CAUSAL_SAMPLING_POLICY
                if causal_review else _STANDARD_SAMPLING_POLICY
            ),
            "image_detail": config.SEMANTIC_REVIEW_IMAGE_DETAIL,
            "reference_hashes": [
                _sha256_file(Path(path)) for path in reference_paths
            ],
            "identity_crop_entities": crop_entities,
            "composition_metric": "luminance-ssim-v1",
            "composition_sample": (
                "midpoint" if first_frame_handoff else "first-sample"
            ),
            "causal_evidence_policy": (
                "ordered-samples-v2" if evidence_review else "legacy-aggregate"
            ),
            "boundary_evidence_policy": (
                "state-handoff-v1" if state_handoff_review else "not-required"
            ),
            "response_format_policy": "strict-json-schema-v1",
            "max_completion_tokens": completion_token_budget,
        }
        evaluator_hash = hashlib.sha256(
            json.dumps(evaluator, sort_keys=True).encode()
        ).hexdigest()
        cache_path = self.cache_dir / f"{video_hash}.json"
        cached = _read_cache(cache_path)
        if (
            cached
            and cached.get("contract_hash") == contract_hash
            and cached.get("evaluator_hash") == evaluator_hash
        ):
            try:
                cached_review = dict(cached["review"])
                if evidence_review:
                    _apply_causal_evidence_verdict(
                        cached_review,
                        action_contract,
                        cached.get("causal_sample_evidence"),
                    )
                return SemanticReview.from_dict(
                    cached_review,
                    require_causality=causal_review,
                    require_narrative=narrative_review,
                    require_blocking=blocking_review,
                    require_composition=composition_review,
                )
            except ValueError:
                pass

        client = self.client or OpenAI(
            api_key=config.ARK_API_KEY,
            base_url=config.ARK_BASE_URL,
        )
        duration = get_video_duration(video_path)
        timestamps = _sample_timestamps(duration, dense=causal_review)
        required_entities = [
            str(entity) for entity in contract.get("required_visible_entities") or []
        ]
        visibility_order = json.dumps(required_entities, ensure_ascii=False)
        crop_order = json.dumps(crop_entities, ensure_ascii=False)
        causal_fields = (
            "effect_path_valid, reaction_causality_valid, "
            if causal_review else ""
        )
        causal_evidence_field = (
            " Also return causal_sample_evidence in exact current-sample order."
            if evidence_review else ""
        )
        narrative_fields = (
            "narrative_state_change_valid, " if narrative_review else ""
        )
        blocking_fields = "blocking_valid, " if blocking_review else ""
        composition_fields = (
            "composition_change_valid, " if composition_review else ""
        )
        causal_instruction = action_contract.review_instruction
        causal_evidence_instruction = (
            action_contract.evidence_prompt(len(timestamps))
            if evidence_review else ""
        )
        response_format = _review_response_format(
            required_entity_count=len(required_entities),
            identity_crop_count=len(crop_entities),
            action_contract=action_contract,
            sample_count=len(timestamps),
            causal_review=causal_review,
            evidence_review=evidence_review,
            narrative_review=narrative_review,
            blocking_review=blocking_review,
            composition_review=composition_review,
            boundary_review=boundary_review,
            state_handoff_review=state_handoff_review,
        )
        boundary_evidence_instruction = (
            "Return boundary_state_evidence with strict booleans "
            "prior_state_preserved, state_progress_not_reversed, "
            "open_motion_handoff_valid, persistent_entities_preserved, and "
            "scene_identity_preserved. Compare only the accepted previous tail and "
            "the first current sample. state_progress_not_reversed is false when a "
            "completed, consumed, displaced, damaged, fallen, opened, emptied, or "
            "otherwise advanced state reappears in an earlier state. A new camera angle "
            "does not excuse reversal. "
            if state_handoff_review else ""
        )
        narrative_instruction = (
            narrative_review_instruction(contract) if narrative_review else ""
        )
        visible_result_instruction = (
            "When action_beats.visible_result is present, require that exact state change. "
            "Its wording never defines subject count or reaction scope; those come only from "
            "interaction_geometry. "
            if action_contract.contracted_visible_result else ""
        )
        blocking_instruction = (
            "blocking_valid requires every named participant to match its contracted frame "
            "position, body orientation, facing target, eyeline, travel direction, and "
            "action target. Reject a readable effect when the body, weapon, gaze, or effect "
            "source points away from its target. "
            if blocking_review else ""
        )
        plan_instruction = (
            "The production slot reserves this take for a visible story result. Reject "
            "primary_action_completed unless the chronological samples show that result; "
            "camera motion, an emitted effect without consequence, or an inferred off-screen "
            "outcome cannot satisfy the slot. "
            if plan_requires_result else ""
        )
        composition_instruction = (
            "composition_change_valid compares the accepted previous tail with the current "
            + ("temporal midpoint" if first_frame_handoff else "first sample")
            + ". "
            "take. medium requires a clearly readable new shot size or angle; large requires "
            "unmistakably different coverage. Preserving identity and environment does not "
            "excuse copying the previous framing. "
            if composition_review else ""
        )
        with tempfile.TemporaryDirectory(prefix="review-", dir=self.cache_dir) as frame_dir:
            current_images = []
            composition_similarity = None
            composition_sample_index = (
                min(
                    range(len(timestamps)),
                    key=lambda index: abs(timestamps[index] - duration * 0.5),
                )
                if first_frame_handoff else 0
            )
            for index, timestamp in enumerate(timestamps, start=1):
                frame_path = Path(frame_dir) / f"frame_{index}.jpg"
                extract_frame(video_path, str(frame_path), timestamp=timestamp)
                if index - 1 == composition_sample_index and composition_review:
                    composition_similarity = frame_structure_similarity(
                        previous_frame_path,
                        frame_path,
                    )
                current_images.append((timestamp, _image_content(frame_path)))

            content: list[dict[str, Any]] = [{
                "type": "text",
                "text": (
                    "Shot contract:\n"
                    + json.dumps(contract, ensure_ascii=False)
                    + "\nReturn strict JSON booleans for accepted, "
                    + "action_geometry_valid, "
                    + causal_fields
                    + narrative_fields
                    + blocking_fields
                    + composition_fields
                    + "primary_action_completed, boundary_continuity_valid, and "
                    "identity_continuity_valid. When a previous-shot tail is supplied, "
                    "also return strict booleans for environment_continuity_valid, "
                    "action_handoff_valid, screen_direction_valid, and "
                    "prop_continuity_valid. Return observed_end_state with "
                    + ", ".join(_OBSERVED_STATE_KEYS)
                    + ", plus one concise failure_reason. "
                    "required_entities_visible MUST be a JSON boolean array in "
                    f"this exact entity order {visibility_order}; return exactly "
                    f"{len(required_entities)} booleans and do not return an object "
                    "or rename entity IDs. "
                    "identity_crop_boxes MUST be a JSON array in this exact character "
                    f"order {crop_order}. For each character return either null or a "
                    "normalized [x1,y1,x2,y2] box from the frame labelled temporal "
                    "midpoint. A box is allowed only when it contains one clear, mostly "
                    "complete identity character and excludes every other recognizable "
                    "character, crowd member, reflection, and silhouette."
                        + causal_evidence_field
                        + plan_instruction
                        + (
                            " Also return boundary_state_evidence."
                            if state_handoff_review else ""
                        )
                ),
            }]
            if previous_frame_path and Path(previous_frame_path).is_file():
                content.extend([
                    {
                        "type": "text",
                        "text": "Accepted previous-shot tail for boundary continuity:",
                    },
                    _image_content(Path(previous_frame_path)),
                ])
            for index, path in enumerate(identity_reference_paths or [], start=1):
                if not Path(path).is_file():
                    continue
                content.extend([
                    {
                        "type": "text",
                        "text": f"Canonical identity reference {index}:",
                    },
                    _image_content(Path(path)),
                ])
            content.append({
                "type": "text",
                "text": "Current candidate take in chronological order:",
            })
            midpoint = duration * 0.5
            for index, (timestamp, image) in enumerate(current_images, start=1):
                label = (
                    "temporal midpoint"
                    if abs(timestamp - midpoint) <= 0.01
                    else f"sample {index}"
                )
                content.extend([
                    {"type": "text", "text": f"Current {label} at {timestamp:.3f}s:"},
                    image,
                ])

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict film continuity supervisor. Review the candidate "
                        "frames against the shot contract and supplied accepted references. "
                        "Reject identity drift, unmotivated boundary changes, or when a "
                        "required entity is absent or unreadable, the contracted spatial "
                        "relationship is not readable, action direction is wrong, or the "
                        "primary action is not completed. Camera polish cannot compensate "
                        "for failed action geography. "
                        + causal_instruction
                        + causal_evidence_instruction
                        + boundary_evidence_instruction
                        + narrative_instruction
                        + visible_result_instruction
                        + blocking_instruction
                        + composition_instruction
                        + "For a "
                        "same-scene boundary, compare the accepted previous tail only with "
                        "the first current frame: preserve "
                        "recognizable environment landmarks and lighting; continue the prior "
                        "open motion, pose, gaze, and action phase; preserve screen direction "
                        "and the 180-degree axis; preserve persistent props, damage, and exact "
                        "weapon form. A new camera angle or shot size is allowed, but it cannot "
                        "excuse a redesigned street, character, weapon, subject count, or reset "
                        "action. identity_continuity_valid must reject armor, silhouette, color, "
                        "face, or weapon-mount drift even when no canonical identity image exists. "
                        "Return JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]
            parsed = None
            for attempt in range(1, _MAX_RESPONSE_ATTEMPTS + 1):
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    response_format=response_format,
                    max_completion_tokens=completion_token_budget,
                )
                raw, diagnostic, refused = _completion_payload(response)
                if refused:
                    raise SemanticReviewUnavailableError(
                        f"语义验收被服务端拒绝，未重试 ({diagnostic})"
                    )
                try:
                    candidate = json.loads(raw)
                except (TypeError, json.JSONDecodeError) as exc:
                    if attempt < _MAX_RESPONSE_ATTEMPTS:
                        print(
                            "     ⚠ 语义验收响应异常，复核同一视频一次: "
                            f"{diagnostic}"
                        )
                        continue
                    raise SemanticReviewUnavailableError(
                        f"语义验收连续 {_MAX_RESPONSE_ATTEMPTS} 次未返回可解析 JSON "
                        f"({diagnostic}): {exc}"
                    ) from exc
                shape_issues = _response_shape_issues(
                    candidate,
                    required_entities=required_entities,
                    action_contract=action_contract,
                    sample_count=len(timestamps),
                    causal_review=causal_review,
                    evidence_review=evidence_review,
                    narrative_review=narrative_review,
                    blocking_review=blocking_review,
                    composition_review=composition_review,
                    boundary_review=boundary_review,
                    state_handoff_review=state_handoff_review,
                )
                if shape_issues:
                    detail = "；".join(shape_issues)
                    if attempt < _MAX_RESPONSE_ATTEMPTS:
                        print(
                            "     ⚠ 语义验收结构异常，复核同一视频一次: "
                            f"{detail}"
                        )
                        repair = (
                            "Your previous review JSON had schema errors: "
                            f"{detail}. Return the entire review JSON again without "
                            "changing any visual judgment. Correct only its structure. "
                        )
                        if evidence_review:
                            repair += action_contract.evidence_prompt(len(timestamps))
                        messages = [
                            *messages,
                            {"role": "assistant", "content": raw},
                            {"role": "user", "content": repair},
                        ]
                        continue
                    raise SemanticReviewUnavailableError(
                        "语义验收连续 "
                        f"{_MAX_RESPONSE_ATTEMPTS} 次返回了无效结构: {detail}"
                    )
                parsed = candidate
                break
        assert parsed is not None
        causal_evidence = None
        if evidence_review:
            causal_evidence = parsed.get("causal_sample_evidence")
            _apply_causal_evidence_verdict(
                parsed, action_contract, causal_evidence
            )
        boundary_state_evidence = None
        if state_handoff_review:
            boundary_state_evidence = parsed.get("boundary_state_evidence")
            required_boundary_evidence = (
                "prior_state_preserved",
                "state_progress_not_reversed",
                "open_motion_handoff_valid",
                "persistent_entities_preserved",
                "scene_identity_preserved",
            )
            if (
                not isinstance(boundary_state_evidence, dict)
                or any(
                    not isinstance(boundary_state_evidence.get(field), bool)
                    for field in required_boundary_evidence
                )
            ):
                raise SemanticReviewUnavailableError(
                    "语义验收缺少结构化跨镜状态交接证据"
                )
            failed = [
                field for field in required_boundary_evidence
                if not boundary_state_evidence[field]
            ]
            if failed:
                parsed["boundary_continuity_valid"] = False
                if not boundary_state_evidence["scene_identity_preserved"]:
                    parsed["environment_continuity_valid"] = False
                if not boundary_state_evidence["open_motion_handoff_valid"]:
                    parsed["action_handoff_valid"] = False
                if not boundary_state_evidence["persistent_entities_preserved"]:
                    parsed["prop_continuity_valid"] = False
                existing_reason = _clean_optional_text(parsed.get("failure_reason"))
                reason = "跨镜状态交接失败: " + ", ".join(failed)
                parsed["failure_reason"] = "；".join(
                    [reason, *([existing_reason] if existing_reason else [])]
                )
        if composition_review and composition_similarity is not None:
            declared_change = str(shot.get("composition_change"))
            parsed["composition_change_valid"] = bool(
                parsed.get("composition_change_valid")
            ) and composition_change_is_readable(
                declared_change,
                composition_similarity,
            )
        returned_visibility = parsed.get("required_entities_visible")
        if isinstance(returned_visibility, list):
            if (
                len(returned_visibility) != len(required_entities)
                or any(not isinstance(visible, bool) for visible in returned_visibility)
            ):
                raise SemanticReviewUnavailableError(
                    "语义验收实体可见性数组与分镜契约数量或类型不一致"
                )
            parsed["required_entities_visible"] = dict(
                zip(required_entities, returned_visibility)
            )
        elif isinstance(returned_visibility, dict):
            if set(returned_visibility) != set(required_entities):
                raise SemanticReviewUnavailableError(
                    "语义验收实体可见性键与分镜契约不一致；"
                    "已暂停且不会触发视频重拍"
                )
            parsed["required_entities_visible"] = {
                entity: returned_visibility[entity] for entity in required_entities
            }
        else:
            raise SemanticReviewUnavailableError(
                "语义验收实体可见性必须是按契约顺序返回的布尔数组"
            )
        if previous_frame_path and Path(previous_frame_path).is_file():
            missing_boundary = [
                field_name for field_name in _BOUNDARY_REVIEW_FIELDS
                if not isinstance(parsed.get(field_name), bool)
            ]
            if missing_boundary:
                raise SemanticReviewUnavailableError(
                    "语义验收缺少跨镜头边界字段: " + ", ".join(missing_boundary)
                )

        returned_crops = parsed.get("identity_crop_boxes", [])
        normalized_crops = {}
        if isinstance(returned_crops, list) and len(returned_crops) == len(crop_entities):
            normalized_crops = {
                name: box
                for name, box in zip(crop_entities, returned_crops)
                if box is not None
            }
        elif isinstance(returned_crops, dict):
            normalized_crops = {
                name: returned_crops[name]
                for name in crop_entities
                if name in returned_crops and returned_crops[name] is not None
            }
        parsed["identity_crop_boxes"] = normalized_crops
        try:
            review = SemanticReview.from_dict(
                parsed,
                require_causality=causal_review,
                require_narrative=narrative_review,
                require_blocking=blocking_review,
                require_composition=composition_review,
            )
        except ValueError as exc:
            raise SemanticReviewUnavailableError(
                f"语义验收字段类型无效: {exc}"
            ) from exc
        if review.accepted and not all(
            review.observed_end_state.get(key)
            for key in ("location", "subject", "action_phase")
        ):
            raise SemanticReviewUnavailableError(
                "语义验收通过但未返回可交接的 observed_end_state"
            )
        _atomic_write_json(cache_path, {
            "video_hash": video_hash,
            "contract_hash": contract_hash,
            "evaluator_hash": evaluator_hash,
            "evaluator": evaluator,
            "composition_similarity": composition_similarity,
            "causal_sample_evidence": causal_evidence,
            "boundary_state_evidence": boundary_state_evidence,
            "review": asdict(review),
        })
        return review


def _review_contract(shot: dict) -> dict[str, Any]:
    contract = {
        key: shot.get(key)
        for key in (
            "shot_id", "coverage_role", "required_visible_entities",
            "camera", "blocking", "start_state", "characters",
            "scene_id", "continuity_from_previous", "composition_change",
            "extract_character_ref", "production_slot",
        )
    }
    contract.update(compile_action_contract(shot).review_projection)
    contract["narrative_carriers"] = list(compile_narrative_carriers(contract))
    return contract


def _completion_token_budget(causal_review: bool) -> int:
    return (
        _CAUSAL_COMPLETION_TOKENS
        if causal_review
        else _STANDARD_COMPLETION_TOKENS
    )


def _response_shape_issues(
    value: object,
    *,
    required_entities: list[str],
    action_contract: CompiledActionContract,
    sample_count: int,
    causal_review: bool,
    evidence_review: bool,
    narrative_review: bool,
    blocking_review: bool,
    composition_review: bool,
    boundary_review: bool,
    state_handoff_review: bool,
) -> list[str]:
    if not isinstance(value, dict):
        return ["响应必须是 JSON 对象"]

    required_booleans = [
        "accepted",
        "action_geometry_valid",
        "primary_action_completed",
    ]
    if causal_review and not evidence_review:
        required_booleans.extend([
            "effect_path_valid",
            "reaction_causality_valid",
        ])
    if narrative_review:
        required_booleans.append("narrative_state_change_valid")
    if blocking_review:
        required_booleans.append("blocking_valid")
    if composition_review:
        required_booleans.append("composition_change_valid")
    if boundary_review:
        required_booleans.extend(_BOUNDARY_REVIEW_FIELDS)
    issues = [
        f"{field} 必须是布尔值"
        for field in required_booleans
        if not isinstance(value.get(field), bool)
    ]

    visibility = value.get("required_entities_visible")
    if isinstance(visibility, list):
        if len(visibility) != len(required_entities):
            issues.append(
                "required_entities_visible 应有 "
                f"{len(required_entities)} 项，实际 {len(visibility)} 项"
            )
        elif any(not isinstance(item, bool) for item in visibility):
            issues.append("required_entities_visible 必须只包含布尔值")
    elif isinstance(visibility, dict):
        if set(visibility) != set(required_entities):
            issues.append(
                "实体可见性 required_entities_visible 的实体键与分镜契约不一致"
            )
        elif any(not isinstance(item, bool) for item in visibility.values()):
            issues.append("required_entities_visible 的实体值必须是布尔值")
    else:
        issues.append("required_entities_visible 必须是布尔数组或实体布尔映射")

    if not isinstance(value.get("observed_end_state"), dict):
        issues.append("observed_end_state 必须是对象")
    if evidence_review:
        issues.extend(action_contract.evidence_schema_issues(
            value.get("causal_sample_evidence"),
            sample_count,
        ))
    if state_handoff_review:
        boundary_evidence = value.get("boundary_state_evidence")
        required_boundary_evidence = (
            "prior_state_preserved",
            "state_progress_not_reversed",
            "open_motion_handoff_valid",
            "persistent_entities_preserved",
            "scene_identity_preserved",
        )
        if not isinstance(boundary_evidence, dict):
            issues.append("boundary_state_evidence 必须是对象")
        else:
            issues.extend(
                f"boundary_state_evidence.{field} 必须是布尔值"
                for field in required_boundary_evidence
                if not isinstance(boundary_evidence.get(field), bool)
            )
    return issues


def _review_response_format(
    *,
    required_entity_count: int,
    identity_crop_count: int,
    action_contract: CompiledActionContract,
    sample_count: int,
    causal_review: bool,
    evidence_review: bool,
    narrative_review: bool,
    blocking_review: bool,
    composition_review: bool,
    boundary_review: bool,
    state_handoff_review: bool,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "accepted": {"type": "boolean"},
        "required_entities_visible": {
            "type": "array",
            "items": {"type": "boolean"},
            "minItems": required_entity_count,
            "maxItems": required_entity_count,
        },
        "action_geometry_valid": {"type": "boolean"},
        "primary_action_completed": {"type": "boolean"},
        "boundary_continuity_valid": {"type": "boolean"},
        "identity_continuity_valid": {"type": "boolean"},
        "observed_end_state": {
            "type": "object",
            "properties": {
                field: {"type": "string"}
                for field in _OBSERVED_STATE_KEYS
            },
            "required": list(_OBSERVED_STATE_KEYS),
            "additionalProperties": False,
        },
        "identity_crop_boxes": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "array",
                        "items": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "minItems": 4,
                        "maxItems": 4,
                    },
                ]
            },
            "minItems": identity_crop_count,
            "maxItems": identity_crop_count,
        },
        "failure_reason": {"type": "string"},
    }
    if causal_review:
        properties.update({
            "effect_path_valid": {"type": "boolean"},
            "reaction_causality_valid": {"type": "boolean"},
        })
    if evidence_review:
        properties["causal_sample_evidence"] = (
            action_contract.evidence_json_schema(sample_count)
        )
    if narrative_review:
        properties["narrative_state_change_valid"] = {"type": "boolean"}
    if blocking_review:
        properties["blocking_valid"] = {"type": "boolean"}
    if composition_review:
        properties["composition_change_valid"] = {"type": "boolean"}
    if boundary_review:
        properties.update({
            field: {"type": "boolean"}
            for field in _BOUNDARY_REVIEW_FIELDS
        })
    if state_handoff_review:
        properties["boundary_state_evidence"] = {
            "type": "object",
            "properties": {
                field: {"type": "boolean"}
                for field in (
                    "prior_state_preserved",
                    "state_progress_not_reversed",
                    "open_motion_handoff_valid",
                    "persistent_entities_preserved",
                    "scene_identity_preserved",
                )
            },
            "required": [
                "prior_state_preserved",
                "state_progress_not_reversed",
                "open_motion_handoff_valid",
                "persistent_entities_preserved",
                "scene_identity_preserved",
            ],
            "additionalProperties": False,
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_take_review",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def _completion_payload(response: Any) -> tuple[str, str, bool]:
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    raw = content if isinstance(content, str) else ""
    finish_reason = str(getattr(choice, "finish_reason", None) or "unknown")
    refusal = getattr(message, "refusal", None)
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    diagnostic = (
        f"finish_reason={finish_reason}, content_length={len(raw)}, "
        f"completion_tokens={completion_tokens if completion_tokens is not None else 'unknown'}"
    )
    refused = bool(refusal) or finish_reason in {"content_filter", "safety"}
    return raw, diagnostic, refused


def _sample_timestamps(duration: float, *, dense: bool = False) -> tuple[float, ...]:
    end = max(0.1, duration - 0.1)
    fractions = (
        (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
        if dense else (0.25, 0.5, 0.75)
    )
    return tuple(dict.fromkeys(round(value, 3) for value in (
        0.1,
        *(max(0.1, duration * fraction) for fraction in fractions),
        end,
    )))


def _image_content(path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encoded}",
            "detail": config.SEMANTIC_REVIEW_IMAGE_DETAIL,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and isinstance(value.get("review"), dict) else None


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
