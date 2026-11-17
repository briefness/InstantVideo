"""Bounded semantic review for generated takes."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from openai import OpenAI

import config
from pipeline.llm_options import structured_output_options
from pipeline.structured_response import load_unique_json, StructuredResponseError, completion_response_record
from pipeline.event_assessment import assess_event_evidence
from pipeline.review_observability import (
    OBSERVABILITY_INSTRUCTION, nullable_observation_schema, unobservable_fields,
    project_boundary_observations,
)
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
from pipeline.visual_invariants import compile_visual_invariants
from tools.ffmpeg_ops import get_video_duration
from tools.frame_extractor import (
    composition_change_is_readable,
    extract_frame,
    frame_structure_similarity,
)


SEMANTIC_REVIEW_VERSION = "continuity-v29-sequence-reuse"
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
    visual_invariants_valid: bool = True
    action_timing_valid: bool = True
    requested_outcome_valid: bool = True
    action_sequence_valid: bool = True
    creative_brief_valid: bool = True
    identity_crop_boxes: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    failure_reason: str = ""
    event_assessment: dict = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        *,
        require_causality: bool = False,
        require_narrative: bool = False,
        require_blocking: bool = False,
        require_composition: bool = False,
        require_visual_invariants: bool = False,
        require_timing: bool = False,
        require_outcome: bool = False,
        require_sequence: bool = False,
        require_creative_brief: bool = False,
        require_observability: bool = False,
    ) -> "SemanticReview":
        if require_observability and any(name not in value for name in ('identity_continuity_valid', 'boundary_continuity_valid')):
            raise SemanticReviewUnavailableError('身份或连续性观察缺失；保留原片，不自动重拍')
        boolean_schema = {'type': 'object', 'properties': {
            name: {'type': 'boolean'} for name, definition in cls.__dataclass_fields__.items()
            if definition.type in (bool, 'bool')
        }}
        boolean_schema['properties']['required_entities_visible'] = {
            'type': 'array', 'items': {'type': 'boolean'}}
        if unobservable_fields(value, boolean_schema):
            raise SemanticReviewUnavailableError('画面观察不足，不能确认验收；保留原片，不自动重拍')
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
            visual_invariants_valid=_strict_bool(
                value.get("visual_invariants_valid"),
                "visual_invariants_valid",
                default=not require_visual_invariants,
            ),
            action_timing_valid=_strict_bool(value.get('action_timing_valid'), 'action_timing_valid',
                                            default=not require_timing),
            requested_outcome_valid=_strict_bool(value.get('requested_outcome_valid'), 'requested_outcome_valid',
                                                default=not require_outcome),
            action_sequence_valid=_strict_bool(value.get('action_sequence_valid'), 'action_sequence_valid',
                                               default=not require_sequence),
            creative_brief_valid=_strict_bool(value.get('creative_brief_valid'), 'creative_brief_valid',
                                              default=not require_creative_brief),
            identity_crop_boxes=_parse_crop_boxes(
                value.get("identity_crop_boxes")
            ),
            failure_reason=_clean_optional_text(value.get("failure_reason")),
            event_assessment=dict(value.get('event_assessment') or {}),
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
            and review.visual_invariants_valid
            and review.action_timing_valid
            and review.requested_outcome_valid
            and review.action_sequence_valid
            and review.creative_brief_valid
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
    if not review.visual_invariants_valid:
        reasons.append("镜头违反声明的环境、主体、道具、灯光或空间不变量")
    if not review.action_timing_valid:
        reasons.append("有效动作时间或结果建立与持续确认不符合时间契约")
    if not review.requested_outcome_valid:
        reasons.append("未实现用户明确要求的可见结局；完成分镜动作不能替代用户目标")
    if not review.action_sequence_valid:
        reasons.append("有序动作事件的执行者、作用、反应或时间窗未逐一成立")
    if not review.creative_brief_valid:
        reasons.append("未满足本镜应呈现的用户原始要求；编舞自由不能取消明确要求")
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

    def __init__(self, output_dir: str | Path, *, client=None, model: str | None = None,
                 review_mode: str = 'strict'):
        if review_mode not in {'strict', 'normal'}:
            raise ValueError('无效审查模式')
        self.review_mode = review_mode
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
        if not cached or cached.get("video_hash") != video_hash:
            return {}
        if cached['review'].get('accepted') is not True:
            from pipeline.review_policy import decide_review
            if self.review_mode != 'normal' or cached.get('evaluator', {}).get('delivery_review_mode') != 'normal-v1':
                return {}
            try:
                if not decide_review(SemanticReview.from_dict(cached['review']), 'normal').allowed:
                    return {}
            except (ValueError, TypeError):
                return {}
        return _parse_crop_boxes(cached["review"].get("identity_crop_boxes"))

    def review(
        self,
        video_path: str,
        shot: dict,
        *,
        previous_frame_path: str | None = None,
        identity_reference_paths: list[str] | None = None,
        identity_reference_entities: list[str] | None = None,
        boundary_context: dict[str, Any] | None = None,
        identity_crop_entities: list[str] | None = None,
        previous_video_path: str | None = None,
        previous_shot: dict | None = None,
    ) -> SemanticReview:
        supplied_references = [path for path in [previous_frame_path, *(identity_reference_paths or [])]
                               if path is not None]
        if any(not path or not Path(path).is_file() for path in supplied_references):
            raise SemanticReviewUnavailableError('已指定的验收参考素材缺失；保留原片，不自动重拍')
        if identity_reference_entities is not None and (
            len(identity_reference_entities) != len(identity_reference_paths or [])
            or any(not isinstance(name, str) or not name.strip() for name in identity_reference_entities)
            or len(set(identity_reference_entities)) != len(identity_reference_entities)
        ):
            raise SemanticReviewUnavailableError('身份参考与人物绑定不完整或重复；未开始评审')
        video_hash = _sha256_file(Path(video_path))
        contract = _review_contract(shot)
        from pipeline.media_selection import media_selection_policy, resolve_media_window
        selection = media_selection_policy(shot)
        window = resolve_media_window(shot, video_path)
        if selection:
            contract['media_selection'] = selection
        invariant_review = bool(contract.get("visual_invariants"))
        causal_review = requires_causal_review(shot)
        boundary_review = bool(
            previous_frame_path and Path(previous_frame_path).is_file()
        )
        from pipeline.temporal_evidence import enabled, sample_clip, combine, INSTRUCTION as TEMPORAL_INSTRUCTION
        temporal_review = enabled(shot)
        previous_history = None
        if temporal_review:
            contract['continuity_evidence'] = shot['continuity_evidence']
            if boundary_review:
                if not previous_video_path or previous_shot is None:
                    raise SemanticReviewUnavailableError('跨镜动态验收缺少上一镜原片；不能只凭尾帧判断动作')
                previous_history = sample_clip(previous_video_path, previous_shot,
                    scope='previous_clip' if shot.get('observed_origin') else 'previous_tail')
                contract['previous_temporal_evidence'] = previous_history['manifest']
                if shot.get('observed_origin'):
                    from tools.sequence_reuse import measure_reuse
                    try:
                        reuse = measure_reuse(previous_video_path, video_path,
                                              previous_shot['duration'], shot['duration'])
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        raise SemanticReviewUnavailableError('跨镜连续画面测量不可用；保留原片，不重拍') from exc
                    reuse_dir = self.cache_dir / 'sequence_reuse'
                    reuse_dir.mkdir(exist_ok=True)
                    _atomic_write_json(reuse_dir / f'{video_hash}.json', reuse)
                    if reuse['status'] == 'replayed_prefix':
                        return SemanticReview(accepted=False, required_entities_visible={},
                            action_geometry_valid=True, primary_action_completed=False,
                            observed_end_state={}, action_handoff_valid=False, boundary_continuity_valid=False,
                            failure_reason=f"跨镜动态画面重演：本镜开头{reuse['matched_seconds']:g}秒与上一镜"
                                f"{reuse['previous_start_seconds']:g}秒起的已用序列近似复制，不能作为新的动作推进")
        state_handoff_review = bool(
            boundary_review
            and isinstance(boundary_context, dict)
            and boundary_context.get("same_scene") is True
        )
        first_frame_handoff = bool(
            state_handoff_review
            and boundary_context.get("state_reference_role") == "first_frame"
            # 派生首帧已经执行切镜，不能仍假设它复制了上一镜构图而推迟取证。
            and not boundary_context.get('reference_view_version')
        )
        completion_token_budget = _completion_token_budget(
            causal_review or state_handoff_review
        )
        action_contract = compile_action_contract(shot)
        from pipeline.result_observations import (decode_observations, INSTRUCTION as RESULT_INSTRUCTION,
            SNAPSHOT_INSTRUCTION, LINKED_STATE_INSTRUCTION, ObservationUncertaintyError)
        completion = action_contract.review_projection.get('completion_contract')
        from pipeline.observation_support import verify_support, apply_support, SupportJudgmentError
        support_evidence = None
        action_events = action_contract.review_projection.get('action_events') or []
        if completion and completion.get('intent_policy'):
            causal_review = bool(action_events)
        outcome_review = (action_contract.review_projection.get('performance_policy') or {}).get('version') == 'performance-v2-outcomes'
        evidence_review = action_contract.requires_evidence and not action_events and not completion
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
        from pipeline.performance import scene_frame_execution
        from tools.camera_measurement import VERSION as CAMERA_METRIC, measure_video_composition, composition_verdict
        geometry_review = composition_review and scene_frame_execution(shot)
        contract["boundary_context"] = boundary_context or {}
        contract['observation_scope'] = {'cross_shot_boundary': boundary_review}
        crop_entities = list(dict.fromkeys(identity_crop_entities or []))
        contract["identity_crop_entities"] = crop_entities
        if identity_reference_entities is not None:
            contract['identity_reference_entities'] = identity_reference_entities
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
                'outcome-coverage-4fps-max64-v1' if outcome_review else _CAUSAL_SAMPLING_POLICY
                if causal_review else _STANDARD_SAMPLING_POLICY
            ),
            "image_detail": config.SEMANTIC_REVIEW_IMAGE_DETAIL,
            "reference_hashes": [
                _sha256_file(Path(path)) for path in reference_paths
            ],
            "identity_crop_entities": crop_entities,
            "composition_metric": CAMERA_METRIC if geometry_review else "luminance-ssim-v1",
            "composition_sample": (
                "midpoint" if first_frame_handoff else "first-sample"
            ),
            "causal_evidence_policy": (
                "ordered-events-v1" if action_events else
                "ordered-samples-v2" if evidence_review else "legacy-aggregate"
            ),
            "boundary_evidence_policy": (
                "state-handoff-v1" if state_handoff_review else "not-required"
            ),
            "response_format_policy": "strict-json-schema-v1",
            "max_completion_tokens": completion_token_budget,
        }
        if window:
            evaluator['media_selection'] = window
        if self.review_mode == 'normal':
            evaluator['delivery_review_mode'] = 'normal-v1'
        if completion:
            evaluator['response_format_policy'] = 'functional-observations-v1'
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
                if geometry_review:
                    measurement = cached.get('composition_measurement')
                    at = (min(cached['sample_timestamps'], key=lambda t: abs(t - cached['media_duration'] * .5))
                          if first_frame_handoff else cached['sample_timestamps'][0])
                    actual = measure_video_composition(previous_frame_path, video_path, at)
                    if measurement != actual:
                        raise ValueError('镜头几何证据与原片不一致')
                    raw_camera = (cached.get('observation_response') or cached_review).get('composition_change_valid')
                    cached_review['composition_change_valid'] = composition_verdict(raw_camera, actual)
                if completion:
                    cached_schema = _review_response_format(
                        required_entity_count=len(contract.get('required_visible_entities') or []),
                        identity_crop_count=len(crop_entities), action_contract=action_contract,
                        sample_count=len(cached['sample_timestamps']), causal_review=causal_review,
                        evidence_review=evidence_review, narrative_review=narrative_review,
                        blocking_review=blocking_review, composition_review=composition_review,
                        invariant_review=invariant_review, boundary_review=boundary_review,
                        state_handoff_review=state_handoff_review)['json_schema']['schema']
                    if cached_schema != cached.get('observation_schema'):
                        raise ValueError('观察协议缓存发生变化')
                    observation = cached.get('observation_response')
                    if cached.get('support_evidence') is not None:
                        visual_manifest = (combine(previous_history, sample_clip(video_path, shot,
                            scope='current_clip', timestamps=cached['sample_timestamps']))['manifest']
                            if temporal_review else None)
                        observation = apply_support(completion, observation, cached['support_evidence'],
                                                    visual_manifest=visual_manifest)
                    elif cached_review.get('accepted') or (self.review_mode == 'normal'
                            and cached_review.get('primary_action_completed')):
                        raise ValueError('通过记录缺少独立观察支持证据')
                    normalized = decode_observations(observation, completion,
                        cached_schema, len(cached['sample_timestamps']))
                    for key in ('accepted', 'primary_action_completed', 'narrative_state_change_valid',
                                'observed_end_state', 'failure_reason'):
                        cached_review[key] = normalized[key]
                    # 恢复与首次审查遵循相同的证据合取：原始聚合标签和几何
                    # 都不能覆盖逐项观察的否定，也不能让新肯定抹去其他反证。
                    for key in ('boundary_continuity_valid', 'composition_change_valid'):
                        if key in normalized and normalized[key] is not True:
                            cached_review[key] = normalized[key]
                    if normalized.get('action_event_evidence') != cached.get('action_event_evidence'):
                        raise SemanticReviewUnavailableError('缓存事件结论与原始观察不一致；未重新审查或重拍')
                gaps = unobservable_fields(cached.get('action_event_evidence'), action_contract.event_evidence_schema())
                if evidence_review:
                    gaps.extend(unobservable_fields(cached.get('causal_sample_evidence'),
                        action_contract.evidence_json_schema(len(cached.get('sample_timestamps') or []))))
                if gaps:
                    raise SemanticReviewUnavailableError('缓存画面观察不足；保留原片，不自动重拍')
                if evidence_review:
                    _apply_causal_evidence_verdict(
                        cached_review,
                        action_contract,
                        cached.get("causal_sample_evidence"),
                    )
                if action_events:
                    _apply_event_evidence_verdict(cached_review, action_contract, cached.get('action_event_evidence'),
                        timestamps=cached.get('sample_timestamps'), media_duration=cached.get('media_duration'),
                        defer_timing_uncertainty=self.review_mode == 'normal')
                return SemanticReview.from_dict(
                    cached_review,
                    require_causality=causal_review,
                    require_narrative=narrative_review,
                    require_blocking=blocking_review,
                    require_composition=composition_review,
                    require_visual_invariants=invariant_review,
                    require_timing=bool(action_contract.review_projection.get('action_timing')),
                    require_outcome=bool(action_contract.review_projection.get('required_final_outcome')),
                    require_sequence=bool(action_events),
                    require_creative_brief=bool(action_contract.review_projection.get('performance_policy')),
                    require_observability=True,
                )
            except (ValueError, KeyError, TypeError):
                if completion:
                    raise SemanticReviewUnavailableError('同源观察缓存无效；保留原片，未重复调用审查') from None
                pass

        unavailable_path = self.cache_dir / f'{video_hash}.evidence.json'
        unavailable = _read_cache(unavailable_path, payload_field='response')
        if unavailable_path.exists() and unavailable is None:
            raise SemanticReviewUnavailableError('已保存的观察记录损坏；保留原片，未重复调用审查')
        if (unavailable and unavailable.get('status') == 'unavailable'
                and unavailable.get('video_hash') == video_hash
                and unavailable.get('contract_hash') == contract_hash
                and unavailable.get('evaluator_hash') == evaluator_hash):
            if completion:
                if unavailable.get('failure_kind') in ('semantic_uncertainty', 'evidence_conflict'):
                    raise SemanticReviewUnavailableError(unavailable['failure_reason'])
                raise SemanticReviewUnavailableError('已有未完成或不可用的观察记录；保留原片，未重复调用审查或生成')
            assessment = unavailable.get('event_assessment')
            if assessment is not None:
                if not isinstance(assessment, dict) or not isinstance(assessment.get('timing'), dict):
                    raise SemanticReviewUnavailableError('已保存的事件测量记录损坏；未重复调用审查')
                if assessment['timing'].get('status') in ('invalid', 'uncertain'):
                    raise SemanticReviewUnavailableError(unavailable.get('failure_reason')
                        or '已有不可用时间证据；保留原片，未重复调用审查或生成')

        if completion:
            from pipeline.observation_support import assert_support_available
            try:
                assert_support_available(self.cache_dir, video_hash)
            except (ValueError, TypeError, KeyError) as exc:
                raise SemanticReviewUnavailableError(str(exc)) from exc
        client = self.client or OpenAI(
            api_key=config.ARK_API_KEY,
            base_url=config.ARK_BASE_URL,
            **({'max_retries': 0} if completion else {}),
        )
        duration = window['duration'] if window else get_video_duration(video_path)
        timestamps = _sample_timestamps(duration, dense=causal_review)
        timing = action_contract.review_projection.get('action_timing')
        if timing:
            # 覆盖每段动作内部，尤其是很短的结果保持区；不只按全片等距抽样。
            centers = [(item['start_seconds'] + item['end_seconds']) / 2 for item in timing['intervals']]
            timestamps = tuple(sorted(set((*timestamps, *(round(min(duration - .05, t), 3) for t in centers)))))
        if action_events:
            event_times = [t for e in action_events for t in (
                e['effect_at_seconds'], e['reaction_at_seconds'],
                (e['reaction_at_seconds'] + e['end_seconds']) / 2)]
            event_times.extend(t for e in action_events if e.get('flow_from_previous')
                               for t in (e['start_seconds'] - .1, e['start_seconds'] + .1))
            timestamps = tuple(sorted(set((*timestamps, *(round(max(0, min(duration - .05, t)), 3) for t in event_times)))))
        if outcome_review:
            # Uniform coverage makes gaps visible independently of the planned choreography.
            # Bound vision payloads for longer clips; the actual gaps remain in the evidence.
            count = min(64, max(2, int(duration * 4) + 1))
            timestamps = tuple(round(min(duration - .05, i * duration / (count - 1)), 3)
                               for i in range(count))
        if window:
            # 末端观察与续拍使用同一个仍在剪辑区间内的样本。
            timestamps = tuple(sorted(set([*(t for t in timestamps if 0 <= t <= window['last_sample_seconds']),
                                           window['last_sample_seconds']])))
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
        invariant_fields = (
            "visual_invariants_valid, " if invariant_review else ""
        )
        causal_instruction = action_contract.review_instruction
        if action_events:
            causal_instruction = action_contract.event_review_instruction()
        if action_contract.review_projection.get('required_final_outcome'):
            causal_instruction += (
                ' Return requested_outcome_valid=false unless the actual frames visibly satisfy '
                'required_final_outcome from the user, independently of the storyboard-authored '
                'endpoint and causal evidence. A weaker storyboard outcome cannot replace it. '
            )
        if timing and not action_contract.review_projection.get('performance_policy'):
            causal_instruction += (
                ' Return action_timing_valid=false if interaction windows are mostly posing, '
                'waiting or camera-only movement, if actions occur out of order, or if the '
                'final held outcome has not first been visibly established. Judge the actual '
                'timestamped samples, not the schedule text. Sparse samples do not certify '
                'frame-exact timing; reject when the required activity is not visibly supported. '
            )
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
            invariant_review=invariant_review,
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
            "Body orientation is an initial camera-relative "
            "constraint: subjects may turn during contracted actions even with a fixed camera; "
            "its apparent view may also evolve with a moving camera, while physical targeting "
            "and the declared screen-axis rule must remain coherent. "
            if blocking_review else ""
        )
        if completion and completion.get('intent_policy') and blocking_review:
            blocking_instruction = (
                'blocking_valid checks readable physical targeting and coherent spatial relations '
                'needed by completion_contract and action_events. Do not require a planned resting '
                'orientation, gaze or travel path unless an admitted goal or fixed state requires it. '
                'An effect aimed away from its actual target remains invalid. ')
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
        if geometry_review:
            composition_instruction = (
                'composition_change_valid checks the current camera framing and angle against camera in '
                'the shot contract, using the previous tail and the indicated current sample. '
                'Describe the actual framing in end_state_context.camera, not the planned label. '
                'Subject movement, pose, lighting and object displacement alone are not reframing. '
                'A tighter shot must actually change subject scale/coverage; copying a wide composition '
                'cannot satisfy a medium or detail shot. Independent geometry measurements also test this claim. ')
        invariant_instruction = (
            "visual_invariants_valid is true only when every declared visual invariant "
            "holds across the chronological samples. Reject environment replacement, "
            "disappearance of required entities or continuity props, duplication of a "
            "declared individual identity, identity or relative-scale drift, undeclared "
            "prop-form changes, unrelated lighting replacement, or an incorrect initial "
            "screen position. A prop may change only when the Action Contract visibly "
            "requires it. "
            if invariant_review else ""
        )
        if scene_frame_execution(shot):
            invariant_instruction = (
                'visual_invariants_valid checks setting, lighting, identity, object form and required '
                'spatial continuity. visibility_scope separates world persistence from required frame visibility. '
                'Off-screen persistent props retain their world state and need not be visible in every '
                'camera view; do not require inventory-wide framing. This does not excuse absent required '
                'subjects, wrong spatial results, unexplained world-state changes or off-screen inferred completion. ')
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
            composition_measurement = (measure_video_composition(previous_frame_path, video_path,
                timestamps[composition_sample_index]) if geometry_review else None)
            if geometry_review:
                # 独立物理证据不依赖后续语义服务是否可用；不能被其他未决项遮蔽。
                geometry_dir = self.cache_dir / 'geometry'
                geometry_dir.mkdir(parents=True, exist_ok=True)
                _atomic_write_json(geometry_dir / f'{video_hash}.json', {
                    'video_hash': video_hash, 'contract_hash': contract_hash, 'evaluator_hash': evaluator_hash,
                    'composition_measurement': composition_measurement})
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
                    + "\nReturn JSON boolean-or-null observations for accepted, "
                    + "action_geometry_valid, "
                    + causal_fields
                    + narrative_fields
                    + blocking_fields
                    + composition_fields
                    + invariant_fields
                    + "primary_action_completed, and "
                    "identity_continuity_valid. When a previous-shot tail is supplied, "
                    "also return boolean-or-null observations for boundary_continuity_valid, environment_continuity_valid, "
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
            if previous_history:
                content.extend([{'type': 'text', 'text': TEMPORAL_INSTRUCTION}, *previous_history['content']])
            for index, path in enumerate(identity_reference_paths or [], start=1):
                label = (json.dumps(identity_reference_entities[index - 1], ensure_ascii=False)
                         if identity_reference_entities is not None else str(index))
                content.extend([
                    {
                        "type": "text",
                        "text": f"Canonical identity reference {index} for character {label}:",
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
                        "required entity is visibly absent, the contracted spatial "
                        "relationship is visibly violated, action direction is wrong, or the "
                        "primary action is not completed. Camera polish cannot compensate "
                        "for failed action geography. "
                        + causal_instruction
                        + causal_evidence_instruction
                        + boundary_evidence_instruction
                        + narrative_instruction
                        + visible_result_instruction
                        + blocking_instruction
                        + composition_instruction
                        + invariant_instruction
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
                        "Return JSON only. " + OBSERVABILITY_INSTRUCTION
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]
            if completion:
                # 新协议不同时要求模型返回可相互矛盾的聚合结论与交接事实。
                messages[0]['content'] = ((RESULT_INSTRUCTION.replace(LINKED_STATE_INSTRUCTION, SNAPSHOT_INSTRUCTION)
                    if completion.get('intent_policy') in ('source-goals-v1', 'goal-allocation-v1') else RESULT_INSTRUCTION)
                    + (action_contract.event_review_instruction()
                       if action_contract.review_projection.get('event_evidence_policy') == 'functional-integrity-v1' else
                      ' Return action_event_evidence in event_id order. effect_visible and target_reaction_visible '
                      'require a readable cause and actual response by the named participants; scope_valid forbids '
                      'unintended affected subjects. Evasion requires avoidance without injury; deflection requires '
                      'redirection, not a body hit. causal_order_valid checks visible causal progression. '
                      'time_window_valid checks the shot bounds, not suggested beat times. flow_continuity_valid '
                      'checks momentum handoff, not exact poses. When timestamp fields or active_intervals are '
                      'required, measure them from current samples, excluding preparation, waits, posing and camera-only '
                      'movement; never copy the planned schedule. Report all uninterrupted interaction intervals. ')
                    + 'creative_brief_valid and requested_outcome_valid check the explicit user requirements relevant '
                      'to this shot. identity_continuity_valid rejects identity, costume or prop-form drift. '
                    + boundary_evidence_instruction + blocking_instruction + composition_instruction + invariant_instruction)
                content[0]['text'] = ('Shot contract:\n' + json.dumps(contract, ensure_ascii=False)
                    + '\nReturn exactly the supplied schema. required_entities_visible order: ' + visibility_order
                    + '. identity_crop_boxes order: ' + crop_order
                    + '. Crops are normalized [x1,y1,x2,y2] from temporal midpoint, one clear individual only; '
                      'otherwise null. Match identity references to their named character. '
                      'end_state_context must not repeat dynamic action, pose or prop-state observations.')
            parsed = None
            response_path = self.cache_dir / 'responses' / video_hash / f'{uuid4().hex}.json'
            response_path.parent.mkdir(parents=True, exist_ok=True)
            received = []
            attempts = 1 if completion else _MAX_RESPONSE_ATTEMPTS
            if completion:
                # 持久化调用意图：中断、拒绝、歧义或结构错误都不能在恢复时偷偷重审。
                _atomic_write_json(unavailable_path, {
                    'video_hash': video_hash, 'contract_hash': contract_hash, 'evaluator_hash': evaluator_hash,
                    'status': 'unavailable', 'response': {}, 'response_path': str(response_path)})
            for attempt in range(1, attempts + 1):
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    response_format=response_format,
                    max_completion_tokens=completion_token_budget,
                    **structured_output_options(self.model),
                )
                received.append(completion_response_record(response, model=self.model))
                _atomic_write_json(response_path, {
                    'video_hash': video_hash, 'contract_hash': contract_hash,
                    'evaluator_hash': evaluator_hash, 'attempts': received,
                })
                raw, diagnostic, refused = _completion_payload(response)
                if refused:
                    raise SemanticReviewUnavailableError(
                        f"语义验收被服务端拒绝，未重试 ({diagnostic})"
                    )
                try:
                    candidate = load_unique_json(raw)
                except StructuredResponseError as exc:
                    # 冲突判断不能按 JSON 格式修复，否则后值可能覆盖否定证据。
                    raise SemanticReviewUnavailableError(
                        '语义验收响应包含重复字段；保留原片，未重试或提交重拍'
                    ) from exc
                except (TypeError, json.JSONDecodeError) as exc:
                    if attempt < attempts:
                        print(
                            "     ⚠ 语义验收响应异常，复核同一视频一次: "
                            f"{diagnostic}"
                        )
                        continue
                    raise SemanticReviewUnavailableError(
                        f"语义验收连续 {_MAX_RESPONSE_ATTEMPTS} 次未返回可解析 JSON "
                        f"({diagnostic}): {exc}"
                    ) from exc
                observed_candidate = candidate
                if completion:
                    try:
                        candidate = decode_observations(candidate, completion,
                            response_format['json_schema']['schema'], len(timestamps))
                    except (ValueError, TypeError, KeyError) as exc:
                        uncertain = isinstance(exc, ObservationUncertaintyError)
                        _atomic_write_json(self.cache_dir / f'{video_hash}.evidence.json', {
                            'video_hash': video_hash, 'contract_hash': contract_hash, 'evaluator_hash': evaluator_hash,
                            'sample_timestamps': timestamps, 'media_duration': duration,
                            'status': 'unavailable',
                            'failure_kind': 'semantic_uncertainty' if uncertain else 'observation_source_invalid',
                            **({'failure_reason': str(exc), 'unresolved_criteria': exc.keys} if uncertain else {}),
                            'observation_schema': response_format['json_schema']['schema'],
                            'response': observed_candidate})
                        if uncertain:
                            raise SemanticReviewUnavailableError(str(exc)) from exc
                        raise SemanticReviewUnavailableError(f'动作结果观察无效：{exc}；保留原片，不自动重试') from exc
                candidate = project_boundary_observations(candidate, applicable=boundary_review,
                    fields=('boundary_continuity_valid', *_BOUNDARY_REVIEW_FIELDS))
                gaps = unobservable_fields(candidate, response_format['json_schema']['schema'])
                if gaps:
                    _atomic_write_json(self.cache_dir / f'{video_hash}.evidence.json', {
                        'video_hash': video_hash, 'contract_hash': contract_hash, 'evaluator_hash': evaluator_hash,
                        'sample_timestamps': timestamps, 'media_duration': duration,
                        'status': 'unavailable', 'unobservable_fields': gaps,
                        'observation_scope': contract['observation_scope'], 'response': observed_candidate})
                    raise SemanticReviewUnavailableError('画面观察不足或看不清，无法确认验收；保留原片，不自动重拍')
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
                    invariant_review=invariant_review,
                    boundary_review=boundary_review,
                    state_handoff_review=state_handoff_review,
                )
                if shape_issues:
                    detail = "；".join(shape_issues)
                    if attempt < attempts:
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
        if completion and (parsed.get('accepted') or (self.review_mode == 'normal'
                and parsed.get('primary_action_completed'))):
            try:
                visual_args = ({'visual_evidence': combine(previous_history, sample_clip(video_path, shot,
                    scope='current_clip', timestamps=timestamps))} if temporal_review else {})
                checked, support_evidence = verify_support(completion, observed_candidate,
                    client=client, model=self.model, cache_dir=self.cache_dir, video_hash=video_hash, **visual_args)
                parsed = decode_observations(checked, completion,
                    response_format['json_schema']['schema'], len(timestamps))
                parsed = project_boundary_observations(parsed, applicable=boundary_review,
                    fields=('boundary_continuity_valid', *_BOUNDARY_REVIEW_FIELDS))
            except SupportJudgmentError as exc:
                _atomic_write_json(self.cache_dir / f'{video_hash}.evidence.json', {
                    'video_hash': video_hash, 'contract_hash': contract_hash, 'evaluator_hash': evaluator_hash,
                    'sample_timestamps': timestamps, 'media_duration': duration,
                    'status': 'unavailable', 'failure_kind': exc.code, 'failure_reason': str(exc),
                    **({'composition_measurement': composition_measurement,
                        'independent_findings': {'composition_change_valid': composition_verdict(
                            observed_candidate.get('composition_change_valid'), composition_measurement)}}
                       if geometry_review else {}),
                    'unresolved_criteria': exc.keys,
                    **({'unresolved_evidence':exc.details} if exc.details else {}), 'response': observed_candidate})
                raise SemanticReviewUnavailableError(str(exc)) from exc
            except (ValueError, TypeError, KeyError) as exc:
                raise SemanticReviewUnavailableError(
                    f'观察事实与要求的核对未完成：{exc}；保留原片，不自动重试') from exc
        causal_evidence = None
        event_evidence = parsed.get('action_event_evidence') if action_events else None
        if action_events:
            try:
                _apply_event_evidence_verdict(parsed, action_contract, event_evidence, timestamps=timestamps,
                    media_duration=duration, defer_timing_uncertainty=self.review_mode == 'normal')
            except SemanticReviewUnavailableError as exc:
                # Evidence failure is recoverable review work, never a rejected take/paid retake.
                _atomic_write_json(self.cache_dir / f'{video_hash}.evidence.json', {
                    'video_hash': video_hash, 'contract_hash': contract_hash, 'evaluator_hash': evaluator_hash,
                    'sample_timestamps': timestamps, 'action_event_evidence': event_evidence,
                    'media_duration': duration,
                    'event_assessment': parsed.get('event_assessment'), 'failure_reason': str(exc),
                    **({'failure_kind': parsed['event_assessment']['failure_kind']}
                       if parsed.get('event_assessment', {}).get('failure_kind') else {}),
                    'status': 'unavailable', 'observation_scope': contract['observation_scope'],
                    'response': parsed})
                raise
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
        if geometry_review:
            parsed['composition_change_valid'] = composition_verdict(
                parsed.get('composition_change_valid'), composition_measurement)
            if composition_measurement['status'] == 'unchanged':
                parsed['failure_reason'] = '；'.join(filter(None, [parsed.get('failure_reason'),
                    '镜头几何证据显示取景未变，不能以人物或道具移动代替计划切镜']))
        elif composition_review and composition_similarity is not None:
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
                require_visual_invariants=invariant_review,
                require_timing=bool(action_contract.review_projection.get('action_timing')),
                require_outcome=bool(action_contract.review_projection.get('required_final_outcome')),
                require_sequence=bool(action_events),
                require_creative_brief=bool(action_contract.review_projection.get('performance_policy')),
                require_observability=True,
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
            **({'composition_measurement': composition_measurement} if geometry_review else {}),
            "causal_sample_evidence": causal_evidence,
            "action_event_evidence": event_evidence,
            "sample_timestamps": timestamps,
            "media_duration": duration,
            "timing_assessment": parsed.get('timing_assessment'),
            "event_assessment": parsed.get('event_assessment'),
            "action_contract": asdict(action_contract) if outcome_review else None,
            "boundary_state_evidence": boundary_state_evidence,
            "observation_scope": contract['observation_scope'],
            **({'observation_response': observed_candidate,
                'support_evidence': support_evidence,
                'observation_schema': response_format['json_schema']['schema']} if completion else {}),
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
    if (contract.get('completion_contract') or {}).get('intent_policy'):
        # 作者调度不再成为另一套必要性目录。物理指向由事件主体/目标/范围承担。
        contract['blocking'] = {e['actor']: {'action_target': e['target']}
                                for e in contract.get('action_events', [])}
        # 摄影检查消费与生成共用的摄影约束，不重新解释构图文案中的人物表演。
        # 完整构图/位置描述仍留在生成指导；明确指定的表演由目标目录承担。
        from pipeline.camera_contract import compile_camera
        _, camera = compile_camera(shot)
        contract['camera'] = {key: camera[source] for key, source in (
            ('primary_movement', 'camera_movement'), ('speed', 'camera_speed'),
            ('axis_change', 'camera_axis')) if source in camera}
        contract['camera'].update({f'{key}_framing': value
                                  for key, value in camera.get('camera_framing', {}).items()})
    if (contract.get('performance_policy') or {}).get('version') == 'performance-v2-outcomes':
        # 自由编舞的计划时间不是观测依据；只给验收器实际采样时间和时长要求。
        # 冻结计划与精确动作复现仍保留原时间表，不在这里修改共享合同。
        contract.pop('action_timing', None)
        for event in contract['action_events']:
            for key in ('start_seconds', 'end_seconds', 'effect_at_seconds', 'reaction_at_seconds'):
                event.pop(key, None)
    visual_invariants = compile_visual_invariants(shot)
    if visual_invariants.required:
        contract["visual_invariants"] = visual_invariants.review_projection
    contract["narrative_carriers"] = list(compile_narrative_carriers(contract))
    if contract.get('completion_contract'):
        from pipeline.assessment_scope import local_assessment, INSTRUCTION
        contract['completion_contract'] = local_assessment(contract['completion_contract'])
        contract['completion_contract']['assessment_instruction'] = INSTRUCTION
    return contract


def _apply_event_evidence_verdict(value: dict, contract: CompiledActionContract, evidence: object,
                                  *, timestamps=None, media_duration=None,
                                  defer_timing_uncertainty: bool = False) -> None:
    report = assess_event_evidence(contract, evidence, timestamps=timestamps, media_duration=media_duration)
    value['event_assessment'] = report
    from pipeline.review_policy import timing_is_quality_only
    if report['status'] == 'unavailable' and not (defer_timing_uncertainty and timing_is_quality_only(report)):
        detail = '；'.join([report['reason'], *report['content_issues']])
        if report.get('failure_kind') == 'semantic_uncertainty':
            raise SemanticReviewUnavailableError(f'时间观察语义不确定：{detail}；保留原片，不自动重拍')
        raise SemanticReviewUnavailableError(f'时间证据不可用：{detail}；保留原片，不自动重拍')
    if (contract.review_projection.get('performance_policy') or {}).get('version') == 'performance-v2-outcomes':
        value['timing_assessment'] = report['timing']
        value['action_timing_valid'] = report['timing']['status'] in ('passed', 'not_required')
    issues = report['issues']
    value['action_sequence_valid'] = not issues
    if issues:
        value['accepted'] = False
        value['failure_reason'] = '；'.join(filter(None, [value.get('failure_reason', ''), *issues]))


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
    invariant_review: bool,
    boundary_review: bool,
    state_handoff_review: bool,
) -> list[str]:
    if not isinstance(value, dict):
        return ["响应必须是 JSON 对象"]

    required_booleans = [
        "accepted",
        "action_geometry_valid",
        "primary_action_completed",
        "identity_continuity_valid",
    ]
    if action_contract.review_projection.get('action_timing'):
        required_booleans.append('action_timing_valid')
    if action_contract.review_projection.get('required_final_outcome'):
        required_booleans.append('requested_outcome_valid')
    if action_contract.review_projection.get('performance_policy'):
        required_booleans.append('creative_brief_valid')
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
    if invariant_review:
        required_booleans.append("visual_invariants_valid")
    if boundary_review:
        required_booleans.append('boundary_continuity_valid')
        required_booleans.extend(_BOUNDARY_REVIEW_FIELDS)
    issues = [
        f"{field} 必须是布尔值"
        for field in required_booleans
        if not isinstance(value.get(field), bool)
    ]
    if action_contract.review_projection.get('action_events'):
        issues.extend(action_contract.event_evidence_shape_issues(value.get('action_event_evidence')))

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
    invariant_review: bool,
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
    if action_contract.review_projection.get('action_timing'):
        properties['action_timing_valid'] = {'type': 'boolean'}
    if action_contract.review_projection.get('required_final_outcome'):
        properties['requested_outcome_valid'] = {'type': 'boolean'}
    if action_contract.review_projection.get('performance_policy'):
        properties['creative_brief_valid'] = {'type': 'boolean'}
    if action_contract.review_projection.get('action_events'):
        properties['action_event_evidence'] = action_contract.event_evidence_schema()
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
    if invariant_review:
        properties["visual_invariants_valid"] = {"type": "boolean"}
    if boundary_review:
        properties['boundary_continuity_valid'] = {'type': 'boolean'}
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
    if action_contract.review_projection.get('completion_contract'):
        from pipeline.result_observations import observation_properties
        properties = observation_properties(properties, action_contract.review_projection['completion_contract'], sample_count)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_take_review",
            "strict": True,
            "schema": nullable_observation_schema({
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            }),
        },
    }


def _completion_payload(response: Any) -> tuple[str, str, bool]:
    choices = getattr(response, "choices", None) or []
    if len(choices) != 1:
        raise SemanticReviewUnavailableError('语义验收必须返回一个完整判断；保留原片，未重试')
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    raw = content if isinstance(content, str) else ""
    finish = getattr(choice, "finish_reason", None)
    finish_reason = finish if isinstance(finish, str) and finish in (
        'stop', 'length', 'tool_calls', 'function_call', 'content_filter', 'safety',
    ) else 'unknown'
    refusal = getattr(message, "refusal", None)
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if type(completion_tokens) is not int or completion_tokens < 0:
        completion_tokens = None
    diagnostic = (
        f"finish_reason={finish_reason}, content_length={len(raw)}, "
        f"completion_tokens={completion_tokens if completion_tokens is not None else 'unknown'}"
    )
    refused = bool(refusal) or finish_reason in {"content_filter", "safety"}
    if not refused and finish_reason != 'stop':
        raise SemanticReviewUnavailableError(
            f'语义验收响应未完整结束；保留原片，未重试 ({diagnostic})'
        )
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


def _read_cache(path: Path, *, payload_field: str = 'review') -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and isinstance(value.get(payload_field), dict) else None


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
