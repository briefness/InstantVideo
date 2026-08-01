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
from tools.ffmpeg_ops import get_video_duration
from tools.frame_extractor import extract_frame


_REVIEWER_VERSION = "continuity-v4"
_SAMPLING_POLICY = "five-point-v1"
_MAX_COMPLETION_TOKENS = 4096
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
    return "" if value is None else str(value).strip()


@dataclass(frozen=True)
class SemanticReview:
    accepted: bool
    required_entities_visible: dict[str, bool]
    action_geometry_valid: bool
    primary_action_completed: bool
    observed_end_state: dict[str, str]
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
    def from_dict(cls, value: dict[str, Any]) -> "SemanticReview":
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
            and review.primary_action_completed
            and review.boundary_continuity_valid
            and review.identity_continuity_valid
            and review.environment_continuity_valid
            and review.action_handoff_valid
            and review.screen_direction_valid
            and review.prop_continuity_valid
        )
        reason = review.failure_reason
        if not objectively_valid and not reason:
            reason = _objective_failure_reason(review)
        elif review.accepted != objectively_valid and not reason:
            reason = "语义验收字段互相矛盾"
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
            "reviewer_version": _REVIEWER_VERSION,
            "sampling_policy": _SAMPLING_POLICY,
            "image_detail": config.SEMANTIC_REVIEW_IMAGE_DETAIL,
            "reference_hashes": [
                _sha256_file(Path(path)) for path in reference_paths
            ],
            "identity_crop_entities": crop_entities,
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
                return SemanticReview.from_dict(cached["review"])
            except ValueError:
                pass

        client = self.client or OpenAI(
            api_key=config.ARK_API_KEY,
            base_url=config.ARK_BASE_URL,
        )
        duration = get_video_duration(video_path)
        timestamps = _sample_timestamps(duration)
        required_entities = [
            str(entity) for entity in contract.get("required_visible_entities") or []
        ]
        visibility_order = json.dumps(required_entities, ensure_ascii=False)
        crop_order = json.dumps(crop_entities, ensure_ascii=False)
        with tempfile.TemporaryDirectory(prefix="review-", dir=self.cache_dir) as frame_dir:
            current_images = []
            for index, timestamp in enumerate(timestamps, start=1):
                frame_path = Path(frame_dir) / f"frame_{index}.jpg"
                extract_frame(video_path, str(frame_path), timestamp=timestamp)
                current_images.append((timestamp, _image_content(frame_path)))

            content: list[dict[str, Any]] = [{
                "type": "text",
                "text": (
                    "Shot contract:\n"
                    + json.dumps(contract, ensure_ascii=False)
                    + "\nReturn strict JSON booleans for accepted, "
                    "action_geometry_valid, "
                    "primary_action_completed, boundary_continuity_valid, and "
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
                        "required entity is absent or unreadable, actor and target do not "
                        "share a readable line of action, attack direction is wrong, or the "
                        "primary action is not completed. Camera polish cannot compensate "
                        "for failed action geography. For a same-scene boundary, compare the "
                        "accepted previous tail only with the first current frame: preserve "
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
                    response_format={"type": "json_object"},
                    max_completion_tokens=_MAX_COMPLETION_TOKENS,
                )
                raw, diagnostic, refused = _completion_payload(response)
                if refused:
                    raise SemanticReviewUnavailableError(
                        f"语义验收被服务端拒绝，未重试 ({diagnostic})"
                    )
                try:
                    parsed = json.loads(raw)
                    break
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
        assert parsed is not None
        if not isinstance(parsed, dict):
            raise SemanticReviewUnavailableError("语义验收响应必须是 JSON 对象")
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
            review = SemanticReview.from_dict(parsed)
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
            "review": asdict(review),
        })
        return review


def _review_contract(shot: dict) -> dict[str, Any]:
    return {
        key: shot.get(key)
        for key in (
            "shot_id", "primary_action", "coverage_role",
            "required_visible_entities", "interaction_geometry", "action_beats",
            "camera", "blocking", "start_state", "end_state", "characters",
            "scene_id", "continuity_from_previous", "composition_change",
            "extract_character_ref",
        )
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


def _sample_timestamps(duration: float) -> tuple[float, ...]:
    end = max(0.1, duration - 0.1)
    return tuple(dict.fromkeys(round(value, 3) for value in (
        0.1,
        max(0.1, duration * 0.25),
        max(0.1, duration * 0.5),
        max(0.1, duration * 0.75),
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
