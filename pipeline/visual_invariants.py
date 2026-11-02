"""Compile bounded visual invariants from existing structured shot fields."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from pipeline.causality import EFFECT_PHASES, interaction_geometry
from pipeline.camera_contract import compile_camera


VISUAL_INVARIANTS_VERSION = "visual-invariants-v3-subject-motion"
_POSITION_TERMS = (
    "left", "center", "right", "foreground", "midground", "background",
)


@dataclass(frozen=True)
class CompiledVisualInvariants:
    """One bounded projection shared by prompting, review, and provenance."""

    prompt_parts: tuple[str, ...]
    review_projection: dict[str, object]

    @property
    def required(self) -> bool:
        return bool(self.review_projection)


def compile_visual_invariants(shot: dict) -> CompiledVisualInvariants:
    """Derive locks without consuming free-form action or narrative prose."""
    phase = str(interaction_geometry(shot).get("effect_phase", "")).strip()
    if phase not in EFFECT_PHASES:
        return CompiledVisualInvariants((), {})

    setting = _label(shot.get("scene_id"))
    lighting = _text(shot.get("lighting"))
    required_entities = _labels(shot.get("required_visible_entities"))
    key_props = _labels(shot.get("key_props"))
    continuity_props = _labels(shot.get("continuity_props"))
    screen_positions = safe_screen_positions(
        (shot.get("camera") or {}).get("screen_positions")
        if isinstance(shot.get("camera"), dict)
        else None
    )

    prompt_parts: list[str] = []
    projection: dict[str, object] = {"version": VISUAL_INVARIANTS_VERSION}
    if setting:
        projection["setting_identity"] = setting
        prompt_parts.append(
            f"positive lock - preserve setting identity {setting}; "
            "do not replace or invent a different environment"
        )
    if lighting:
        projection["lighting"] = lighting
        prompt_parts.append(
            "positive lock - preserve the declared lighting treatment without "
            f"replacing it with an unrelated setup: {lighting}"
        )
    if required_entities:
        projection["required_entity_identity_and_relative_scale"] = list(
            required_entities
        )
        prompt_parts.append(
            "positive lock - preserve identity and relative scale for these declared "
            "entities; do not duplicate a declared subject identity: "
            + ", ".join(required_entities)
        )
    if key_props:
        projection["key_prop_identity_and_form"] = list(key_props)
        prompt_parts.append(
            "positive lock - preserve the identity and form of these props unless the "
            "Action Contract visibly changes them: "
            + ", ".join(key_props)
        )
    if continuity_props:
        projection["persistent_props"] = list(continuity_props)
        prompt_parts.append(
            "positive lock - these continuity props must not disappear, duplicate, "
            "or change form unless the Action Contract requires it: "
            + ", ".join(continuity_props)
        )
    if screen_positions:
        projection["initial_screen_positions"] = dict(screen_positions)
        prompt_parts.append(
            "positive lock - begin with these screen positions: "
            + ", ".join(
                f"{entity}={position}"
                for entity, position in screen_positions
            )
        )

    camera_parts, camera_projection = compile_camera(shot)
    orientations = safe_body_orientations(shot.get("blocking"))
    if orientations:
        projection["body_orientations"] = orientations
        projection["body_orientation_scope"] = "initial; subject pose and camera-relative view may evolve with contracted actions"
        prompt_parts.append(
            "positive lock - begin with these body orientations; allow subject turns and camera-relative changes required by the Action Contract, even with a fixed camera: "
            + ", ".join(f"{entity}={orientation}" for entity, orientation in orientations.items())
        )
    prompt_parts.extend(camera_parts)
    projection.update(camera_projection)

    if len(projection) == 1:
        return CompiledVisualInvariants((), {})
    return CompiledVisualInvariants(tuple(prompt_parts), projection)


def safe_body_orientations(value: object) -> dict[str, str]:
    """只传递明确、可枚举的朝向；不把自由动作描述带入安全重试。"""
    if not isinstance(value, dict):
        return {}
    result = {}
    for entity, blocking in value.items():
        if not isinstance(entity, str) or not entity.strip() or not isinstance(blocking, dict):
            continue
        raw = blocking.get("body_orientation")
        if not isinstance(raw, str):
            continue
        text = " ".join(raw.casefold().split())
        profile = re.fullmatch(r"(?:facing|profile toward) screen-(left|right)", text)
        if profile:
            result[entity] = f"profile toward screen-{profile[1]}"
        elif text in {"facing camera", "toward camera", "front toward camera"}:
            result[entity] = "facing camera"
        elif text in {"away from camera", "back to camera"}:
            result[entity] = "back to camera"
        elif re.fullmatch(
            r"three-quarter toward (?:screen-(?:left|right) and )?(?:foreground|background), "
            r"(?:toward|away from) camera", text
        ):
            result[entity] = text
    return result


def visual_invariants_fingerprint(shot: dict) -> str | None:
    projection = compile_visual_invariants(shot).review_projection
    if not projection:
        return None
    payload = json.dumps(
        projection, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def safe_screen_positions(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        return ()
    positions: list[tuple[str, str]] = []
    for entity, raw_position in value.items():
        text = str(raw_position or "").casefold()
        tokens = [
            term for term in _POSITION_TERMS
            if re.search(rf"\b{term}\b", text)
        ]
        label = _label(entity)
        if label and tokens:
            positions.append((label, " ".join(tokens)))
    return tuple(positions)


def _labels(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(label for item in value if (label := _label(item))))


def _label(value: object) -> str:
    return " ".join(
        str(value or "").strip().replace("_", " ").replace("-", " ").split()
    )


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())
