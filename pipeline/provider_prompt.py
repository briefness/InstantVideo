"""Provider-facing prompt profiles compiled from the Production Contract."""

from __future__ import annotations

import re

from pipeline.causality import compile_action_contract, interaction_geometry
from pipeline.production_plan import classify_framing


_FRAMING_LABELS = {
    "wide": "wide shot",
    "medium": "medium shot",
    "close_detail": "close detail shot",
}
_POSITION_TERMS = (
    "left", "center", "right", "foreground", "midground", "background",
)


def has_explicit_action_contract(shot: dict) -> bool:
    """Whether this shot opts into phase-owned action compilation."""
    geometry = interaction_geometry(shot)
    return str(geometry.get("effect_phase", "")).strip() in {
        "none", "setup", "active", "aftermath",
    }


def compile_normal_provider_prompt(
    shot: dict,
    storyboard: dict,
    *,
    has_observed_start: bool,
) -> str:
    """Compile a normal provider prompt without re-consuming action prose.

    Explicit Action Contract shots intentionally do not inherit ``prompt_en``,
    ``scene_description``, or ``negative_prompt``.  Those fields predate the
    contract and can otherwise smuggle an outcome into a setup or aftermath.
    Legacy shots keep their old caller path for backward compatibility.
    """
    if not has_explicit_action_contract(shot):
        return ""

    contract = compile_action_contract(shot)
    parts = ["Create one coherent cinematic shot"]
    parts.extend(_normal_visual_parts(shot, storyboard))
    if has_observed_start:
        parts.append(
            "begin from the observed physical and action state in Image 1 while applying the current camera composition"
        )
    elif contract.prompt_start_state:
        parts.append(f"start exactly with {contract.prompt_start_state}")
    parts.extend(contract.prompt_parts)
    parts.extend(_normal_composition_parts(shot, has_observed_start=has_observed_start))
    parts.append("do not introduce another major action")
    return ". ".join(part for part in parts if part).strip() + "."


def compile_policy_safe_prompt(
    shot: dict,
    *,
    storyboard: dict | None = None,
    has_state_reference: bool,
    image_role: str | None,
    reference_count: int,
    retake_instruction: str | None = None,
) -> str:
    """Compile a non-graphic provider retry without reusing rejected prose.

    The normal profile is intentionally not an input.  A moderation retry must
    be independently compiled from bounded Production Contract fields, or the
    rejected prompt will leak back through action, narrative, character, and
    negative-prompt injectors.
    """
    action_contract = compile_action_contract(shot)
    geometry = action_contract.canonical_geometry
    actor = _entity_label(geometry.get("actor"))
    target = _entity_label(geometry.get("target"))
    parts = [
        "Create one coherent cinematic shot with a restrained, non-graphic fictional treatment",
    ]
    parts.extend(_normal_visual_parts(shot, storyboard or {}))
    if retake_instruction:
        parts.append(retake_instruction)
    camera = shot.get("camera", {})
    camera = camera if isinstance(camera, dict) else {}
    framing = _FRAMING_LABELS.get(classify_framing(camera.get("start_framing")))
    if framing:
        parts.append(f"camera: stable {framing}")
    positions = _safe_screen_positions(camera.get("screen_positions"))
    if positions:
        parts.append("screen positions: " + ", ".join(positions))

    parts.append(_policy_action_clause(action_contract, actor, target))
    parts.extend(_reference_clauses(
        has_state_reference=has_state_reference,
        image_role=image_role,
        reference_count=reference_count,
    ))
    if action_contract.phase == "setup":
        parts.append(
            "show only a readable preparation endpoint and independent target motion"
        )
    elif action_contract.phase == "aftermath":
        parts.append("show only the already established state with no new action")
    else:
        parts.append(
            "show readable cause and response with abstract, broad-audience visual staging"
        )
    parts.append("stable subjects, coherent motion, no text artifacts, no added major action")
    return ". ".join(part for part in parts if part).strip() + "."


def _policy_action_clause(contract: object, actor: str, target: str) -> str:
    """Use canonical Contract values instead of reinterpreting raw geometry."""
    phase = str(getattr(contract, "phase", "none")).strip()
    mode = str(getattr(contract, "mode", "none")).strip()
    scope = str(getattr(contract, "outcome_scope", "none")).strip()
    motion = str(getattr(contract, "effect_motion", "none")).strip()
    geometry = getattr(contract, "canonical_geometry", {})
    geometry = geometry if isinstance(geometry, dict) else {}
    visible_result = str(
        getattr(contract, "contracted_visible_result", "") or ""
    ).strip()
    unaffected_behavior = str(geometry.get("unaffected_behavior", "") or "").strip()
    source = actor or "the primary subject"
    receiver = target or "the intended subject"

    if phase == "setup":
        return (
            f"setup phase: {source} visibly prepares toward {receiver}; "
            "show intent only, with no contact or target-state change"
        )
    if phase == "aftermath":
        return (
            f"aftermath phase: show the already established non-graphic state of "
            f"{receiver}; do not introduce a new physical action"
        )
    if phase != "active":
        return f"show one readable non-graphic story beat centered on {source}"

    scope_label = {
        "single": "one clearly isolated intended target within the visible effect region",
        "subset": "only the intended subset",
        "all": "the full intended group",
    }.get(scope, "only the intended scope")
    if mode == "direct_contact":
        action = "one controlled contact action with actor and target sharing the frame"
    elif mode == "directed_path":
        action = "one directed action with a clearly readable path from source to target"
    elif mode == "area_effect":
        action = "one contained area action with a clearly readable affected region"
    elif mode == "indirect_effect":
        action = "one indirect action with a clearly readable intermediary and result"
    else:
        action = "one clearly readable action"
    result_clause = (
        f"contracted visible result: {visible_result}"
        if visible_result else "show only the contract-defined visible endpoint"
    )
    boundary_clause = (
        f"subjects outside that scope keep this declared unaffected behavior: {unaffected_behavior}"
        if unaffected_behavior else "subjects outside that scope preserve their prior independent behavior"
    )
    return (
        f"active {mode or 'action'} phase, effect motion {motion}: {source} performs {action} toward {receiver}; "
        f"{scope_label}; {result_clause}; {boundary_clause}"
    )


def _normal_visual_parts(shot: dict, storyboard: dict) -> list[str]:
    """Render bounded visual metadata; never free-form action/narrative prose."""
    parts: list[str] = []
    for key, label in (("style", "visual style"), ("mood", "mood")):
        value = _structured_text(storyboard.get(key))
        if value:
            parts.append(f"{label}: {value}")

    scene_id = _entity_label(shot.get("scene_id"))
    if scene_id:
        parts.append(f"setting: {scene_id}")
    lighting = _structured_text(shot.get("lighting"))
    if lighting:
        parts.append(f"lighting: {lighting}")

    visible = [
        _entity_label(entity)
        for entity in shot.get("required_visible_entities", [])
        if _entity_label(entity)
    ]
    if visible:
        parts.append("keep these subjects clearly visible: " + ", ".join(visible))
    props = [_entity_label(prop) for prop in shot.get("key_props", []) if _entity_label(prop)]
    if props:
        parts.append("visible props: " + ", ".join(props))
    parts.extend(_normal_identity_parts(shot, storyboard))

    return parts


def _normal_composition_parts(shot: dict, *, has_observed_start: bool) -> list[str]:
    camera = shot.get("camera")
    camera = camera if isinstance(camera, dict) else {}
    parts: list[str] = []
    framing = _FRAMING_LABELS.get(classify_framing(camera.get("start_framing")))
    if framing:
        parts.append(f"camera: {framing}")
    positions = _safe_screen_positions(camera.get("screen_positions"))
    if positions:
        parts.append("screen positions: " + ", ".join(positions))
    composition_change = str(shot.get("composition_change", "")).strip()
    if composition_change in {"medium", "large"} and has_observed_start:
        change = (
            "a clearly different shot size or angle"
            if composition_change == "medium"
            else "unmistakably different coverage"
        )
        parts.append(f"use {change} from the supplied previous tail")
    return parts


def _normal_identity_parts(shot: dict, storyboard: dict) -> list[str]:
    """Project declared character identity only, never the shot's free-form prose."""
    requested = {
        str(name).strip()
        for name in [*shot.get("characters", []), *shot.get("required_visible_entities", [])]
        if str(name).strip()
    }
    characters = storyboard.get("characters")
    if not isinstance(characters, list):
        return []
    identities = []
    for character in characters:
        if not isinstance(character, dict):
            continue
        name = str(character.get("name", "")).strip()
        description = _structured_text(character.get("description"))
        mobility = _structured_text(character.get("mobility"))
        if not name or name not in requested:
            continue
        details = [description] if description else []
        if mobility and mobility != "unspecified":
            details.append(f"mobility {mobility}")
        if details:
            identities.append(f"declared appearance for {_entity_label(name)}: {', '.join(details)}")
    return identities


def _structured_text(value: object) -> str:
    """Normalize scalar visual metadata without claiming semantic sanitization."""
    return " ".join(str(value or "").strip().split())


def _reference_clauses(
    *,
    has_state_reference: bool,
    image_role: str | None,
    reference_count: int,
) -> list[str]:
    clauses: list[str] = []
    if has_state_reference:
        clauses.append(
            "Image 1 controls only the accepted prior scene state, subject placement, and pose"
        )
        if reference_count > 1:
            clauses.append(
                "remaining images control identity and appearance only, not pose or framing"
            )
    elif image_role == "reference_image" and reference_count:
        clauses.append(
            "reference images control identity and appearance only, not pose, background, or framing"
        )
    elif image_role == "first_frame" and reference_count:
        clauses.append("begin from the supplied first frame without replaying a completed action")
    return clauses


def _safe_screen_positions(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    positions: list[str] = []
    for entity, raw_position in value.items():
        text = str(raw_position or "").casefold()
        tokens = [term for term in _POSITION_TERMS if re.search(rf"\b{term}\b", text)]
        if tokens:
            positions.append(f"{_entity_label(entity)}={' '.join(tokens)}")
    return positions


def _entity_label(value: object) -> str:
    return " ".join(
        str(value or "").strip().replace("_", " ").replace("-", " ").split()
    )
