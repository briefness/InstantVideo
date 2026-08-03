"""Canonical visible-participant rules shared across pipeline stages."""

from __future__ import annotations

from collections.abc import Iterable


_BLOCKING_TARGET_FIELDS = (
    "facing_target",
    "eyeline_target",
    "action_target",
)


def canonical_participant_id(
    value: object,
    candidates: Iterable[str],
    *,
    exclude: Iterable[str] = (),
    fallback_to_single: bool = False,
) -> str:
    """Resolve descriptive LLM labels only when one stable entity ID is unambiguous."""
    raw = str(value or "").strip()
    if not raw or raw.casefold() == "none":
        return raw
    excluded = {str(name).strip() for name in exclude if str(name).strip()}
    catalog = list(dict.fromkeys(
        str(name).strip()
        for name in candidates
        if str(name).strip() and str(name).strip() not in excluded
    ))
    if raw in catalog:
        return raw

    key = _participant_key(raw)
    exact = [name for name in catalog if _participant_key(name) == key]
    if len(exact) == 1:
        return exact[0]
    contained = [
        name for name in catalog
        if _participant_key(name) in key or key in _participant_key(name)
    ]
    if len(contained) == 1:
        return contained[0]
    if fallback_to_single and len(catalog) == 1:
        return catalog[0]
    return raw


def _participant_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def visible_character_names(
    shot: dict,
    character_names: Iterable[str],
) -> list[str]:
    """Return declared and structurally visible character IDs in stable order."""
    catalog = [str(name).strip() for name in character_names if str(name).strip()]
    known = set(catalog)
    declared = [
        str(name).strip()
        for name in shot.get("characters", [])
        if str(name).strip()
    ]
    observed = set(declared)

    camera = shot.get("camera", {})
    positions = camera.get("screen_positions", {}) if isinstance(camera, dict) else {}
    if isinstance(positions, dict):
        observed.update(str(name).strip() for name in positions if str(name).strip() in known)

    blocking = shot.get("blocking", {})
    if isinstance(blocking, dict):
        observed.update(str(name).strip() for name in blocking if str(name).strip() in known)
        for intent in blocking.values():
            if not isinstance(intent, dict):
                continue
            observed.update(
                str(intent.get(field, "")).strip()
                for field in _BLOCKING_TARGET_FIELDS
                if str(intent.get(field, "")).strip() in known
            )

    for beat in shot.get("action_beats", []):
        if not isinstance(beat, dict):
            continue
        observed.update(
            str(beat.get(field, "")).strip()
            for field in ("actor", "target")
            if str(beat.get(field, "")).strip() in known
        )

    ordered: list[str] = []
    for name in [*declared, *catalog]:
        if name in observed and name not in ordered:
            ordered.append(name)
    return ordered
