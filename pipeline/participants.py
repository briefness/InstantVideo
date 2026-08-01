"""Canonical visible-participant rules shared across pipeline stages."""

from __future__ import annotations

from collections.abc import Iterable


_BLOCKING_TARGET_FIELDS = (
    "facing_target",
    "eyeline_target",
    "action_target",
)


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
