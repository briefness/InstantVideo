"""Canonical visible-participant rules shared across pipeline stages."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re
from typing import Literal
from pipeline.action_vocabulary import INDEPENDENT_ACTIVITIES


_BLOCKING_TARGET_FIELDS = (
    "facing_target",
    "eyeline_target",
    "action_target",
)
_GENERIC_ENTITY_TOKENS = frozenset({
    "group", "member", "unit", "final", "last", "remaining", "lead",
    "front", "target", "subject", "participant",
})


@dataclass(frozen=True)
class EntityRegistry:
    """Canonical structured entities available to one storyboard shot."""

    characters: tuple[str, ...]
    props: tuple[str, ...]
    themes: tuple[str, ...]

    @property
    def target_ids(self) -> tuple[str, ...]:
        return (*self.characters, *self.props, *self.themes)


@dataclass(frozen=True)
class StructuredEntityReference:
    """One registered-entity reference exposed by a shot contract."""

    field: str
    value: object
    kind: Literal["character", "target"]


def canonical_participant_id(
    value: object,
    candidates: Iterable[str],
    *,
    exclude: Iterable[str] = (),
    fallback_to_single: bool = False,
) -> str:
    """Resolve descriptive LLM labels only when one stable entity ID is unambiguous."""
    return _resolve_entity_id(value, candidates, exclude=exclude, fallback_to_single=fallback_to_single)


def _resolve_entity_id(
    value: object,
    candidates: Iterable[str],
    *,
    exclude: Iterable[str] = (),
    lexical_candidates: Iterable[str] = (),
    fallback_to_single: bool = False,
) -> str:
    """一次解析决定优先级；精确身份、排除与歧义都不能降级成更弱的猜测。"""
    raw = str(value or "").strip()
    if not raw or raw.casefold() == "none":
        return raw
    catalog = _entity_ids(candidates)
    excluded = set(_entity_ids(exclude))
    # exclude约束别名候选，不能把已经明确指向的ID改成另一身份。
    if raw in catalog:
        return raw
    key = _participant_key(raw)
    if not key:
        return raw
    exact = [name for name in catalog if _participant_key(name) == key]
    if exact:
        return exact[0] if len(exact) == 1 and exact[0] not in excluded else raw
    allowed = [name for name in catalog if name not in excluded]
    contained = [
        name for name in allowed
        if _participant_key(name) and (_participant_key(name) in key or key in _participant_key(name))
    ]
    if contained:
        return contained[0] if len(contained) == 1 else raw
    lexical = set(lexical_candidates)
    tokens = _entity_tokens(raw)
    matches = [name for name in allowed if name in lexical and tokens.intersection(_entity_tokens(name))]
    if matches:
        return matches[0] if len(matches) == 1 else raw
    if fallback_to_single and len(allowed) == 1:
        return allowed[0]
    return raw


def _participant_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def canonical_entity_id(
    value: object,
    candidates: Iterable[str],
    *,
    exclude: Iterable[str] = (),
) -> str:
    """Resolve an entity only with exact, contained, or unique lexical evidence."""
    catalog = _entity_ids(candidates)
    return _resolve_entity_id(value, catalog, exclude=exclude, lexical_candidates=catalog)


def canonical_target_id(
    value: object,
    registry: EntityRegistry,
    *,
    exclude: Iterable[str] = (),
) -> str:
    """Resolve targets across structured entities, then use lexical evidence for characters only."""
    return _resolve_entity_id(value, registry.target_ids, exclude=exclude, lexical_candidates=registry.characters)


def _entity_tokens(value: object) -> set[str]:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(value))
    tokens = set()
    for token in re.findall(r"[A-Za-z0-9]+", spaced):
        token = token.casefold()
        variants = {token}
        if token.endswith("ies") and len(token) > 4:
            variants.add(token[:-3] + "y")
            variants.add(token[:-1])
        elif token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            variants.add(token[:-1])
        tokens.update(
            variant for variant in variants
            if variant not in _GENERIC_ENTITY_TOKENS
        )
    return tokens


def shot_entity_registry(
    shot: dict,
    character_names: Iterable[str],
    theme_elements: Iterable[str] = (),
) -> EntityRegistry:
    """Return the canonical character, prop, and theme IDs available to a shot."""
    characters = _entity_ids(character_names)
    themes = _entity_ids(theme_elements)
    props = _canonical_prop_ids(shot, themes)
    return EntityRegistry(tuple(characters), tuple(props), tuple(themes))


def _entity_ids(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(
        str(value).strip() for value in values if str(value).strip()
    ))


def _canonical_prop_ids(shot: dict, themes: Iterable[str]) -> list[str]:
    declared = _declared_prop_ids(shot)
    if declared is not None:
        return declared
    canonical: list[str] = []
    theme_ids = _entity_ids(themes)
    for field in ("key_props", "continuity_props"):
        values = shot.get(field, [])
        if not isinstance(values, list):
            continue
        for value in values:
            raw = str(value).strip()
            if not raw:
                continue
            resolved = canonical_participant_id(raw, [*canonical, *theme_ids])
            canonical.append(resolved)
    return list(dict.fromkeys(canonical))


def _normalize_prop_fields(shot: dict, themes: Iterable[str]) -> list[str]:
    declared = _declared_prop_ids(shot)
    if declared is not None:
        for field in ('key_props', 'continuity_props'):
            if isinstance(shot.get(field), list):
                shot[field] = _entity_ids(shot[field])
        return declared
    canonical: list[str] = []
    theme_ids = _entity_ids(themes)
    for field in ("key_props", "continuity_props"):
        values = shot.get(field, [])
        if not isinstance(values, list):
            continue
        normalized = []
        for value in values:
            raw = str(value).strip()
            if not raw:
                continue
            resolved = canonical_participant_id(raw, [*canonical, *theme_ids])
            normalized.append(resolved)
            canonical.append(resolved)
        shot[field] = list(dict.fromkeys(normalized))
    return list(dict.fromkeys(canonical))


def _declared_prop_ids(shot: dict) -> list[str] | None:
    # 新结构化创作的登记已经冻结，不能把两个独立主键当成描述性别名合并。
    if (shot.get('production_slot') or {}).get('authoring_protocol') != 'scene-units-v1':
        return None
    return _entity_ids(value for field in ('key_props', 'continuity_props')
                       if isinstance(shot.get(field), list) for value in shot[field])


def iter_structured_entity_references(
    shot: dict,
    *,
    include_action_targets: bool = True,
) -> tuple[StructuredEntityReference, ...]:
    """Enumerate every entity-bearing contract field from one authoritative map."""
    references: list[StructuredEntityReference] = []

    for value in shot.get("characters", []):
        references.append(StructuredEntityReference("characters", value, "character"))
    for value in shot.get("required_visible_entities", []):
        references.append(StructuredEntityReference(
            "required_visible_entities", value, "target"
        ))

    camera = shot.get("camera", {})
    positions = camera.get("screen_positions", {}) if isinstance(camera, dict) else {}
    if isinstance(positions, dict):
        references.extend(
            StructuredEntityReference("camera.screen_positions", name, "target")
            for name in positions
        )

    blocking = shot.get("blocking", {})
    if isinstance(blocking, dict):
        for name, intent in blocking.items():
            references.append(StructuredEntityReference("blocking", name, "character"))
            if include_action_targets and isinstance(intent, dict):
                references.append(StructuredEntityReference(
                    "blocking.action_target", intent.get("action_target"), "target"
                ))

    for beat in shot.get("action_beats", []):
        if not isinstance(beat, dict):
            continue
        scene_subject = ((shot.get('production_slot') or {}).get('authoring_protocol') == 'scene-units-v1'
                         and beat.get('activity') in INDEPENDENT_ACTIVITIES)
        references.append(StructuredEntityReference(
            "action_beats.actor", beat.get("actor"), "target" if scene_subject else "character"
        ))
        references.append(StructuredEntityReference(
            "action_beats.target", beat.get("target"), "target"
        ))
        for change in beat.get('scene_changes') or []:
            if isinstance(change, dict):
                references.append(StructuredEntityReference(
                    'action_beats.scene_changes.entity', change.get('entity'), 'target'))
        geometry = beat.get("interaction_geometry")
        if isinstance(geometry, dict):
            for field, kind in (("actor", "character"), ("target", "target")):
                references.append(StructuredEntityReference(
                    f"action_beats.interaction_geometry.{field}", geometry.get(field), kind
                ))

    geometry = shot.get("interaction_geometry", {})
    if isinstance(geometry, dict):
        references.append(StructuredEntityReference(
            "interaction_geometry.actor", geometry.get("actor"), "character"
        ))
        references.append(StructuredEntityReference(
            "interaction_geometry.target", geometry.get("target"), "target"
        ))
    return tuple(references)


def structured_entity_reference_issues(
    shot: dict,
    character_names: Iterable[str],
    theme_elements: Iterable[str] = (),
) -> list[str]:
    """Validate every structured entity reference against one registry."""
    registry = shot_entity_registry(shot, character_names, theme_elements)
    if not registry.characters and (shot.get('production_slot') or {}).get('authoring_protocol') != 'scene-units-v1':
        return []

    shot_id = shot.get("shot_id", "?")
    issues: list[str] = []

    def validate(value: object, field: str, allowed: tuple[str, ...]) -> None:
        raw = str(value or "").strip()
        if raw and raw.casefold() != "none" and raw not in allowed:
            if field == "characters":
                issues.append(f"Shot {shot_id}: 未定义角色 {raw}")
            else:
                issues.append(f"Shot {shot_id}: {field} 引用未注册实体 {raw}")

    blocking = shot.get("blocking", {})
    has_action_target = False
    if isinstance(blocking, dict):
        geometry = shot.get("interaction_geometry", {})
        phase = str(geometry.get("effect_phase", "")).strip() if isinstance(geometry, dict) else ""
        has_action_target = phase == "active" or any(
            isinstance(beat, dict) and str(beat.get("target", "")).strip()
            for beat in shot.get("action_beats", [])
        )
    for reference in iter_structured_entity_references(
        shot,
        include_action_targets=has_action_target,
    ):
        allowed = (
            registry.characters
            if reference.kind == "character"
            else registry.target_ids
        )
        validate(reference.value, reference.field, allowed)
    return list(dict.fromkeys(issues))


def normalize_shot_participants(
    shot: dict,
    character_names: Iterable[str],
    theme_elements: Iterable[str] = (),
) -> None:
    """Rewrite unambiguous shot-local participant aliases to catalog IDs."""
    catalog = _entity_ids(character_names)
    props = _normalize_prop_fields(shot, theme_elements)
    declared = [
        str(name).strip()
        for name in shot.get("characters", [])
        if str(name).strip()
    ]
    replacements: dict[str, str] = {}
    prop_replacements: set[str] = set()
    occupied: set[str] = set()
    unresolved: list[str] = []
    for name in declared:
        resolved = canonical_entity_id(name, catalog, exclude=occupied)
        if resolved in catalog and resolved not in occupied:
            replacements[name] = resolved
            occupied.add(resolved)
            continue
        prop = canonical_participant_id(name, props)
        if prop in props:
            replacements[name] = prop
            prop_replacements.add(name)
            continue
        unresolved.append(name)

    if not replacements:
        return

    def replace(value: object) -> object:
        raw = str(value).strip()
        return replacements.get(raw, value)

    shot["characters"] = list(dict.fromkeys(
        resolved
        for name in declared
        if name not in prop_replacements
        and (resolved := str(replace(name)).strip())
    ))
    for field in ("required_visible_entities",):
        values = shot.get(field)
        if isinstance(values, list):
            shot[field] = list(dict.fromkeys(
                str(replace(value)).strip() for value in values if str(value).strip()
            ))

    camera = shot.get("camera")
    if isinstance(camera, dict) and isinstance(camera.get("screen_positions"), dict):
        camera["screen_positions"] = {
            str(replace(name)).strip(): position
            for name, position in camera["screen_positions"].items()
        }

    blocking = shot.get("blocking")
    if isinstance(blocking, dict):
        normalized_blocking = {}
        for name, intent in blocking.items():
            if isinstance(intent, dict):
                intent = dict(intent)
                for field in _BLOCKING_TARGET_FIELDS:
                    if field in intent:
                        intent[field] = replace(intent[field])
            normalized_blocking[str(replace(name)).strip()] = intent
        shot["blocking"] = normalized_blocking

    for beat in shot.get("action_beats", []):
        if isinstance(beat, dict):
            for field in ("actor", "target"):
                beat[field] = replace(beat.get(field, ""))
            geometry = beat.get("interaction_geometry")
            if isinstance(geometry, dict):
                for field in ("actor", "target"):
                    geometry[field] = replace(geometry.get(field, ""))

    geometry = shot.get("interaction_geometry")
    if isinstance(geometry, dict):
        for field in ("actor", "target"):
            geometry[field] = replace(geometry.get(field, ""))


def normalize_structured_entity_references(
    shot: dict,
    registry: EntityRegistry,
) -> None:
    """Canonicalize every structured entity reference from one shot registry.

    LLMs often use descriptive aliases such as ``remaining_zombies`` in one
    field and the registered ``zombie_group`` in another.  Keeping those
    aliases until readiness makes equivalent plans look inconsistent and
    blocks generation.  This pass is deliberately field-aware: actor/blocking
    keys are characters, target/visibility/position references may be any
    registered entity, and natural-language facing/eyeline descriptions are
    preserved for spatial semantics.
    """
    reference_map: dict[tuple[str, str], str] = {}
    for reference in iter_structured_entity_references(shot):
        raw = str(reference.value or "").strip()
        if not raw:
            continue
        if reference.kind == "character":
            resolved = canonical_entity_id(raw, registry.characters)
        else:
            resolved = canonical_target_id(raw, registry)
        reference_map[(reference.field, raw)] = resolved

    def resolve(
        field: str,
        value: object,
        kind: Literal["character", "target"],
        *,
        exclude: Iterable[str] = (),
    ) -> str:
        raw = str(value or "").strip()
        if not raw:
            return raw
        mapped = reference_map.get((field, raw))
        if mapped is not None and not exclude:
            return mapped
        return (
            canonical_entity_id(raw, registry.characters, exclude=exclude)
            if kind == "character"
            else canonical_target_id(raw, registry, exclude=exclude)
        )

    required = shot.get("required_visible_entities")
    if isinstance(required, list):
        normalized: list[str] = []
        for value in required:
            resolved = resolve("required_visible_entities", value, "target")
            if resolved and resolved not in normalized:
                normalized.append(resolved)
        shot["required_visible_entities"] = normalized

    camera = shot.get("camera")
    if isinstance(camera, dict) and isinstance(camera.get("screen_positions"), dict):
        normalized_positions: dict[str, object] = {}
        for raw_name, position in camera["screen_positions"].items():
            resolved = resolve("camera.screen_positions", raw_name, "target")
            if not resolved:
                continue
            # Prefer an explicitly canonical key when both alias and canonical
            # entries are present; otherwise retain first-seen composition.
            if resolved not in normalized_positions or str(raw_name).strip() == resolved:
                normalized_positions[resolved] = position
        camera["screen_positions"] = normalized_positions

    blocking = shot.get("blocking")
    if isinstance(blocking, dict):
        normalized_blocking: dict[str, object] = {}
        for raw_name, intent in blocking.items():
            resolved_name = resolve("blocking", raw_name, "character")
            if isinstance(intent, dict):
                normalized_intent = dict(intent)
                if "action_target" in normalized_intent:
                    normalized_intent["action_target"] = resolve(
                        "blocking.action_target",
                        normalized_intent["action_target"],
                        "target",
                        exclude=[resolved_name],
                    )
                intent = normalized_intent
            if (
                resolved_name not in normalized_blocking
                or str(raw_name).strip() == resolved_name
            ):
                normalized_blocking[resolved_name] = intent
        shot["blocking"] = normalized_blocking

    beats = shot.get("action_beats")
    if isinstance(beats, list):
        for beat in beats:
            if not isinstance(beat, dict):
                continue
            actor = resolve("action_beats.actor", beat.get("actor"), "character")
            beat["actor"] = actor
            beat["target"] = resolve(
                "action_beats.target", beat.get("target"), "target", exclude=[actor]
            )
            geometry = beat.get("interaction_geometry")
            if isinstance(geometry, dict):
                event_actor = resolve(
                    "action_beats.interaction_geometry.actor", geometry.get("actor"), "character"
                )
                geometry["actor"] = event_actor
                geometry["target"] = resolve(
                    "action_beats.interaction_geometry.target", geometry.get("target"),
                    "target", exclude=[event_actor],
                )

    geometry = shot.get("interaction_geometry")
    if isinstance(geometry, dict):
        actor = resolve("interaction_geometry.actor", geometry.get("actor"), "character")
        geometry["actor"] = actor
        geometry["target"] = resolve(
            "interaction_geometry.target", geometry.get("target"), "target",
            exclude=[actor],
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
