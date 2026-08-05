"""Narrative state-change contract shared by planning, prompting, and readiness."""

from __future__ import annotations


_ARC_FIELDS = ("goal", "stakes", "turning_point", "resolution")
_BEAT_FIELDS = ("function", "state_before", "state_change", "state_after")
_COVERAGE_CARRIERS = {
    "establish": "make the initial situation and its change readable in the frame",
    "action_subject": "keep the primary subject's visible behavior readable",
    "target_reaction": "keep the intended receiver's visible response readable",
    "interaction": "keep the visible cause and response in one readable spatial relation",
    "aftermath": "hold the established result long enough to read",
    "insert": "use the declared detail as visible evidence of the change",
}


def narrative_contract_present(storyboard: dict) -> bool:
    if isinstance(storyboard.get("story_arc"), dict):
        return True
    return any(
        isinstance(shot.get("narrative_beat"), dict)
        for shot in storyboard.get("shots", [])
        if isinstance(shot, dict)
    )


def narrative_readiness_issues(
    storyboard: dict,
    *,
    required: bool = False,
) -> list[str]:
    """Reject incomplete or reset story state without prescribing a plot formula."""
    if not required and not narrative_contract_present(storyboard):
        return []

    issues: list[str] = []
    arc = storyboard.get("story_arc")
    arc = arc if isinstance(arc, dict) else {}
    for field in _ARC_FIELDS:
        if not str(arc.get(field, "")).strip():
            issues.append(f"story_arc.{field} 不能为空")

    previous_after = ""
    previous_id: object = "?"
    for shot in storyboard.get("shots", []):
        if not isinstance(shot, dict):
            continue
        shot_id = shot.get("shot_id", "?")
        beat = shot.get("narrative_beat")
        beat = beat if isinstance(beat, dict) else {}
        for field in _BEAT_FIELDS:
            if not str(beat.get(field, "")).strip():
                issues.append(f"Shot {shot_id}: narrative_beat.{field} 不能为空")

        before = _state_key(beat.get("state_before"))
        after = _state_key(beat.get("state_after"))
        if before and after and before == after:
            issues.append(f"Shot {shot_id}: 故事状态没有发生可见变化")
        if previous_after and before and previous_after != before:
            issues.append(
                f"Shot {shot_id}: 故事状态交接断裂，state_before 必须复用 "
                f"Shot {previous_id} 的 state_after"
            )
        if after:
            previous_after = after
            previous_id = shot_id
    return issues


def normalize_narrative_handoffs(storyboard: dict) -> list[tuple[object, object]]:
    """Compile each shot's entry state from the preceding canonical result."""
    corrections: list[tuple[object, object]] = []
    shots = [shot for shot in storyboard.get("shots", []) if isinstance(shot, dict)]
    for previous, current in zip(shots, shots[1:]):
        previous_beat = previous.get("narrative_beat")
        current_beat = current.get("narrative_beat")
        if not isinstance(previous_beat, dict) or not isinstance(current_beat, dict):
            continue
        previous_after = str(previous_beat.get("state_after", "")).strip()
        if not previous_after or current_beat.get("state_before") == previous_after:
            continue
        current_beat["state_before"] = previous_after
        corrections.append((current.get("shot_id", "?"), previous.get("shot_id", "?")))
    return corrections


def compile_narrative_carriers(shot: dict) -> tuple[dict[str, str], ...]:
    """Project a Narrative Beat into concrete evidence without enlarging LLM JSON."""
    beat = shot.get("narrative_beat")
    if not isinstance(beat, dict):
        return ()
    change = str(beat.get("state_change", "")).strip()
    after = str(beat.get("state_after", "")).strip()
    if not change or not after:
        return ()

    carriers = [
        {"kind": "visible_change", "value": change},
        {"kind": "readable_endpoint", "value": after},
    ]
    coverage = str(shot.get("coverage_role", "")).strip()
    coverage_carrier = _COVERAGE_CARRIERS.get(coverage)
    if coverage_carrier:
        carriers.append({"kind": "coverage", "value": coverage_carrier})
    if coverage == "insert":
        detail = next(
            (
                str(prop).strip()
                for prop in shot.get("key_props", [])
                if str(prop).strip()
            ),
            "",
        )
        if detail:
            carriers.append({"kind": "signature_detail", "value": detail})
    return tuple(carriers)


def narrative_prompt_constraint(shot: dict) -> str:
    beat = shot.get("narrative_beat")
    if not isinstance(beat, dict):
        return ""
    function = str(beat.get("function", "")).strip()
    before = str(beat.get("state_before", "")).strip()
    carriers = compile_narrative_carriers(shot)
    if not function or not before or not carriers:
        return ""
    carrier_text = "; ".join(
        f"{carrier['kind']}={carrier['value']}" for carrier in carriers
    )
    return (
        f"narrative {function}: begin with {before}; filmable carriers: "
        f"{carrier_text}; camera movement or mood alone cannot express the change"
    )


def narrative_review_instruction(shot: dict) -> str:
    """Compile the same filmable carriers into the semantic review instruction."""
    carriers = compile_narrative_carriers(shot)
    if not carriers:
        return ""
    carrier_text = "; ".join(
        f"{carrier['kind']}={carrier['value']}" for carrier in carriers
    )
    return (
        "For narrative_state_change_valid, compare the earliest and latest current "
        "samples and require the contracted filmable carriers to be visibly readable: "
        f"{carrier_text}. Do not infer completion from the prompt, smoke, a partial "
        "reaction, camera movement, mood, or a new angle alone. "
    )


def requires_narrative_review(shot: dict) -> bool:
    beat = shot.get("narrative_beat")
    return isinstance(beat, dict) and all(
        str(beat.get(field, "")).strip() for field in _BEAT_FIELDS
    )


def _state_key(value: object) -> str:
    return " ".join(str(value or "").casefold().split())
