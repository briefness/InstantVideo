"""Stable, machine-readable evidence for failed provider work and rejected Takes."""

from __future__ import annotations

from typing import Any, Mapping


_PROVIDER_POLICY_TYPES = {"copyright_policy", "moderation", "privacy"}
_PROVIDER_TRANSIENT_TYPES = {
    "poll_error",
    "poll_timeout",
    "rate_limit",
    "timeout",
    "transient_provider",
}


def provider_failure_evidence(
    error_type: object,
    *,
    error_code: object = None,
    error_locus: object = None,
) -> list[dict[str, Any]]:
    """Map provider-owned structured fields without parsing localized messages."""
    normalized_type = str(error_type or "unknown").strip().casefold() or "unknown"
    if normalized_type in _PROVIDER_POLICY_TYPES:
        category = "provider_policy"
    elif normalized_type in _PROVIDER_TRANSIENT_TYPES:
        category = "provider_transient"
    else:
        category = "unknown"

    details = {"error_type": normalized_type}
    normalized_code = str(error_code or "").strip()
    normalized_locus = str(error_locus or "").strip()
    if normalized_code:
        details["provider_error_code"] = normalized_code
    if normalized_locus:
        details["provider_error_locus"] = normalized_locus
    return [{
        "category": category,
        "code": f"provider.{normalized_type}",
        "source": "provider",
        "details": details,
    }]


def technical_failure_evidence(*, unavailable: bool = False) -> list[dict[str, Any]]:
    code = "technical.qa_unavailable" if unavailable else "technical.qa_failed"
    return [{
        "category": "technical",
        "code": code,
        "source": "technical_qa",
    }]


def pipeline_state_failure_evidence(code: str) -> list[dict[str, Any]]:
    normalized = str(code).strip()
    if not normalized:
        raise ValueError("pipeline state evidence code must not be empty")
    return [{
        "category": "state",
        "code": f"state.{normalized}",
        "source": "pipeline_state",
    }]


def semantic_review_failure_evidence(review: object) -> list[dict[str, Any]]:
    """Project objective review fields; free-form failure text is not evidence."""
    if bool(getattr(review, "accepted", False)):
        return []

    evidence: list[dict[str, Any]] = []
    required_entities = getattr(review, "required_entities_visible", {})
    if isinstance(required_entities, Mapping):
        missing = sorted(
            str(name) for name, visible in required_entities.items() if not visible
        )
        if missing:
            evidence.append({
                "category": "contract",
                "code": "contract.required_entity_missing",
                "source": "semantic_review",
                "details": {"entities": ",".join(missing)},
            })

    field_codes = (
        ("action_geometry_valid", "contract", "contract.action_geometry_invalid"),
        ("effect_path_valid", "contract", "contract.effect_path_invalid"),
        ("reaction_causality_valid", "contract", "contract.reaction_causality_invalid"),
        ("action_timing_valid", "contract", "contract.action_timing_invalid"),
        ("action_sequence_valid", "contract", "contract.action_sequence_invalid"),
        ("requested_outcome_valid", "contract", "contract.requested_outcome_invalid"),
        (
            "narrative_state_change_valid",
            "contract",
            "contract.narrative_state_change_invalid",
        ),
        ("blocking_valid", "contract", "contract.blocking_invalid"),
        (
            "composition_change_valid",
            "contract",
            "contract.composition_change_invalid",
        ),
        (
            "primary_action_completed",
            "contract",
            "contract.primary_action_incomplete",
        ),
        ("boundary_continuity_valid", "state", "state.boundary_discontinuity"),
        ("identity_continuity_valid", "state", "state.identity_discontinuity"),
        (
            "environment_continuity_valid",
            "state",
            "state.environment_discontinuity",
        ),
        ("action_handoff_valid", "state", "state.action_handoff_invalid"),
        ("screen_direction_valid", "state", "state.screen_direction_invalid"),
        ("prop_continuity_valid", "state", "state.prop_discontinuity"),
        ("visual_invariants_valid", "state", "state.visual_invariant_violation"),
    )
    for field, category, code in field_codes:
        if getattr(review, field, True) is False:
            evidence.append({
                "category": category,
                "code": code,
                "source": "semantic_review",
            })

    if not evidence:
        evidence.append({
            "category": "unknown",
            "code": "semantic_review.unclassified_rejection",
            "source": "semantic_review",
        })
    return evidence


def legacy_unclassified_evidence(source: str) -> dict[str, Any]:
    """Mark missing historical structure without guessing from human text."""
    return {
        "category": "unknown",
        "code": f"legacy.{source}_unclassified",
        "source": "legacy_manifest",
    }
