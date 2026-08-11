"""Regression corpus for stable failure evidence classification."""

import json
from pathlib import Path

from pipeline.failure_evidence import (
    provider_failure_evidence,
    semantic_review_failure_evidence,
)
from pipeline.semantic_review import SemanticReview


def test_provider_failure_regression_corpus():
    cases = json.loads(
        (Path(__file__).parent / "fixtures" / "failure_evidence_cases.json")
        .read_text(encoding="utf-8")
    )

    for case in cases:
        evidence = provider_failure_evidence(
            case["error_type"],
            error_code=case["error_code"],
            error_locus=case["error_locus"],
        )
        assert evidence[0]["category"] == case["category"], case["name"]
        assert evidence[0]["code"] == case["code"], case["name"]


def test_semantic_evidence_uses_objective_fields_not_free_text_reason():
    review = SemanticReview(
        accepted=False,
        required_entities_visible={"hero": True, "target": False},
        action_geometry_valid=False,
        primary_action_completed=True,
        observed_end_state={},
        boundary_continuity_valid=False,
        failure_reason="arbitrary evaluator prose",
    )

    evidence = semantic_review_failure_evidence(review)

    assert {item["code"] for item in evidence} == {
        "contract.required_entity_missing",
        "contract.action_geometry_invalid",
        "state.boundary_discontinuity",
    }
    assert evidence[0]["details"] == {"entities": "target"}
