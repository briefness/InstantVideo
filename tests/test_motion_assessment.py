"""Counterexamples for human evidence interpretation, not simulated vision accuracy."""

import copy

import pytest

from pipeline.motion_assessment import (
    REVIEW_VERSION, assess_document, assess_rating, blank_rating, compare_assessments, contract_digest,
)

IDENTIFIER = "a" * 32
ENTRY = {"sha256": "b" * 64, "frames": 96}
CONTRACT = {"actor": "attacker", "target": "defender"}


def rated(**changes):
    row = blank_rating(IDENTIFIER, ENTRY["sha256"], ENTRY["frames"])
    row.update(assessment="reviewed", identity_ok=True, scope_ok=True, endpoint_ok=True, visual_invariants_ok=True,
               causal_link_ok=True, contact_frame=48, reaction_frame=55)
    row.update(changes)
    return row


def document(row, reviewer="r1"):
    return {"version": REVIEW_VERSION, "contract": CONTRACT, "contract_sha256": contract_digest(CONTRACT),
            "reviewer_id": reviewer, "ratings": [row]}


@pytest.mark.parametrize("changes,expected", [
    ({}, "pass"),
    ({"reaction_frame": 12}, "fail"),
    ({"reaction_frame": 48}, "fail"),
    ({"contact_frame": None}, "fail"),
    ({"reaction_frame": None}, "fail"),
    ({"identity_ok": False}, "fail"),
    ({"scope_ok": False}, "fail"),
    ({"endpoint_ok": False}, "fail"),
    ({"causal_link_ok": False}, "fail"),
    ({"assessment": "unjudgeable", "notes": "接触被遮挡", "causal_link_ok": None}, "unjudgeable"),
])
def test_counterexample_observations(changes, expected):
    assert assess_rating(rated(**changes), 96) == expected


@pytest.mark.parametrize("changes", [
    {"contact_frame": True}, {"reaction_frame": 96}, {"contact_frame": -1},
    {"reaction_frame": 1.5}, {"identity_ok": "true"}, {"causal_link_ok": None},
    {"assessment": "unknown"}, {"assessment": "unreviewed"},
    {"assessment": "unjudgeable", "notes": ""}, {"notes": ["invalid"]}, {"notes": "a" * 4001},
])
def test_invalid_observations_fail_closed(changes):
    with pytest.raises(ValueError):
        assess_rating(rated(**changes), 96)


def test_unreviewed_stays_distinct_from_unjudgeable_and_failure():
    row = blank_rating(IDENTIFIER, ENTRY["sha256"], 96)
    assert assess_rating(row, 96) == "unreviewed"
    row["contact_frame"] = 999
    with pytest.raises(ValueError):
        assess_rating(row, 96)


@pytest.mark.parametrize("change", ["version", "contract", "digest", "reviewer", "duplicate", "unknown", "sha", "frames"])
def test_document_integrity(change):
    doc = document(rated())
    if change == "version":
        doc["version"] = "future-version"
    elif change == "contract":
        doc["contract"] = {"actor": "someone_else"}
    elif change == "digest":
        doc["contract_sha256"] = "changed"
    elif change == "reviewer":
        doc["reviewer_id"] = ""
    elif change == "duplicate":
        doc["ratings"].append(copy.deepcopy(doc["ratings"][0]))
    elif change == "unknown":
        doc["ratings"][0]["review_id"] = "unknown"
    elif change == "sha":
        doc["ratings"][0]["sha256"] = "changed"
    else:
        doc["ratings"][0]["frames"] = 95
    with pytest.raises(ValueError):
        assess_document(doc, {IDENTIFIER: ENTRY}, CONTRACT)


def test_legacy_readable_but_not_eligible_for_new_comparison():
    old = {"ratings": [rated()]}
    old["ratings"][0].pop("causal_link_ok")
    first = assess_document(old, {IDENTIFIER: ENTRY}, CONTRACT)
    second = assess_document(document(rated(), "r2"), {IDENTIFIER: ENTRY}, CONTRACT)
    assert first["version"] == "legacy-v1" and first["verdicts"][IDENTIFIER] == "pass"
    with pytest.raises(ValueError):
        compare_assessments(first, second, [IDENTIFIER])


@pytest.mark.parametrize("change,expected", [
    ({}, "pass"), ({"reaction_frame": 57}, "pass"), ({"reaction_frame": 58}, "disputed"),
    ({"causal_link_ok": False}, "disputed"),
    ({"assessment": "unjudgeable", "notes": "不可见"}, "disputed"),
])
def test_two_reviewers_never_hide_disagreement(change, expected):
    first = assess_document(document(rated()), {IDENTIFIER: ENTRY}, CONTRACT)
    second = assess_document(document(rated(**change), "r2"), {IDENTIFIER: ENTRY}, CONTRACT)
    result = compare_assessments(first, second, [IDENTIFIER])
    assert result["verdicts"][IDENTIFIER] == expected
    assert result["comparable"] == 1


def test_missing_rating_and_identical_reviewer_guard():
    first = assess_document(document(rated()), {IDENTIFIER: ENTRY}, CONTRACT)
    with pytest.raises(ValueError):
        compare_assessments(first, first, [IDENTIFIER])
    empty = document(blank_rating(IDENTIFIER, ENTRY["sha256"], 96), "r2")
    second = assess_document(empty, {IDENTIFIER: ENTRY}, CONTRACT)
    result = compare_assessments(first, second, [IDENTIFIER])
    assert result["verdicts"][IDENTIFIER] == "unreviewed"
    assert result["verdict_agreement"] is None


@pytest.mark.parametrize("frames", [0, -1, True, "96", 901])
def test_invalid_frame_count(frames):
    with pytest.raises(ValueError):
        assess_rating(rated(), frames)


@pytest.mark.parametrize("field,value", [("notes", []), ("reviewer_id", [])])
def test_unreviewed_forms_still_validate_types(field, value):
    doc = document(blank_rating(IDENTIFIER, ENTRY["sha256"], 96), "")
    if field == "notes":
        doc["ratings"][0][field] = value
    else:
        doc[field] = value
    with pytest.raises(ValueError):
        assess_document(doc, {IDENTIFIER: ENTRY}, CONTRACT)
