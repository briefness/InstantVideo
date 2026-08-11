"""Manifest-derived statistics and conservative repeated-failure decisions."""

from pathlib import Path

from pipeline.models import RunOptions
from pipeline.production_analytics import (
    assess_failure_pattern,
    derive_production_statistics,
    production_statistics_from_manifest,
)
from pipeline.run_state import RunWorkspace


def test_failure_pattern_requires_three_observations():
    result = assess_failure_pattern(
        accepted=0,
        failure_signatures=["contract:x", "contract:x"],
    )

    assert result == {
        "observations": 2,
        "accepted": 0,
        "failed": 2,
        "decision": "insufficient_evidence",
    }


def test_repeated_identical_failures_are_only_a_structural_candidate():
    result = assess_failure_pattern(
        accepted=0,
        failure_signatures=["contract:x", "contract:x", "contract:x"],
    )

    assert result["decision"] == "structural_candidate"
    assert result["dominant_failure_count"] == 3


def test_mixed_success_and_failure_is_not_labeled_stochastic():
    result = assess_failure_pattern(
        accepted=1,
        failure_signatures=["technical:x", "technical:x"],
    )

    assert result["decision"] == "variable_outcomes"


def test_repeated_transient_failures_are_not_structural_candidates():
    result = assess_failure_pattern(
        accepted=0,
        failure_signatures=[
            "provider_transient:provider.rate_limit",
            "provider_transient:provider.rate_limit",
            "provider_transient:provider.rate_limit",
        ],
    )

    assert result["decision"] == "inconclusive"


def test_statistics_derive_repeated_take_evidence_from_manifest(tmp_path: Path):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="analytics"))
    fingerprint = "a" * 64
    for take_number in range(1, 4):
        video = workspace.path / "shots" / f"shot_001_rejected_{take_number:02d}.mp4"
        video.parent.mkdir(exist_ok=True)
        video.write_bytes(b"video")
        workspace.record_shot(
            shot_id=1,
            status="failed",
            local_path=str(video),
            semantic_accepted=False,
            prompt_fingerprint=fingerprint,
            rejection_evidence=[{
                "category": "contract",
                "code": "contract.primary_action_incomplete",
                "source": "semantic_review",
            }],
        )

    stats = production_statistics_from_manifest(workspace.manifest_path)

    assert stats["takes"] == {"total": 3, "accepted": 0, "rejected": 3}
    assert stats["failure_evidence"]["by_category"] == {"contract": 3}
    assert stats["prompt_failure_patterns"][fingerprint]["decision"] == (
        "structural_candidate"
    )


def test_legacy_rejected_take_is_unknown_without_parsing_error_text(tmp_path: Path):
    workspace = RunWorkspace.create(tmp_path, RunOptions(request="legacy"))
    video = workspace.path / "shots" / "shot_001_rejected_01.mp4"
    video.parent.mkdir(exist_ok=True)
    video.write_bytes(b"video")
    workspace.record_shot(
        shot_id=1,
        status="failed",
        local_path=str(video),
        semantic_accepted=False,
        errors=["some historical localized error"],
    )
    payload = workspace.manifest.model_dump(mode="json", exclude_none=True)
    payload["shots"]["1"]["take_history"][0].pop("rejection_evidence", None)
    payload["options"]["resolution"] = "1080p"

    stats = derive_production_statistics(payload)

    assert stats["failure_evidence"]["by_code"] == {
        "legacy.take_unclassified": 1
    }
