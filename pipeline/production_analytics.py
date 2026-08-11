"""Derive production statistics from the existing run manifest truth."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from pipeline.failure_evidence import (
    legacy_unclassified_evidence,
    provider_failure_evidence,
)
from pipeline.models import FailureEvidenceState, RunManifest, ShotTaskState


MIN_PATTERN_OBSERVATIONS = 3
MIN_REPEATED_FAILURES = 3
DOMINANT_FAILURE_RATIO = 0.75


def assess_failure_pattern(
    *,
    accepted: int,
    failure_signatures: list[str],
    minimum_observations: int = MIN_PATTERN_OBSERVATIONS,
) -> dict[str, Any]:
    """Return a conservative candidate label, never a causal conclusion."""
    observations = accepted + len(failure_signatures)
    base = {
        "observations": observations,
        "accepted": accepted,
        "failed": len(failure_signatures),
    }
    if observations < minimum_observations:
        return {**base, "decision": "insufficient_evidence"}
    if not failure_signatures:
        return {**base, "decision": "no_failure_pattern"}

    signature_counts = Counter(failure_signatures)
    dominant_signature, dominant_count = signature_counts.most_common(1)[0]
    dominant_ratio = dominant_count / len(failure_signatures)
    detail = {
        **base,
        "dominant_failure_signature": dominant_signature,
        "dominant_failure_count": dominant_count,
    }
    # 同一提示词既成功又失败时，现有证据只支持“结果可变”，不能命名为随机故障。
    if accepted:
        return {**detail, "decision": "variable_outcomes"}
    structural_evidence = not dominant_signature.startswith(
        ("provider_transient:", "unknown:")
    )
    if (
        structural_evidence
        and dominant_count >= MIN_REPEATED_FAILURES
        and dominant_ratio >= DOMINANT_FAILURE_RATIO
    ):
        return {**detail, "decision": "structural_candidate"}
    return {**detail, "decision": "inconclusive"}


def derive_production_statistics(
    manifest: RunManifest | Mapping[str, Any],
) -> dict[str, Any]:
    """Compute read-only aggregates; the result is deliberately not persisted."""
    if isinstance(manifest, RunManifest):
        shots = list(manifest.shots.values())
    else:
        raw_shots = manifest.get("shots", {})
        if not isinstance(raw_shots, Mapping):
            raise ValueError("run manifest shots must be an object")
        # 统计只消费镜头账本；旧运行的分辨率等无关配置不应被当前能力约束拒绝。
        shots = [
            ShotTaskState.model_validate(raw_shot)
            for raw_shot in raw_shots.values()
            if isinstance(raw_shot, Mapping)
        ]
    take_counts = Counter(total=0, accepted=0, rejected=0)
    attempt_counts = Counter(total=0, pending=0, succeeded=0, failed=0)
    category_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    pattern_inputs: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"accepted": 0, "failure_signatures": []}
    )
    unclassified_failed_shots = 0

    def count_failure_event(
        records: list[FailureEvidenceState] | list[dict[str, Any]],
    ) -> str:
        normalized = [
            record
            if isinstance(record, FailureEvidenceState)
            else FailureEvidenceState.model_validate(record)
            for record in records
        ]
        for record in normalized:
            category_counts[record.category] += 1
            code_counts[record.code] += 1
        return "|".join(sorted(f"{item.category}:{item.code}" for item in normalized))

    for shot in shots:
        shot_has_classified_failure = False
        for take in shot.take_history:
            take_counts["total"] += 1
            take_counts[take.disposition] += 1
            if take.disposition == "accepted":
                if take.prompt_fingerprint:
                    pattern_inputs[take.prompt_fingerprint]["accepted"] += 1
                continue

            evidence = list(take.rejection_evidence)
            if not evidence:
                evidence = [legacy_unclassified_evidence("take")]
            signature = count_failure_event(evidence)
            shot_has_classified_failure = True
            if take.prompt_fingerprint:
                pattern_inputs[take.prompt_fingerprint][
                    "failure_signatures"
                ].append(signature)

        for attempt in shot.prompt_attempts:
            attempt_counts["total"] += 1
            attempt_counts[attempt.outcome] += 1
            if attempt.outcome != "failed":
                continue
            evidence = list(attempt.failure_evidence)
            if not evidence:
                # 旧 manifest 只在 shot 当前态保存 provider 类型；仅在指纹一致时复用。
                if (
                    shot.prompt_fingerprint == attempt.fingerprint
                    and shot.provider_error_type
                ):
                    evidence = provider_failure_evidence(
                        shot.provider_error_type,
                        error_code=attempt.provider_error_code,
                        error_locus=attempt.provider_error_locus,
                    )
                else:
                    evidence = [legacy_unclassified_evidence("provider_attempt")]
            signature = count_failure_event(evidence)
            shot_has_classified_failure = True
            pattern_inputs[attempt.fingerprint]["failure_signatures"].append(signature)

        if shot.status.value == "failed" and not shot_has_classified_failure:
            unclassified_failed_shots += 1

    patterns = {
        fingerprint: assess_failure_pattern(
            accepted=values["accepted"],
            failure_signatures=values["failure_signatures"],
        )
        for fingerprint, values in sorted(pattern_inputs.items())
    }
    return {
        "takes": dict(take_counts),
        "provider_attempts": dict(attempt_counts),
        "failure_evidence": {
            "by_category": dict(sorted(category_counts.items())),
            "by_code": dict(sorted(code_counts.items())),
            "unclassified_failed_shots": unclassified_failed_shots,
        },
        "prompt_failure_patterns": patterns,
    }


def production_statistics_from_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run manifest root must be an object")
    return derive_production_statistics(payload)
