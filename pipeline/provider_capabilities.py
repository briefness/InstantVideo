"""Internal consistency checks for the configured provider capability contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import config


def capability_contract_issues(
    *,
    resolutions: Sequence[str] | None = None,
    aspect_ratios: Sequence[str] | None = None,
    dimensions: Mapping[str, Mapping[str, str]] | None = None,
    generation_chains: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    default_resolution: str | None = None,
    default_ratio: str | None = None,
) -> list[str]:
    """Return deterministic configuration gaps without contacting the provider."""
    supported_resolutions = tuple(
        config.SUPPORTED_RESOLUTIONS if resolutions is None else resolutions
    )
    supported_ratios = tuple(
        config.SUPPORTED_ASPECT_RATIOS if aspect_ratios is None else aspect_ratios
    )
    output_dimensions = (
        config.SEEDANCE_OUTPUT_DIMENSIONS if dimensions is None else dimensions
    )
    chains = config.GENERATION_CHAINS if generation_chains is None else generation_chains
    selected_default_resolution = (
        config.DEFAULT_RESOLUTION
        if default_resolution is None
        else default_resolution
    )
    selected_default_ratio = (
        config.DEFAULT_RATIO if default_ratio is None else default_ratio
    )
    issues: list[str] = []

    if len(set(supported_resolutions)) != len(supported_resolutions):
        issues.append("supported resolutions contain duplicates")
    if len(set(supported_ratios)) != len(supported_ratios):
        issues.append("supported aspect ratios contain duplicates")
    if selected_default_resolution not in supported_resolutions:
        issues.append("default resolution is not supported")
    if selected_default_ratio not in supported_ratios:
        issues.append("default aspect ratio is not supported")

    resolution_set = set(supported_resolutions)
    ratio_set = set(supported_ratios)
    if set(output_dimensions) != resolution_set:
        issues.append("output dimension resolutions do not match supported resolutions")
    if set(chains) != resolution_set:
        issues.append("generation chain resolutions do not match supported resolutions")

    for resolution in supported_resolutions:
        ratio_dimensions = output_dimensions.get(resolution, {})
        if set(ratio_dimensions) != ratio_set:
            issues.append(f"{resolution} dimensions do not cover every supported ratio")
        for ratio, value in ratio_dimensions.items():
            parts = str(value).split(":")
            if (
                len(parts) != 2
                or not all(part.isdigit() and int(part) > 0 for part in parts)
            ):
                issues.append(f"{resolution}/{ratio} has invalid dimensions: {value}")

        profiles = list(chains.get(resolution, ()))
        if not profiles:
            issues.append(f"{resolution} generation chain is empty")
            continue
        if profiles[0].get("resolution") != resolution:
            issues.append(f"{resolution} generation chain does not start at requested quality")
        seen_profiles: set[tuple[object, object]] = set()
        for profile in profiles:
            profile_resolution = profile.get("resolution")
            profile_key = (profile.get("model"), profile_resolution)
            if profile_key in seen_profiles:
                issues.append(f"{resolution} generation chain contains duplicate profiles")
            seen_profiles.add(profile_key)
            if not str(profile.get("model") or "").strip():
                issues.append(f"{resolution} generation profile has no model")
            if profile_resolution not in resolution_set:
                issues.append(
                    f"{resolution} generation profile uses unsupported resolution "
                    f"{profile_resolution}"
                )
            max_duration = profile.get("max_duration")
            if (
                isinstance(max_duration, bool)
                or not isinstance(max_duration, int)
                or not config.MIN_SHOT_DURATION
                <= max_duration
                <= config.MAX_SHOT_DURATION
            ):
                issues.append(f"{resolution} generation profile has invalid max_duration")
    return issues
