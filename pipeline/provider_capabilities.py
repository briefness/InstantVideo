"""Internal consistency checks for the configured provider capability contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import config


def execution_profile(resolution: str, *, model: str | None = None,
                      requested_resolution: str | None = None) -> dict:
    """选取实际配置的执行档位；创意标签没有改变接口能力的权限。"""
    if (not isinstance(resolution, str) or not resolution
            or requested_resolution is not None and (not isinstance(requested_resolution, str) or not requested_resolution)
            or model is not None and (not isinstance(model, str) or not model.strip())):
        raise ValueError('provider model/resolution 必须是有效的配置标识')
    issues = capability_contract_issues()
    if issues:
        raise ValueError('Provider capability contract 不完整: ' + '; '.join(issues))
    requested = resolution if requested_resolution is None else requested_resolution
    chain = config.GENERATION_CHAINS.get(requested, ())
    matches = [p for p in chain if p['resolution'] == resolution
               and (model is None or p['model'] == model)]
    if not matches:
        raise ValueError(f'provider model={model}、resolution={resolution} 不属于请求分辨率 {requested} 的 generation chain')
    profile = matches[0]
    return {'model': profile['model'], 'resolution': resolution,
            'min_duration': config.MIN_SHOT_DURATION,
            'max_duration': min(p['max_duration'] for p in matches if p['model']==profile['model']),
            'aspect_ratios': list(config.SUPPORTED_ASPECT_RATIOS)}


def execution_request_issues(*, model: str | None = None, resolution: str,
                             ratio: str, duration: int,
                             requested_resolution: str | None = None) -> list[str]:
    try:
        profile = execution_profile(resolution, model=model, requested_resolution=requested_resolution)
    except ValueError as exc:
        return [str(exc)]
    issues = []
    if ratio not in profile['aspect_ratios']:
        issues.append(f'provider 不支持画幅 {ratio}')
    if type(duration) is not int or not profile['min_duration'] <= duration <= config.MAX_SHOT_DURATION:
        issues.append(f'provider 时长必须是 {profile["min_duration"]}-{config.MAX_SHOT_DURATION} 秒整数')
    elif duration > profile['max_duration']:
        issues.append(f'duration={duration} 超出当前 generation profile 上限 {profile["max_duration"]}；未截短计划时长')
    return issues


def planning_execution_contract(resolution: str, ratio: str, plan: dict | None = None) -> dict:
    """创作前拒绝不可执行的既定槽位，同一限制在提交边界再次验证。"""
    profile = execution_profile(resolution)
    issues = execution_request_issues(resolution=resolution, ratio=ratio, duration=profile['min_duration'])
    for slot in (plan or {}).get('slots', []):
        issues.extend(f'Shot {slot["shot_id"]}: {issue}' for issue in execution_request_issues(
            resolution=resolution, ratio=ratio, duration=slot['duration']))
    if issues:
        raise ValueError('执行能力约束不满足：'+'；'.join(issues))
    return {'version':'configured-execution-v1', 'hard_limits':profile,
            'duration_policy':'explicit-integer-seconds; preserve planned duration',
            'semantic_dimensions':{
                'camera.speed':'camera movement only; not performer tempo or a duration limit',
                'motion_timing.performer_tempo':'subject motion tempo',
                'motion_timing.playback':'real-time/slow-motion treatment',
                'interaction_seconds':'activity occupancy, not speed'},
            'quality_policy':'Camera and action readability require evidence; style labels do not prove success or impose provider limits.'}


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
