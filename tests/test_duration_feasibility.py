"""时长是用户目标、容差和执行能力的共同约束，不由题材决定。"""
from copy import deepcopy
import json

import pytest

import config
from pipeline.production_plan import build_production_plan, production_plan_issues
from pipeline.production_shape import allocate_requested_shape
from test_flexible_duration_plan import flexible
from test_independent_requirement_review import offline


def limits(maximum=15):
    return {'min_duration': 4, 'max_duration': maximum}


@pytest.mark.parametrize('target', [24, 151, 180, 300, 600])
def test_automatic_topology_keeps_target_reachable(target):
    value = flexible(target, None)
    allocation = allocate_requested_shape(target, value, duration_limits=limits())
    assert sum(allocation) == target
    assert all(4 <= d <= 15 for d in allocation)


def test_configured_limits_and_requested_count_share_feasible_region():
    value = flexible(22, 4)
    assert allocate_requested_shape(22, value, duration_limits=limits(5)) == [5] * 4
    with pytest.raises(ValueError):
        allocate_requested_shape(22, {**value, 'duration_mode': 'exact'}, duration_limits=limits(5))
    with pytest.raises(ValueError):
        allocate_requested_shape(22, value, [6, 6, 5, 5], duration_limits=limits(5))


def test_feasibility_is_complete_across_all_supported_targets_and_profiles():
    # 独立穷举作为小规模整数约束求解的判定器，不复写分配算法。
    for maximum in range(4, 16):
        for target in range(4, 601):
            value = flexible(target, None)
            margin = max(1, target // 10)
            lower, upper = max(4, target - margin), min(600, target + margin)
            candidates = [total for total in range(lower, upper + 1)
                          if any(n * 4 <= total <= n * maximum for n in range(1, 151))]
            if not candidates:
                with pytest.raises(ValueError):
                    allocate_requested_shape(target, value, duration_limits=limits(maximum))
                continue
            allocation = allocate_requested_shape(target, value, duration_limits=limits(maximum))
            assert min(abs(total - target) for total in candidates) == abs(sum(allocation) - target)
            assert all(4 <= duration <= maximum for duration in allocation)


@pytest.mark.parametrize('bad', [True, {}, {'min_duration': 5, 'max_duration': 4},
    {'min_duration': 4, 'max_duration': 16}, {'min_duration': 4, 'max_duration': True},
    {'min_duration': 4, 'max_duration': 5, 'ignored': 1}])
def test_invalid_capacity_is_rejected(bad):
    with pytest.raises(ValueError):
        allocate_requested_shape(24, flexible(), duration_limits=bad)


def test_legacy_plan_is_replayed_with_its_original_allocation():
    value = flexible(180, None)
    legacy = build_production_plan('action', 180, intent=value)
    assert legacy['planned_duration'] == 165
    assert 'duration_limits' not in legacy
    plan = build_production_plan('action', 180, intent=value, duration_limits=limits())
    assert plan['planned_duration'] == 180 and plan['duration_limits'] == limits()


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_formal_authoring_uses_frozen_actual_capacity(monkeypatch, topic):
    from test_creation_schedule_flow import scenario_story
    from test_director_intent import Responses
    from pipeline.storyboard import generate_storyboard
    brief, value, authored, _ = scenario_story(topic)
    value.update(duration_mode='approximate', requested_duration_seconds=16,
                 duration_evidence=brief, requested_shot_count=3, shot_count_evidence=brief,
                 pacing='balanced')
    authored['shot_durations'] = [5, 5, 5]
    chains = deepcopy(config.GENERATION_CHAINS)
    for profiles in chains.values():
        for profile in profiles:
            profile['max_duration'] = 5
    monkeypatch.setattr(config, 'GENERATION_CHAINS', chains)
    client = Responses(value, authored)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_: client)
    result = generate_storyboard(brief, target_duration=16, allow_repair=False)
    assert result['total_duration'] == 15
    assert result['production_plan']['duration_limits'] == limits(5)
    assert not production_plan_issues(result)
    context = json.loads(client.calls[1]['messages'][-1]['content'])
    assert context['execution_capabilities']['hard_limits']['max_duration'] == 5
    assert context['duration_budget']['minimum_seconds'] == 15
    assert context['duration_budget']['maximum_seconds'] == 15
    assert len(client.calls) == 5
    changed = deepcopy(result)
    changed['production_plan']['duration_limits']['max_duration'] = 4
    assert production_plan_issues(changed)
