"""创作前给定总时长容差，创作后冻结实际执行时间；不在执行端拉伸或补帧。"""
from copy import deepcopy

import pytest

from pipeline.production_plan import build_production_plan
from test_request_shape import intent
from test_director_intent import Responses
from test_independent_requirement_review import offline


def flexible(target=24, count=4):
    return {**intent(target, count), 'duration_mode': 'approximate',
            'choreography': 'flexible', 'authoring_protocol': 'scene-units-v1'}


@pytest.mark.parametrize('allocation', [[4, 8, 7, 5], [5, 8, 7, 6], [4, 7, 6, 5]])
def test_nonuniform_content_allocation_within_total_envelope(allocation):
    plan = build_production_plan('action', 24, intent=flexible(), duration_allocation=allocation)
    assert plan['target_duration'] == 24 and plan['planned_duration'] == sum(allocation)
    assert [s['duration'] for s in plan['slots']] == allocation
    assert plan['duration_allocation'] == allocation


@pytest.mark.parametrize('allocation', [[4, 4, 4, 4], [7, 7, 7, 7], [0, 8, 8, 8],
    [4, 16, 4, 0], [True, 8, 8, 7], [6., 6, 6, 6], [8, 8, 8]])
def test_bad_allocation_rejected_without_clamping(allocation):
    with pytest.raises(ValueError):
        build_production_plan('action', 24, intent=flexible(), duration_allocation=allocation)


def test_exact_or_legacy_requirement_cannot_be_silently_relaxed():
    for mode in [None, 'exact']:
        value = intent(24, 4)
        if mode:
            value['duration_mode'] = mode
        with pytest.raises(ValueError):
            build_production_plan('action', 24, intent=value, duration_allocation=[5, 8, 7, 6])
    old = build_production_plan('action', 24, intent=intent(24, 4))
    assert [s['duration'] for s in old['slots']] == [6]*4 and 'duration_allocation' not in old


def test_shape_can_use_nearest_feasible_duration_only_with_explicit_flexibility():
    plan = build_production_plan('action', 15, intent=flexible(15, 4))
    assert plan['target_duration'] == 15 and plan['planned_duration'] == 16
    with pytest.raises(ValueError):
        build_production_plan('action', 15, intent=intent(15, 4))


@pytest.mark.parametrize('topic', ['双人舞', '货物交接', '绳索救援'])
def test_formal_authoring_freezes_content_durations_before_shot_requests(monkeypatch, topic):
    from test_creation_schedule_flow import scenario_story
    from pipeline.storyboard import generate_storyboard
    from pipeline.production_plan import production_plan_issues
    from pipeline.readiness import storyboard_readiness_issues
    brief, value, authored, _ = scenario_story(topic)
    value.update(duration_mode='approximate', requested_duration_seconds=15, duration_evidence=brief,
                 requested_shot_count=3, shot_count_evidence=brief)
    authored['shot_durations'] = [4, 6, 6]
    # 独立合成创作，保留内容；不把旧真实分镜手工改成新时长。
    value['pacing'] = 'balanced'
    client = Responses(value, authored)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **_:client)
    result = generate_storyboard(brief, target_duration=15, allow_repair=False)
    assert result['total_duration'] == 16
    assert [s['duration'] for s in result['shots']] == [4, 6, 6]
    assert not production_plan_issues(result)
    assert not storyboard_readiness_issues(result)
    contexts = [__import__('json').loads(r['messages'][-1]['content']) for r in client.calls[1:]]
    assert contexts[0]['duration_budget']['maximum_seconds'] == 16
    assert [c['slot']['duration'] for c in contexts[1:]] == [4, 6, 6]
    assert len(client.calls) == 5
    bad = deepcopy(result)
    bad['shots'][0]['duration'] += 1
    assert production_plan_issues(bad)

