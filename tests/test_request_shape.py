"""Requested output shape survives semantic parsing, compilation and admission."""
from copy import deepcopy

import pytest

from pipeline.director_intent import resolve_director_intent
from pipeline.models import DirectorIntentSpec
from pipeline.orchestrator import VideoPipeline
from pipeline.production_plan import build_production_plan, production_plan_issues
from test_director_intent import Responses, intent_reply, draft
from authoring_cases import authoring_from_legacy


def intent(duration, count=None):
    brief = f'总时长 {duration} 秒' + (f'，分成 {count} 段' if count is not None else '')
    return dict(source_request=brief, **intent_reply(brief),
                requested_duration_seconds=duration, duration_evidence=brief,
                requested_shot_count=count, shot_count_evidence=brief if count is not None else None)


@pytest.mark.parametrize('duration', [4, 7, 15, 30, 120, 150, 151, 180, 300, 600])
def test_supported_duration_is_preserved_exactly(duration):
    plan = build_production_plan('action', duration, intent=intent(duration))
    assert plan['planned_duration'] == plan['target_duration'] == duration
    assert sum(s['duration'] for s in plan['slots']) == duration
    assert all(4 <= s['duration'] <= 15 for s in plan['slots'])


@pytest.mark.parametrize('duration,count', [(15, 2), (30, 3), (20, 5), (180, 12), (600, 40)])
def test_explicit_shot_count_is_not_replaced_by_duration_formula(duration, count):
    plan = build_production_plan('action', duration, intent=intent(duration, count))
    assert len(plan['slots']) == count
    assert plan['planned_duration'] == duration


@pytest.mark.parametrize('duration,count', [(16, 1), (15, 4), (30, 1), (180, 3)])
def test_infeasible_shape_stops_instead_of_rewriting_requirement(duration, count):
    with pytest.raises(ValueError, match='时长|镜头|段数'):
        build_production_plan('action', duration, intent=intent(duration, count))


@pytest.mark.parametrize('duration', [0, 3, 601, True, 15.5, '15'])
def test_invalid_duration_stops_at_shared_planning_boundary(duration):
    with pytest.raises(ValueError):
        build_production_plan('action', duration, intent=intent(duration))


def test_conflicting_structured_input_does_not_silently_override_duration():
    with pytest.raises(ValueError, match='时长'):
        build_production_plan('action', 30, intent=intent(15, 3))


@pytest.mark.parametrize('field', ['duration_evidence', 'shot_count_evidence'])
def test_new_constraints_require_verbatim_source_evidence(field):
    value = intent(15, 3)
    value[field] = 'invented'
    with pytest.raises(ValueError):
        DirectorIntentSpec.model_validate(value)


def test_intent_protocol_carries_explicit_shape():
    value = intent(30, 3)
    brief = value.pop('source_request')
    client = Responses(value)
    result = resolve_director_intent(client, brief)
    assert result['requested_duration_seconds'] == 30 and result['requested_shot_count'] == 3
    assert 'requested_shot_count' in str(client.calls[0]['response_format'])


def test_formal_planner_uses_semantic_duration_not_fallback_and_preserves_count(monkeypatch):
    from pipeline.storyboard import generate_storyboard
    from authoring_cases import scene_authoring_from_legacy
    value = intent(15, 3)
    brief = value.pop('source_request')
    client = Responses(value, scene_authoring_from_legacy(draft()))
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    story = generate_storyboard(brief, target_duration=30)
    assert len(story['shots']) == 3
    assert [s['duration'] for s in story['shots']] == [5, 5, 5]
    assert not production_plan_issues(story)
    altered = deepcopy(story)
    altered['production_plan']['director_intent']['requested_shot_count'] = 2
    assert production_plan_issues(altered)
    assert len(client.calls) == 5


def test_duration_hint_no_longer_silently_clamps_minutes():
    assert VideoPipeline._parse_duration('3分钟产品介绍') == 180
    assert VideoPipeline._parse_duration('300秒的实验过程') == 300


def test_old_frozen_plan_keeps_its_original_allocation():
    old = {'source_request': '旧需求', **intent_reply('旧需求')}
    plan = build_production_plan('action', 180, intent=old, contract_version='production-plan-v7-film-rhythm')
    assert plan['planned_duration'] == 150
    assert 'requested_duration_seconds' not in plan['director_intent']
