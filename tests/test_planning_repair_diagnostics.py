"""A malformed event plan must be repaired as events, never downgraded to legacy."""
import json
from copy import deepcopy

import pytest

from pipeline.action_timing import action_sequence_issues
from pipeline.readiness import storyboard_readiness_issues
from test_choreography_flow import flowing_story
from test_director_intent import draft, intent_reply, Responses


def test_formal_planner_accepts_explicitly_empty_noninteraction_geometry(monkeypatch):
    from pipeline.models import InteractionGeometrySpec
    from pipeline.storyboard import generate_storyboard
    request = '15秒连续交互'
    raw = draft()
    empty = InteractionGeometrySpec(effect_phase='none', outcome_scope='none', effect_motion='none').model_dump()
    for shot in raw['shots']:
        for beat in shot['action_beats']:
            if beat['activity'] != 'interaction':
                beat['interaction_geometry'] = deepcopy(empty)
    # Historical persisted full storyboards keep their input-boundary compatibility.
    from test_planning_no_repair import frozen_legacy_intent
    frozen_legacy_intent(monkeypatch, request)
    client = Responses(raw)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    result = generate_storyboard(request, target_duration=15)
    assert len(client.calls) == 1
    assert all(b.get('interaction_geometry') is None for s in result['shots']
               for b in s['action_beats'] if b['activity'] != 'interaction')
    assert raw['shots'][0]['action_beats'][0]['interaction_geometry'] == empty


@pytest.mark.parametrize('payload', [
    {'actor': 'attacker'}, {'target': 'defender'}, {'source': 'sword'},
    {'effect_phase': 'active'}, {'effect_motion': 'sweep'}, {'outcome_scope': 'single'},
    {'must_share_frame': True}, {'effect_region': 'contact area'},
])
def test_inactive_geometry_with_meaning_is_not_erased(payload):
    from pipeline.models import ActionBeatSpec, InteractionGeometrySpec
    empty = InteractionGeometrySpec(effect_phase='none', outcome_scope='none', effect_motion='none').model_dump()
    raw = {**draft()['shots'][0]['action_beats'][0], 'interaction_geometry': {**empty, **payload}}
    beat = ActionBeatSpec.model_validate(raw).model_dump()
    assert beat['interaction_geometry'] is not None
    assert any('不是交互' in i for i in action_sequence_issues({'shot_id': 1, 'action_beats': [beat]}))


@pytest.mark.parametrize('count', [3, 6])
def test_correction_receives_full_authoring_contract_for_each_batch(monkeypatch, count):
    from pipeline.storyboard import _correct_storyboard_draft
    story = draft()
    if count == 6:
        story['shots'] += deepcopy(story['shots'])
    for index, shot in enumerate(story['shots'], 1):
        shot['shot_id'] = index
    plan = {'slots': [{'shot_id': i, 'duration': 5} for i in range(1, count + 1)]}
    contract = 'AUTHORING_CONTRACT: a reaction belongs to its initiating event'
    monkeypatch.setattr('pipeline.storyboard._load_system_prompt', lambda: contract)
    monkeypatch.setattr('pipeline.storyboard._build_batch_correction_prompt', lambda *a: 'bounded correction')
    client = Responses(*[{'shots': deepcopy(story['shots'][i:i + 3])} for i in range(0, count, 3)])
    _correct_storyboard_draft(client, 'continuous interaction', story,
                             [f'Shot {i}: invalid interaction_geometry' for i in range(1, count + 1)], plan)
    assert len(client.calls) == count // 3
    for call in client.calls:
        assert contract in call['messages'][0]['content']
        assert '校正器' in call['messages'][0]['content']


def test_invalid_event_link_does_not_request_single_actor_downgrade():
    story = flowing_story()
    story['shots'][0]['action_beats'][0]['flow_from_previous'] = 'from prior shot'
    issues = storyboard_readiness_issues(story)
    assert any('flow_from_previous' in issue for issue in issues)
    assert not any('多个主动动作执行者' in issue for issue in issues)


def test_event_geometry_errors_identify_fields_for_automatic_repair():
    shot = flowing_story()['shots'][0]
    shot['action_beats'][0]['interaction_geometry'].update(source='', effect_region='', effect_phase='setup')
    issues = '\n'.join(action_sequence_issues(shot))
    assert 'source' in issues and 'effect_region' in issues and 'effect_phase' in issues


@pytest.mark.asyncio
async def test_formal_failed_planning_keeps_both_raw_attempts_and_checks(tmp_path, monkeypatch):
    import config
    from pipeline.orchestrator import VideoPipeline
    request = '15秒连续交互'
    bad = draft()
    bad['shots'][0]['action_beats'][1]['interaction_geometry']['source'] = ''
    # The legacy non-event authoring path still owns its bounded correction.
    from test_planning_no_repair import frozen_legacy_intent
    frozen_legacy_intent(monkeypatch, request)
    client = Responses(bad, deepcopy(bad))
    monkeypatch.setattr(config, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr('pipeline.storyboard.OpenAI', lambda **kw: client)
    monkeypatch.setattr('pipeline.orchestrator.VideoGenerator', lambda *a, **kw: pytest.fail('invalid plan reached provider'))
    pipeline = VideoPipeline(paid_take_budget=0)
    with pytest.raises(ValueError, match='自动修正 1 次后仍未通过'):
        await pipeline.run(request)
    assert len(client.calls) == 2
    for attempt in (1, 2):
        raw = json.loads((pipeline.workspace / f'planning_draft_{attempt}.json').read_text())
        check = json.loads((pipeline.workspace / f'planning_check_{attempt}.json').read_text())
        assert raw['shots'][0]['action_beats'][1]['interaction_geometry']['source'] == ''
        assert check['critical'] is True and any('source' in issue for issue in check['issues'])
    assert not (pipeline.workspace / 'storyboard.json').exists()
    assert not pipeline.run_workspace.manifest.paid_take_budget.reservations
